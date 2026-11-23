#include "../server/store.h"

#include <cstdio>
#include <stdexcept>

namespace {

void require(bool pass, const char* message) {
    if (!pass) throw std::runtime_error(message);
}

struct TempDir {
    char path[32] = "/tmp/dubhe-store-XXXXXX";
    TempDir() { require(mkdtemp(path), "mkdtemp failed"); }
    ~TempDir() {
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
    }
};

std::string final_state(store::Downloads& manager, const std::string& id) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(8);
    while (std::chrono::steady_clock::now() < deadline) {
        auto status = manager.status();
        for (const auto& item : status.get("items")->arr) {
            if (item.s("id") != id) continue;
            const auto state = item.s("state");
            if (state == "done" || state == "error" || state == "cancelled") return state;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return "timeout";
}

json::Value status_item(store::Downloads& manager, const std::string& id) {
    const auto result = manager.status();
    for (const auto& item : result.get("items")->arr)
        if (item.s("id") == id) return item;
    throw std::runtime_error("catalog item missing");
}

}  // namespace

int main() {
    try {
        storage::Sha256 abc;
        abc.update("a", 1);
        abc.update("bc", 2);
        require(abc.finish() == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
                "SHA-256 abc mismatch");
        storage::Sha256 long_input;
        const std::string run(1000, 'a');
        for (int i = 0; i < 1000; ++i) long_input.update(run.data(), run.size());
        require(long_input.finish() == "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0",
                "SHA-256 long-stream mismatch");

        TempDir root;
        const std::string dir = std::string(root.path) + "/models";
        require(storage::ensure_dir(dir), "model folder creation failed");
        const std::string file("GGUF\x03\0\0\0", 8);
        storage::Sha256 hash;
        hash.update(file.data(), file.size());
        const auto expected = hash.finish();
        const std::string tower = file + "V";
        storage::Sha256 tower_hash;
        tower_hash.update(tower.data(), tower.size());
        const auto expected_tower = tower_hash.finish();
        const store::Entry fixtures[] = {
            {"matching", "Fixture", "", "", 8, "valid.gguf", "", expected.c_str()},
            {"changed", "Fixture", "", "", 8, "bad.gguf", "",
             "0000000000000000000000000000000000000000000000000000000000000000"},
            {"bundle", "Fixture", "", "", 8, "vision.gguf", "", expected.c_str(), nullptr, false, false,
             {9, "mmproj-vision.gguf", "", expected_tower.c_str()}, "image"},
        };
        store::Downloads manager(dir, std::string(root.path) + "/store.json", fixtures);
        require(storage::write_bytes(dir + "/valid.gguf.part", file), "valid fixture write failed");
        require(storage::write_bytes(dir + "/bad.gguf.part", file), "bad fixture write failed");
        require(manager.download("matching").status == 200, "resume of full file failed");
        require(manager.download("changed").status == 200, "corrupt fixture did not enter verification");
        require(final_state(manager, "matching") == "done", "verified file did not install");
        require(final_state(manager, "changed") == "error", "wrong digest accepted");
        require(store::file_size(dir + "/valid.gguf") == 8 &&
                store::file_size(dir + "/valid.gguf.part") < 0, "publish state inconsistent");
        require(store::file_size(dir + "/bad.gguf") < 0, "bad file reached the model directory");
        require(manager.save_mirror("https://mirror.example") &&
                manager.mirror() == "https://mirror.example", "mirror settings round trip failed");

        const std::string bundle_dir = dir + "/bundle";
        const std::string weights_path = bundle_dir + "/vision.gguf";
        const std::string tower_path = bundle_dir + "/mmproj-vision.gguf";
        require(storage::ensure_dir(bundle_dir), "bundle folder creation failed");
        require(storage::write_bytes(weights_path + ".part", file), "bundle weights write failed");
        require(storage::write_bytes(tower_path + ".part", file + "X"), "bad tower write failed");
        auto bundle = status_item(manager, "bundle");
        require(bundle.n("got") == 17 && bundle.n("total") == 17 && bundle.s("modality") == "image",
                "bundle status did not include both files");
        require(manager.download("bundle").status == 200 && final_state(manager, "bundle") == "error",
                "bundle accepted the wrong projector hash");
        require(store::file_size(weights_path) < 0 && store::file_size(weights_path + ".part") == 8,
                "incomplete bundle was published or verified weights were lost");
        require(storage::write_bytes(tower_path + ".part", tower), "tower write failed");
        require(manager.clear_partial("bundle").status == 200 &&
                store::file_size(weights_path + ".part") < 0 && store::file_size(tower_path + ".part") < 0,
                "bundle partial cleanup did not clear both files");
        require(storage::write_bytes(weights_path + ".part", file) && storage::write_bytes(tower_path + ".part", tower),
                "bundle restore failed");
        require(manager.download("bundle").status == 200 && final_state(manager, "bundle") == "done" &&
                store::file_size(weights_path) == 8 && store::file_size(tower_path) == 9,
                "complete bundle did not install");
        const auto removed = manager.remove_bundle(weights_path);
        require(removed && removed->status == 200 && store::file_size(weights_path) < 0 && store::file_size(tower_path) < 0,
                "bundle removal left a model or component");
        std::puts("商店续传校验、原子安装、配套组件和下载源设置通过");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
