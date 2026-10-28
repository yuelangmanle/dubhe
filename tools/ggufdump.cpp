// ggufdump.cpp — GGUF 元数据/张量表转储，用于排查加载问题
#include "../engine/src/internal.h"

#include <cstdio>
#include <string>

using namespace dubhe;

int main(int argc, char** argv) {
    if (argc < 2) { std::printf("用法: ggufdump <model.gguf> [key前缀]\n"); return 1; }
    GgufFile f;
    std::string err;
    if (!f.open(argv[1], &err)) { std::printf("打开失败: %s\n", err.c_str()); return 1; }
    std::printf("文件 %.2f MB · %zu 个张量\n\n", (double)f.file_size() / 1048576.0, f.tensors().size());

    for (auto& kv : f.kv()) {
        if (argc > 2 && kv.first.rfind(argv[2], 0) != 0) continue;
        const auto& v = kv.second;
        std::printf("%-52s type=%-2u ", kv.first.c_str(), v.type);
        switch (v.type) {
            case 0: case 2: case 4: case 10: std::printf("u=%llu", (unsigned long long)v.u64); break;
            case 1: case 3: case 5: case 11: std::printf("i=%lld", (long long)v.i64); break;
            case 6: case 12: std::printf("f=%g", v.f64); break;
            case 7: std::printf("b=%llu", (unsigned long long)v.u64); break;
            case 8: std::printf("s=\"%s\"", v.str.size() > 60 ? v.str.substr(0, 60).c_str() : v.str.c_str()); break;
            case 9:
                std::printf("[%zu] ", v.arr_str.size() + v.arr_i32.size() + v.arr_f32.size() + v.arr_i64.size() + v.arr_u8.size());
                if (!v.arr_str.empty()) {
                    for (size_t i = 0; i < v.arr_str.size() && i < 6; i++)
                        std::printf("\"%s\" ", v.arr_str[i].c_str());
                } else if (!v.arr_i32.empty()) {
                    for (size_t i = 0; i < v.arr_i32.size() && i < 10; i++) std::printf("%d ", v.arr_i32[i]);
                } else if (!v.arr_f32.empty()) {
                    for (size_t i = 0; i < v.arr_f32.size() && i < 6; i++) std::printf("%g ", v.arr_f32[i]);
                }
                break;
            default: break;
        }
        std::printf("\n");
    }

    std::printf("\n--- 张量 ---\n");
    int shown = 0;
    for (auto& t : f.tensors()) {
        if (argc > 2 && shown >= 12) break;
        std::printf("%-40s %-6s ne=[%lld,%lld,%lld] off=%llu\n", t.name.c_str(), type_name(t.type),
                    (long long)t.ne[0], (long long)t.ne[1], (long long)t.ne[2],
                    (unsigned long long)t.offset);
        shown++;
    }
    return 0;
}
