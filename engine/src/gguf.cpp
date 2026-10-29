// gguf.cpp — GGUF v3 解析（mmap，零拷贝）
#include "internal.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace dubhe {

// ---------------------------------------------------------------- 类型表
struct TypeTraits { int bs; int bb; const char* name; };

static const TypeTraits kTraits[] = {
    /* 0  F32  */ {1, 4, "F32"},
    /* 1  F16  */ {1, 2, "F16"},
    /* 2  Q4_0 */ {32, 18, "Q4_0"},
    /* 3  Q4_1 */ {32, 20, "Q4_1"},
    /* 4  ---- */ {1, 0, "?"},
    /* 5  ---- */ {1, 0, "?"},
    /* 6  Q5_0 */ {32, 22, "Q5_0"},
    /* 7  Q5_1 */ {32, 24, "Q5_1"},
    /* 8  Q8_0 */ {32, 34, "Q8_0"},
    /* 9  Q8_1 */ {32, 36, "Q8_1"},
    /* 10 Q2_K */ {256, 84, "Q2_K"},
    /* 11 Q3_K */ {256, 110, "Q3_K"},
    /* 12 Q4_K */ {256, 144, "Q4_K"},
    /* 13 Q5_K */ {256, 176, "Q5_K"},
    /* 14 Q6_K */ {256, 210, "Q6_K"},
    /* 15 Q8_K */ {256, 292, "Q8_K"},
};

int type_block_size(uint32_t t) {
    if (t < 16) return kTraits[t].bs;
    return 1;
}
int type_block_bytes(uint32_t t) {
    if (t == T_BF16) return 2;
    if (t == T_I8) return 1;
    if (t == T_I16) return 2;
    if (t == T_I32) return 4;
    if (t == T_I64) return 8;
    if (t == T_F64) return 8;
    if (t < 16) return kTraits[t].bb;
    return 0;
}
size_t type_row_bytes(uint32_t t, int64_t n) {
    int bs = type_block_size(t);
    int bb = type_block_bytes(t);
    return (size_t)(n / bs) * (size_t)bb;
}
const char* type_name(uint32_t t) {
    if (t < 16) return kTraits[t].name;
    switch (t) {
        case T_BF16: return "BF16";
        case T_I8:   return "I8";
        case T_I16:  return "I16";
        case T_I32:  return "I32";
        case T_I64:  return "I64";
        case T_F64:  return "F64";
        default:     return "?";
    }
}
bool type_supported(uint32_t t) {
    switch (t) {
        case T_F32: case T_F16: case T_BF16:
        case T_Q4_0: case T_Q4_1: case T_Q5_0: case T_Q5_1: case T_Q8_0:
        case T_Q4_K: case T_Q5_K: case T_Q6_K:
            return true;
        default:
            return false;
    }
}

// ---------------------------------------------------------------- 读取器
namespace {

struct Reader {
    const uint8_t* p;
    const uint8_t* end;
    bool ok = true;

    bool need(size_t n) {
        if (p + n > end) { ok = false; return false; }
        return true;
    }
    template <typename T>
    T rd() {
        T v{};
        if (!need(sizeof(T))) return v;
        std::memcpy(&v, p, sizeof(T));
        p += sizeof(T);
        return v;
    }
    std::string rd_str() {
        uint64_t n = rd<uint64_t>();
        if (!ok || !need(n)) { ok = false; return {}; }
        std::string s((const char*)p, (size_t)n);
        p += n;
        return s;
    }
};

size_t align_up(size_t x, size_t a) { return (x + a - 1) / a * a; }

}  // namespace

// ---------------------------------------------------------------- GgufFile
GgufFile::~GgufFile() { close(); }

void GgufFile::close() {
    if (map_) {
        munmap(map_, file_size_);
        map_ = nullptr;
    }
    base_ = data_ = nullptr;
    file_size_ = data_size_ = 0;
    kv_.clear();
    tensors_.clear();
}

bool GgufFile::open(const char* path, std::string* err) {
    close();
    int fd = ::open(path, O_RDONLY);
    if (fd < 0) { if (err) *err = std::string("无法打开文件: ") + path; return false; }
    struct stat st {};
    if (fstat(fd, &st) != 0 || st.st_size <= 0) {
        ::close(fd);
        if (err) *err = "无法获取文件大小";
        return false;
    }
    file_size_ = (size_t)st.st_size;
    void* m = mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (m == MAP_FAILED) { if (err) *err = "mmap 失败"; return false; }
    map_ = m;
    base_ = (const uint8_t*)m;
    bool r = parse_header(err);
    if (!r) close();
    return r;
}

bool GgufFile::parse_header(std::string* err) {
    Reader r{base_, base_ + file_size_};
    uint32_t magic = r.rd<uint32_t>();
    if (magic != 0x46554747u) {  // "GGUF" 小端
        if (err) *err = "不是 GGUF 文件（magic 不匹配）";
        return false;
    }
    uint32_t version = r.rd<uint32_t>();
    if (version < 2 || version > 3) {
        if (err) *err = "不支持的 GGUF 版本: " + std::to_string(version);
        return false;
    }
    uint64_t n_tensors = r.rd<uint64_t>();
    uint64_t n_kv = r.rd<uint64_t>();

    // KV
    for (uint64_t i = 0; i < n_kv && r.ok; i++) {
        std::string key = r.rd_str();
        uint32_t type = r.rd<uint32_t>();
        GgufValue v;
        v.type = type;
        switch (type) {
            case 0: v.u64 = r.rd<uint8_t>(); break;
            case 1: v.i64 = r.rd<int8_t>(); break;
            case 2: v.u64 = r.rd<uint16_t>(); break;
            case 3: v.i64 = r.rd<int16_t>(); break;
            case 4: v.u64 = r.rd<uint32_t>(); break;
            case 5: v.i64 = r.rd<int32_t>(); break;
            case 6: v.f64 = r.rd<float>(); break;
            case 7: v.u64 = r.rd<uint8_t>() ? 1 : 0; break;
            case 8: v.str = r.rd_str(); break;
            case 10: v.u64 = r.rd<uint64_t>(); break;
            case 11: v.i64 = r.rd<int64_t>(); break;
            case 12: v.f64 = r.rd<double>(); break;
            case 9: {
                uint32_t et = r.rd<uint32_t>();
                uint64_t cnt = r.rd<uint64_t>();
                if (!r.ok) break;
                switch (et) {
                    case 0: case 7:
                        v.arr_u8.resize(cnt);
                        for (uint64_t k = 0; k < cnt && r.ok; k++) v.arr_u8[k] = r.rd<uint8_t>();
                        break;
                    case 1:
                        v.arr_i32.resize(cnt);
                        for (uint64_t k = 0; k < cnt && r.ok; k++) v.arr_i32[k] = r.rd<int8_t>();
                        break;
                    case 2: case 3:
                        v.arr_i32.resize(cnt);
                        for (uint64_t k = 0; k < cnt && r.ok; k++) v.arr_i32[k] = (et == 2) ? (int32_t)r.rd<uint16_t>() : (int32_t)r.rd<int16_t>();
                        break;
                    case 4: case 5:
                        v.arr_i32.resize(cnt);
                        for (uint64_t k = 0; k < cnt && r.ok; k++) v.arr_i32[k] = (et == 4) ? (int32_t)r.rd<uint32_t>() : r.rd<int32_t>();
                        break;
                    case 6:
                        v.arr_f32.resize(cnt);
                        for (uint64_t k = 0; k < cnt && r.ok; k++) v.arr_f32[k] = r.rd<float>();
                        break;
                    case 8:
                        v.arr_str.reserve(cnt);
                        for (uint64_t k = 0; k < cnt && r.ok; k++) v.arr_str.push_back(r.rd_str());
                        break;
                    case 10: case 11:
                        v.arr_i64.resize(cnt);
                        for (uint64_t k = 0; k < cnt && r.ok; k++)
                            v.arr_i64[k] = (et == 10) ? (int64_t)r.rd<uint64_t>() : r.rd<int64_t>();
                        break;
                    case 12:
                        v.arr_f32.resize(cnt);
                        for (uint64_t k = 0; k < cnt && r.ok; k++) v.arr_f32[k] = (float)r.rd<double>();
                        break;
                    default:
                        if (err) *err = "未知的数组元素类型: " + std::to_string(et);
                        return false;
                }
                break;
            }
            default:
                if (err) *err = "未知的元数据类型: " + std::to_string(type) + " (key=" + key + ")";
                return false;
        }
        if (!r.ok) { if (err) *err = "元数据读取越界 (key=" + key + ")"; return false; }
        kv_.emplace(std::move(key), std::move(v));
    }

    // 张量表
    tensors_.reserve(n_tensors);
    for (uint64_t i = 0; i < n_tensors && r.ok; i++) {
        TensorInfo t;
        t.name = r.rd_str();
        t.n_dims = (int)r.rd<uint32_t>();
        if (t.n_dims < 1 || t.n_dims > 4) { if (err) *err = "非法维度数"; return false; }
        for (int d = 0; d < t.n_dims; d++) t.ne[d] = (int64_t)r.rd<uint64_t>();
        t.type = r.rd<uint32_t>();
        t.offset = r.rd<uint64_t>();
        if (!r.ok) { if (err) *err = "张量表读取越界"; return false; }
        tensors_.push_back(std::move(t));
    }
    if (!r.ok) { if (err) *err = "头部解析失败"; return false; }

    size_t alignment = 32;
    auto it = kv_.find("general.alignment");
    if (it != kv_.end() && it->second.u64 > 0) alignment = (size_t)it->second.u64;

    size_t off = align_up((size_t)(r.p - base_), alignment);
    if (off > file_size_) { if (err) *err = "数据段偏移越界"; return false; }
    data_ = base_ + off;
    data_size_ = file_size_ - off;
    return true;
}

const GgufValue* GgufFile::get(const std::string& key) const {
    auto it = kv_.find(key);
    return it == kv_.end() ? nullptr : &it->second;
}
bool GgufFile::has(const std::string& key) const { return kv_.count(key) != 0; }

std::string GgufFile::get_str(const std::string& key, const std::string& def) const {
    auto* v = get(key);
    return (v && v->type == 8) ? v->str : def;
}
uint32_t GgufFile::get_u32(const std::string& key, uint32_t def) const {
    auto* v = get(key);
    if (!v) return def;
    if (v->type == 8) return def;
    return (uint32_t)(v->u64 ? v->u64 : (uint64_t)v->i64);
}
float GgufFile::get_f32(const std::string& key, float def) const {
    auto* v = get(key);
    if (!v) return def;
    return v->f64 != 0.0 ? (float)v->f64 : (float)(v->i64 ? v->i64 : (int64_t)v->u64);
}

const std::vector<int32_t>* GgufFile::get_arr_i32(const std::string& key) const {
    auto* v = get(key);
    return (v && !v->arr_i32.empty()) ? &v->arr_i32 : nullptr;
}
const std::vector<uint8_t>* GgufFile::get_arr_u8(const std::string& key) const {
    auto* v = get(key);
    return (v && !v->arr_u8.empty()) ? &v->arr_u8 : nullptr;
}

const TensorInfo* GgufFile::find_tensor(const std::string& name) const {
    for (auto& t : tensors_)
        if (t.name == name) return &t;
    return nullptr;
}

const void* GgufFile::tensor_data(const TensorInfo& t) const {
    if (!data_ || t.offset >= data_size_) return nullptr;
    return data_ + t.offset;
}

}  // namespace dubhe
