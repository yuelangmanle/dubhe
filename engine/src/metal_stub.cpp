// metal_stub.cpp — 非 Apple 平台的 Metal 空实现
//
// 目标平台是 macOS / iOS / Android。Android 走的是 NEON + 各芯片 NPU 后端，
// 没有 Metal，所以这里提供一组恒失败的桩，让 Model 自然落到 CPU 路径。
#include "metal.h"

#if !defined(__APPLE__)

namespace dubhe {
namespace metal {

bool available() { return false; }
bool init(std::string* err) { if (err) *err = "非 Apple 平台没有 Metal"; return false; }
void shutdown() {}
const char* device_name() { return ""; }
const char* last_error() { return "Metal 不可用"; }
bool bind_weights(const void*, size_t, std::string* err) {
    if (err) *err = "Metal 不可用";
    return false;
}
void unbind_weights() {}
bool supports(uint32_t, int) { return false; }
bool mfvec(uint32_t, const float*, float*, int, int, uint64_t) { return false; }

bool fwd_init(const Spec&, std::string* err) { if (err) *err = "Metal 不可用"; return false; }
bool fwd_grow_kv(int, std::string* err) { if (err) *err = "Metal 不可用"; return false; }
bool fwd_token(const float*, int, float*, std::string* err) {
    if (err) *err = "Metal 不可用";
    return false;
}
bool fwd_batch(const float*, int, int, float*, std::string* err) {
    if (err) *err = "Metal 不可用";
    return false;
}
bool fwd_multi(const float*, int, const int*, const uint64_t*, const uint64_t*,
               float*, std::string* err) {
    if (err) *err = "Metal 不可用";
    return false;
}
bool fwd_batch_at(const float*, int, int, uint64_t, uint64_t, float*, std::string* err) {
    if (err) *err = "Metal 不可用";
    return false;
}
int fwd_batch_max() { return 1; }
void fwd_reset_conv() {}
bool fwd_copy_x(float*, int) { return false; }

}  // namespace metal
}  // namespace dubhe

#endif  // !__APPLE__
