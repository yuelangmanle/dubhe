// sysinfo.h — 硬件信息查询（CPU 名 / 内存 / 核数）
//
// 原来这三样直接写在 engine.cpp 里调 sysctlbyname，Android 上编不过。
// 抽出来按平台分支：Apple 走 sysctl，Android/Linux 读 /proc。
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

namespace dubhe {

// SoC / CPU 名。取不到就返回一个能看的兜底串。
inline std::string cpu_brand() {
#if defined(__APPLE__)
    char brand[256] = {0};
    size_t len = sizeof(brand);
    if (sysctlbyname("machdep.cpu.brand_string", brand, &len, nullptr, 0) != 0)
        return "Apple Silicon";
    return brand;
#else
    // ARM 上 /proc/cpuinfo 的 "model name" 常常为空，SoC 名在 "Hardware" 行，
    // 形如 "Qualcomm Technologies, Inc SM8750"。两个都试。
    std::FILE* f = std::fopen("/proc/cpuinfo", "r");
    if (!f) return "CPU";
    char line[512];
    std::string model, hardware;
    while (std::fgets(line, sizeof(line), f)) {
        if (!model.empty() && !hardware.empty()) break;
        char* colon = std::strchr(line, ':');
        if (!colon) continue;
        std::string key(line, (size_t)(colon - line));
        std::string val(colon + 1);
        while (!val.empty() && (val.back() == '\n' || val.back() == ' ')) val.pop_back();
        const size_t b = val.find_first_not_of(" \t");
        if (b == std::string::npos) continue;
        val = val.substr(b);
        if (key.find("model name") != std::string::npos) model = val;
        else if (key.find("Hardware") != std::string::npos) hardware = val;
    }
    std::fclose(f);
    if (!hardware.empty()) return hardware;
    if (!model.empty()) return model;
    return "CPU";
#endif
}

// 物理内存总量（字节）。取不到返回 0。
inline uint64_t total_ram_bytes() {
#if defined(__APPLE__)
    uint64_t mem = 0;
    size_t len = sizeof(mem);
    if (sysctlbyname("hw.memsize", &mem, &len, nullptr, 0) != 0) return 0;
    return mem;
#else
    std::FILE* f = std::fopen("/proc/meminfo", "r");
    if (!f) return 0;
    char line[256];
    uint64_t kb = 0;
    while (std::fgets(line, sizeof(line), f)) {
        if (std::sscanf(line, "MemTotal: %llu kB", (unsigned long long*)&kb) == 1) break;
    }
    std::fclose(f);
    return kb * 1024ull;
#endif
}

// 可用核数。Apple 优先返回性能核数（hw.perflevel0），因为 E 核跑 matvec
// 会拖慢整体；Android 上返回在线核数，大小核调度交给系统。
inline int cpu_count() {
#if defined(__APPLE__)
    int n = 0;
    size_t len = sizeof(n);
    if (sysctlbyname("hw.perflevel0.logicalcpu", &n, &len, nullptr, 0) == 0 && n > 0) return n;
    len = sizeof(n);
    if (sysctlbyname("hw.ncpu", &n, &len, nullptr, 0) == 0 && n > 0) return n;
    return 4;
#else
    const unsigned hc = std::thread::hardware_concurrency();
    return hc > 0 ? (int)hc : 4;
#endif
}

// 总核数（含能效核）。给"这台机器是什么"用，不给线程池用。
inline int total_cpu_count() {
#if defined(__APPLE__)
    int n = 0;
    size_t len = sizeof(n);
    if (sysctlbyname("hw.ncpu", &n, &len, nullptr, 0) == 0 && n > 0) return n;
    return cpu_count();
#else
    return cpu_count();
#endif
}

// 人读的硬件摘要。Apple 是统一内存，其它平台就是普通内存。
//
// Apple 上核数要说清两档：cpu_count() 给的是性能核（线程池只用这些，
// 见上面的注释），拿它当"这台机器几核"写出来就是错的 —— M4 会被写成
// "4 核"，而它其实是 4 性能 + 6 能效。摘要里两个都给。
inline std::string hardware_summary() {
    const std::string brand = cpu_brand();
    const int ncpu = cpu_count();
    const int ntotal = total_cpu_count();
    const uint64_t mem = total_ram_bytes();
    char buf[640];
#if defined(__APPLE__)
    if (ntotal > ncpu) {
        std::snprintf(buf, sizeof(buf), "%s · %d 核（%d 性能 + %d 能效）· %.0f GB 统一内存",
                      brand.c_str(), ntotal, ncpu, ntotal - ncpu,
                      (double)mem / (1024.0 * 1024.0 * 1024.0));
        return std::string(buf);
    }
#endif
    std::snprintf(buf, sizeof(buf), "%s · %d 核 · %.0f GB %s", brand.c_str(), ntotal,
                  (double)mem / (1024.0 * 1024.0 * 1024.0),
#if defined(__APPLE__)
                  "统一内存"
#else
                  "内存"
#endif
    );
    return std::string(buf);
}

}  // namespace dubhe
