// parallel.h — 并行 for 的跨平台封装
//
// 引擎里只有一处需要"把 n 个独立下标分给多个核"，原来直接调 GCD 的
// dispatch_apply_f。Android 没有 GCD，所以在它前面加一层。
//
// 为什么 Apple 平台不换成自建线程池：过去几轮的基准（prefill 1.42x、
// decode 47→60 tok/s 那些）全部是在 GCD 上测的，换实现等于把历史数字作废，
// 而且 GCD 的全局队列由系统维护，长期实测比手写池稳。两套实现只藏在这个
// 函数背后，调用方看不到差异。
//
// 非 Apple 平台用常驻线程池 + 原子任务分发（动态负载均衡）。这里不用静态
// 分块，因为 matvec 各行耗时并不完全均匀（首行带 cache 冷启动），动态取
// 任务能把尾部拖慢的影响摊平。
#pragma once

#include <cstddef>

#if defined(__APPLE__)

#include <dispatch/dispatch.h>

namespace dubhe {

inline void parallel_for(size_t n, void (*fn)(void*, size_t), void* ctx) {
    if (n == 0) return;
    dispatch_apply_f(n, DISPATCH_APPLY_AUTO, ctx, fn);
}

}  // namespace dubhe

#else  // 非 Apple：自带线程池

#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <vector>

namespace dubhe {
namespace detail {

class Pool {
public:
    Pool() {
        int n = 0;
        if (const char* e = std::getenv("DUBHE_THREADS")) n = std::atoi(e);
        if (n <= 0) n = (int)std::thread::hardware_concurrency();
        if (n <= 0) n = 1;
        if (n > 64) n = 64;   // 再多只是互相抢带宽
        n_ = n;
        // 主线程也参与计算，所以只起 n-1 个 worker
        for (int i = 1; i < n_; i++) th_.emplace_back([this] { worker(); });
    }

    ~Pool() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& t : th_) t.join();
    }

    Pool(const Pool&) = delete;
    Pool& operator=(const Pool&) = delete;

    int size() const { return n_; }

    // 注意：fn 里不能抛异常。抛了就不会执行 --active_，下一次 run 会永久挂住。
    // 引擎全程不用异常，这里不做额外保护。
    void run(size_t n_items, void (*fn)(void*, size_t), void* ctx) {
        if (n_items == 0) return;
        if (n_items == 1 || th_.empty()) {
            for (size_t i = 0; i < n_items; i++) fn(ctx, i);
            return;
        }
        {
            std::lock_guard<std::mutex> lk(mu_);
            fn_ = fn;
            ctx_ = ctx;
            items_ = n_items;
            next_.store(0, std::memory_order_relaxed);
            active_ = (int)th_.size();
            ++gen_;
        }
        cv_.notify_all();
        drain();   // 主线程也干活，不白等
        std::unique_lock<std::mutex> lk(mu_);
        done_.wait(lk, [this] { return active_ == 0; });
    }

private:
    void drain() {
        for (;;) {
            const size_t i = next_.fetch_add(1, std::memory_order_relaxed);
            if (i >= items_) return;
            fn_(ctx_, i);
        }
    }

    void worker() {
        int seen = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [this, &seen] { return stop_ || gen_ != seen; });
                if (stop_) return;
                seen = gen_;
            }
            drain();
            {
                std::lock_guard<std::mutex> lk(mu_);
                if (--active_ == 0) done_.notify_one();
            }
        }
    }

    std::vector<std::thread> th_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::condition_variable done_;
    void (*fn_)(void*, size_t) = nullptr;
    void* ctx_ = nullptr;
    size_t items_ = 0;
    std::atomic<size_t> next_{0};
    int gen_ = 0;
    int active_ = 0;
    int n_ = 1;
    bool stop_ = false;
};

inline Pool& pool() {
    static Pool p;
    return p;
}

}  // namespace detail

inline void parallel_for(size_t n, void (*fn)(void*, size_t), void* ctx) {
    detail::pool().run(n, fn, ctx);
}

// 池里的线程数（含主线程）。只用于打印/调参，不影响正确性。
inline int parallel_threads() { return detail::pool().size(); }

}  // namespace dubhe

#endif  // __APPLE__
