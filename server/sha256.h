#pragma once

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

namespace storage {

class Sha256 {
public:
    void update(const void* data, size_t size) {
        const auto* bytes = static_cast<const uint8_t*>(data);
        total_ += size;
        if (used_) {
            const size_t take = size < block_.size() - used_ ? size : block_.size() - used_;
            std::memcpy(block_.data() + used_, bytes, take);
            used_ += take;
            bytes += take;
            size -= take;
            if (used_ == block_.size()) {
                transform(block_.data());
                used_ = 0;
            }
        }
        while (size >= block_.size()) {
            transform(bytes);
            bytes += block_.size();
            size -= block_.size();
        }
        if (size) {
            std::memcpy(block_.data(), bytes, size);
            used_ = size;
        }
    }

    std::string finish() const {
        Sha256 final = *this;
        const uint64_t bit_length = total_ * 8;
        uint8_t padding[128] = {0x80};
        const size_t pad = used_ < 56 ? 56 - used_ : 120 - used_;
        for (int i = 0; i < 8; ++i) padding[pad + i] = (uint8_t)(bit_length >> (56 - i * 8));
        final.update(padding, pad + 8);
        std::string hex;
        for (uint32_t word : final.state_) {
            for (int shift = 28; shift >= 0; shift -= 4)
                hex += "0123456789abcdef"[(word >> shift) & 15];
        }
        return hex;
    }

private:
    void transform(const uint8_t* bytes) {
        static constexpr uint32_t constants[64] = {
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
            0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
            0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
            0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
        };
        uint32_t words[64];
        for (int i = 0; i < 16; ++i) {
            const uint8_t* p = bytes + 4 * i;
            words[i] = (uint32_t)p[0] << 24 | (uint32_t)p[1] << 16 | (uint32_t)p[2] << 8 | p[3];
        }
        for (int i = 16; i < 64; ++i) {
            const uint32_t u = words[i - 15], v = words[i - 2];
            const uint32_t low = std::rotr(u, 7) ^ std::rotr(u, 18) ^ (u >> 3);
            const uint32_t high = std::rotr(v, 17) ^ std::rotr(v, 19) ^ (v >> 10);
            words[i] = words[i - 16] + low + words[i - 7] + high;
        }
        auto [a, b, c, d, e, f, g, h] = state_;
        for (int i = 0; i < 64; ++i) {
            const uint32_t high = std::rotr(e, 6) ^ std::rotr(e, 11) ^ std::rotr(e, 25);
            const uint32_t choose = (e & f) ^ (~e & g);
            const uint32_t low = std::rotr(a, 2) ^ std::rotr(a, 13) ^ std::rotr(a, 22);
            const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t sum = h + high + choose + constants[i] + words[i];
            h = g; g = f; f = e; e = d + sum;
            d = c; c = b; b = a; a = sum + low + majority;
        }
        state_[0] += a; state_[1] += b; state_[2] += c; state_[3] += d;
        state_[4] += e; state_[5] += f; state_[6] += g; state_[7] += h;
    }

    std::array<uint32_t, 8> state_ = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                      0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    std::array<uint8_t, 64> block_{};
    size_t used_ = 0;
    uint64_t total_ = 0;
};

}  // namespace storage
