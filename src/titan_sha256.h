#ifndef TITAN_SHA256_H
#define TITAN_SHA256_H

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace titan_hash {

class Sha256Builder {
public:
  Sha256Builder() = default;

  void update(std::span<const std::byte> bytes) noexcept {
    for (const std::byte value : bytes)
      byte(std::to_integer<std::uint8_t>(value));
  }

  void update(std::string_view text) noexcept {
    for (const char value : text)
      byte(static_cast<std::uint8_t>(static_cast<unsigned char>(value)));
  }

  void byte(std::uint8_t value) noexcept {
    buffer_[buffer_size_++] = value;
    ++total_bytes_;
    if (buffer_size_ == buffer_.size()) {
      transform(buffer_);
      buffer_size_ = 0;
    }
  }

  void word(std::uint64_t value) noexcept {
    for (unsigned shift = 0; shift < 64; shift += 8)
      byte(static_cast<std::uint8_t>(value >> shift));
  }

  void real(double value) noexcept {
    if (value == 0.0)
      value = 0.0;
    word(std::bit_cast<std::uint64_t>(value));
  }

  [[nodiscard]] std::array<std::byte, 32> finish256() const noexcept {
    Sha256Builder copy = *this;
    const std::uint64_t bit_length = copy.total_bytes_ * 8U;
    copy.buffer_[copy.buffer_size_++] = 0x80U;
    if (copy.buffer_size_ > 56) {
      while (copy.buffer_size_ < 64)
        copy.buffer_[copy.buffer_size_++] = 0;
      copy.transform(copy.buffer_);
      copy.buffer_size_ = 0;
    }
    while (copy.buffer_size_ < 56)
      copy.buffer_[copy.buffer_size_++] = 0;
    for (int shift = 56; shift >= 0; shift -= 8) {
      copy.buffer_[copy.buffer_size_++] =
          static_cast<std::uint8_t>(bit_length >> shift);
    }
    copy.transform(copy.buffer_);

    std::array<std::byte, 32> digest{};
    for (std::size_t idx = 0; idx < copy.state_.size(); ++idx) {
      const std::uint32_t value = copy.state_[idx];
      for (unsigned byte_idx = 0; byte_idx < 4; ++byte_idx) {
        digest[4 * idx + byte_idx] = static_cast<std::byte>(
            value >> (24U - 8U * byte_idx));
      }
    }
    return digest;
  }

  // Pair order is {lo, hi}; formatting hi followed by lo yields the first
  // 128 bits of the standard SHA-256 hexadecimal digest.
  [[nodiscard]] std::array<std::uint64_t, 2> finish128() const noexcept {
    const auto digest = finish256();
    const auto load_be = [&](std::size_t offset) {
      std::uint64_t value = 0;
      for (std::size_t idx = 0; idx < 8; ++idx) {
        value = (value << 8) |
                std::to_integer<std::uint8_t>(digest[offset + idx]);
      }
      return value;
    };
    return {load_be(8), load_be(0)};
  }

private:
  static constexpr std::array<std::uint32_t, 64> kRoundConstants{
      0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
      0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
      0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
      0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
      0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
      0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
      0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
      0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
      0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
      0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
      0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
      0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
      0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
      0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
      0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
      0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};

  static constexpr std::uint32_t rotate_right(std::uint32_t value,
                                               unsigned shift) noexcept {
    return (value >> shift) | (value << (32U - shift));
  }

  void transform(const std::array<std::uint8_t, 64> &block) noexcept {
    std::array<std::uint32_t, 64> words{};
    for (std::size_t idx = 0; idx < 16; ++idx) {
      words[idx] = (static_cast<std::uint32_t>(block[4 * idx]) << 24) |
                   (static_cast<std::uint32_t>(block[4 * idx + 1]) << 16) |
                   (static_cast<std::uint32_t>(block[4 * idx + 2]) << 8) |
                   static_cast<std::uint32_t>(block[4 * idx + 3]);
    }
    for (std::size_t idx = 16; idx < words.size(); ++idx) {
      const std::uint32_t s0 =
          rotate_right(words[idx - 15], 7) ^
          rotate_right(words[idx - 15], 18) ^ (words[idx - 15] >> 3);
      const std::uint32_t s1 =
          rotate_right(words[idx - 2], 17) ^
          rotate_right(words[idx - 2], 19) ^ (words[idx - 2] >> 10);
      words[idx] = words[idx - 16] + s0 + words[idx - 7] + s1;
    }

    std::uint32_t a = state_[0];
    std::uint32_t b = state_[1];
    std::uint32_t c = state_[2];
    std::uint32_t d = state_[3];
    std::uint32_t e = state_[4];
    std::uint32_t f = state_[5];
    std::uint32_t g = state_[6];
    std::uint32_t h = state_[7];
    for (std::size_t idx = 0; idx < words.size(); ++idx) {
      const std::uint32_t sum1 =
          rotate_right(e, 6) ^ rotate_right(e, 11) ^ rotate_right(e, 25);
      const std::uint32_t choice = (e & f) ^ (~e & g);
      const std::uint32_t temp1 =
          h + sum1 + choice + kRoundConstants[idx] + words[idx];
      const std::uint32_t sum0 =
          rotate_right(a, 2) ^ rotate_right(a, 13) ^ rotate_right(a, 22);
      const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
      const std::uint32_t temp2 = sum0 + majority;
      h = g;
      g = f;
      f = e;
      e = d + temp1;
      d = c;
      c = b;
      b = a;
      a = temp1 + temp2;
    }
    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
  }

  std::array<std::uint32_t, 8> state_{
      0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
      0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
  std::array<std::uint8_t, 64> buffer_{};
  std::size_t buffer_size_ = 0;
  std::uint64_t total_bytes_ = 0;
};

} // namespace titan_hash

#endif
