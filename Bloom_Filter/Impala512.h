// Copied from Apache Impala (incubating), usable under the terms in the Apache License,
// Version 2.0.

// This is a block Bloom filter (from Putze et al.'s "Cache-, Hash- and Space-Efficient
// Bloom Filters") with some twists:
//
// 1. Each block is a split Bloom filter - see Section 2.1 of Broder and Mitzenmacher's
// "Network Applications of Bloom Filters: A Survey".
//
// 2. The number of bits set per Add() is contant in order to take advantage of SIMD
// instructions.

#pragma once

#include <cstdint>
#include <cstdlib>

#include <algorithm>
#include <new>

#include "../hashutil.h"


using uint32_t = ::std::uint32_t;
using uint64_t = ::std::uint64_t;

namespace Impala {
    __attribute__((always_inline)) inline uint32_t reduce(uint32_t hash, uint32_t n) {
        // http://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/
        return (uint32_t) (((uint64_t) hash * n) >> 32);
    }

    __attribute__((always_inline)) inline uint64_t reduce64(uint64_t hash, uint64_t n) {
        // return hash % n;
        // http://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/
        return (uint64_t) (((__uint128_t) hash * (__uint128_t) n) >> 64);
        // return (uint32_t) (((uint64_t) hash * n) >> 32);
    }

    static inline uint64_t rotl64(uint64_t n, unsigned int c) {
        // assumes width is a power of 2
        const unsigned int mask = (CHAR_BIT * sizeof(n) - 1);
        // assert ( (c<=mask) &&"rotate by type width or more");
        c &= mask;
        return (n << c) | (n >> ((-c) & mask));
    }

    static inline size_t get_number_of_buckets(size_t max_items) {
        constexpr size_t log2_one_over_eps = 8;
        constexpr double overhead = 1.5225;
        constexpr size_t bucket_size_in_bits = 512;
        size_t blooms_m = std::ceil(max_items * log2_one_over_eps * overhead);
        size_t number_of_buckets = (blooms_m + bucket_size_in_bits - 1) / bucket_size_in_bits;
        return number_of_buckets;
    }
}// namespace Impala

#include <x86intrin.h>

template<typename HashFamily = ::hashing::TwoIndependentMultiplyShift>
class Impala512 {
private:
    // The filter is divided up into Buckets:
    using Bucket = uint64_t[8];

    const int bucketCount;

    Bucket *directory_;

    HashFamily hasher_;

public:
    // Consumes at most (1 << log_heap_space) bytes on the heap:
    explicit Impala512(const int bits);

    ~Impala512() noexcept;

    void Add(const uint64_t key) noexcept;

    bool Find(const uint64_t key) const noexcept;

    uint64_t SizeInBytes() const { return sizeof(Bucket) * bucketCount; }

    size_t get_cap() const noexcept {
        return -1;
    }
    float density() const noexcept {
        size_t set_bits = 0;
        for (int i = 0; i < bucketCount; i++) {
            uint64_t temp;
            memcpy(&temp, directory_+ i, 8);
            set_bits += _mm_popcnt_u64(temp);
        }
        float res = 1.0 * set_bits / (bucketCount * 64);
        return res;
    }

private:
    // A helper function for Insert()/Find(). Turns a 64-bit hash into a 512-bit Bucket
    // with 1 single 1-bit set in each 32-bit lane.
    static __m512i MakeMask(const uint64_t hash) noexcept;
};

template<typename HashFamily>
Impala512<HashFamily>::Impala512(const int n)
    : bucketCount(Impala::get_number_of_buckets(n)),
      directory_(nullptr),
      hasher_() {
    if (!__builtin_cpu_supports("avx2")) {
        throw ::std::runtime_error("Impala512 does not work without AVX2 instructions");
    }
    const size_t alloc_size = bucketCount * sizeof(Bucket);
    const int malloc_failed =
            posix_memalign(reinterpret_cast<void **>(&directory_), 64, alloc_size);
    if (malloc_failed) throw ::std::bad_alloc();
    // std::cout << "Ctor: SIMD-fixed byte size: " << SizeInBytes() << std::endl;
    memset(directory_, 0, alloc_size);
}

template<typename HashFamily>
Impala512<HashFamily>::~Impala512() noexcept {
    // std::cout << "Dtor: SIMD-fixed byte size: " << SizeInBytes() << std::endl;
    // std::cout << "density: " << density() << std::endl;
    free(directory_);
    directory_ = nullptr;
}

// The SIMD reinterpret_casts technically violate C++'s strict aliasing rules. However, we
// compile with -fno-strict-aliasing.
template<typename HashFamily>
[[gnu::always_inline]] inline __m512i
Impala512<HashFamily>::MakeMask(const uint64_t hash) noexcept {
    const __m512i ones = _mm512_set1_epi64(1);
    // Odd contants for hashing:
    const __m512i rehash = _mm512_setr_epi64(0x89cdc1c02b2352b9ULL,
                                             0x2b9aed3c5d9c5085ULL,
                                             0xfb087273c257911bULL,
                                             0x5ffd7847830af377ULL,
                                             0x287348157aed6753ULL,
                                             0x4e7292d5d251e97dULL,
                                             0xe00e4fc1185d71cbULL,
                                             0x4e3ebc18dc4c950bULL);
    // const __m512i rehash = _mm512_setr_epi64(0x47b6137bU, 0x44974d91U, 0x8824ad5bU, 0xa2b7289dU, 0x705495c7U, 0x2df1424bU, 0x9efc4947U, 0x5c6bfb31U);
    // Load hash into a YMM register, repeated eight times
    __m512i hash_data = _mm512_set1_epi64(hash);
    // Multiply-shift hashing ala Dietzfelbinger et al.: multiply 'hash' by eight different
    // odd constants, then keep the 5 most significant bits from each product.
    hash_data = _mm512_mullo_epi64(rehash, hash_data);
    hash_data = _mm512_srli_epi64(hash_data, 64 - 6);
    // Use these 5 bits to shift a single bit to a location in each 32-bit lane
    /* #ifndef DNDEBUG
    uint64_t a[8] = {0};
    memcpy(a, (uint64_t *) (&hash_data), 64);
    if ((hash & ((1ULL<<20)-1)) == 0){
        std::cout << "hash: " << hash << std::endl;
        for (size_t i = 0; i < 8; i++) {
            std::cout << "a[i]: " << a[i] << std::endl;
        }
    }
    auto temp = _mm512_sllv_epi64(ones, hash_data);
    
    // uint64_t a[8] = {0};
    memcpy(a, (uint64_t *) (&temp), 64);
    int s = 0;
    for (size_t i = 0; i < 8; i++) {
        auto block_temp = _mm_popcnt_u64(a[i]);
        assert(block_temp == 1);
        s += _mm_popcnt_u64(a[i]);
    }
    if (s != 8) {
        std::cout << "s: " << s << std::endl;
        for (size_t i = 0; i < 8; i++) {
            std::cout << "a[i]: " << a[i] << std::endl;
        }
    }
    assert(s == 8);
#endif
 */
    return _mm512_sllv_epi64(ones, hash_data);
}

template<typename HashFamily>
[[gnu::always_inline]] inline void
Impala512<HashFamily>::Add(const uint64_t key) noexcept {
    const auto hash = hasher_(key);
    const uint32_t bucket_idx = Impala::reduce(Impala::rotl64(hash, 32), bucketCount);
    // const uint32_t bucket_idx = Impala::reduce64(hash, bucketCount);
    const __m512i mask = MakeMask(hash);
    __m512i *const bucket = &reinterpret_cast<__m512i *>(directory_)[bucket_idx];
    _mm512_store_si512(bucket, _mm512_or_si512(*bucket, mask));
    assert(Find(key));
}

template<typename HashFamily>
[[gnu::always_inline]] inline bool
Impala512<HashFamily>::Find(const uint64_t key) const noexcept {
    const auto hash = hasher_(key);
    // const uint32_t bucket_idx = Impala::reduce64(hash, bucketCount);
    const uint32_t bucket_idx = Impala::reduce(Impala::rotl64(hash, 32), bucketCount);
    const __m512i mask = MakeMask(hash);
    const __m512i bucket = reinterpret_cast<__m512i *>(directory_)[bucket_idx];
    // We should return true if 'bucket' has a one wherever 'mask' does. _mm512_testc_si512
    // takes the negation of its first argument and ands that with its second argument. In
    // our case, the result is zero everywhere iff there is a one in 'bucket' wherever
    // 'mask' is one. testc returns 1 if the result is 0 everywhere and returns 0 otherwise.
    constexpr __m512i zero_vec = {0, 0, 0, 0, 0, 0, 0, 0};
    const __m512i res = _mm512_andnot_epi64(bucket, mask);
    return _mm512_cmpeq_epi64_mask(res, zero_vec) == 0xff;
    // const __m256i* array = reinterpret_cast<__m256i *>(&res);
}
