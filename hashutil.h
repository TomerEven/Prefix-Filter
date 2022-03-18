
#ifndef HASHUTIL_H_
#define HASHUTIL_H_

#include <climits>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdint.h>
#include <stdlib.h>
#include <string>
#include <sys/types.h>
// #include "Hash_functions/BobJenkins.h"
// #include "Hash_functions/wyhash.h"
// #include "Hash_functions/xxhash64.h"
#include <assert.h>
#include <random>
#include <vector>

#include <immintrin.h>
#include <x86intrin.h>

// #include "Hash_functions/woothash.h"

namespace hashing {
    inline size_t sysrandom(void *dst, size_t dstlen) {
        char *buffer = reinterpret_cast<char *>(dst);
        std::ifstream stream("/dev/urandom", std::ios_base::binary | std::ios_base::in);
        stream.read(buffer, dstlen);

        return dstlen;
    }

    // See Martin Dietzfelbinger, "Universal hashing and k-wise independent random
    // variables via integer arithmetic without primes".
    class TwoIndependentMultiplyShift {
        unsigned __int128 multiply_, add_;

    public:
        TwoIndependentMultiplyShift() {
            std::uint_least64_t seed;
            sysrandom(&seed, sizeof(seed));
            std::mt19937_64 rng(seed);
            std::uniform_int_distribution<uint64_t> dist(0, UINT64_MAX);

            for (auto v : {&multiply_, &add_}) {
                for (size_t i = 0; i < 8; i++) {
                    unsigned __int128 hi = dist(rng);
                    unsigned __int128 lo = dist(rng);
                    *v ^= (hi << 64) | lo;
                }
            }
        }
        /*  // ::std::random_device random;
            // for (auto v : {&multiply_, &add_}) {
            //     *v = random();
            //     for (int i = 1; i <= 4; ++i) {
            //         *v = *v << 32;
            //         *v |= random();
            //     }
            // }
        }

        // 
        //  * @brief Construct a new Two Independent Multiply Shift object
        //  * Disable the randomness for debugging.
        //  *
        //  * @param seed1 Garbage
        //  * @param seed2 Garbage
        // 
        // TwoIndependentMultiplyShift(unsigned __int128 seed1, unsigned __int128 seed2) {
        //     std::cout << "hash function is pseudo random" << std::endl;

        //     multiply_ = 0xaaaa'bbbb'cccc'dddd;
        //     multiply_ <<= 64;
        //     multiply_ |= 0xeeee'ffff'1111'0000;
        //     add_ = 0xaaaa'aaaa'bbbb'bbbb;
        //     add_ <<= 64;
        //     add_ |= 0xcccc'cccc'dddd'dddd;

        //     assert(multiply_ > 18446744073709551615ULL);
        //     assert(add_ > 18446744073709551615ULL);
        // } */

        inline uint64_t operator()(uint64_t key) const {
            return (add_ + multiply_ * static_cast<decltype(multiply_)>(key)) >> 64;
        }

        inline uint32_t hash32(uint64_t key) const {
            return ((uint32_t) (add_ + multiply_ * static_cast<decltype(multiply_)>(key)));
        }
        auto get_name() const -> std::string {
            return "TwoIndependentMultiplyShift";
        }
    };


    /**
 * @brief Like others, only with redundancies, to produce more operations.
 * 
 */
    class IdioticHash {
        unsigned __int128 multiply_, add_;
        unsigned __int128 multiply_2, add_2;

    public:
        IdioticHash() {
            std::uint_least64_t seed;
            sysrandom(&seed, sizeof(seed));
            std::mt19937_64 rng(seed);
            std::uniform_int_distribution<uint64_t> dist(0, UINT64_MAX);

            for (auto v : {&multiply_, &add_}) {
                for (size_t i = 0; i < 8; i++) {
                    unsigned __int128 hi = dist(rng);
                    unsigned __int128 lo = dist(rng);
                    *v ^= (hi << 64) | lo;
                }
            }

            for (auto v : {&multiply_2, &add_2}) {
                for (size_t i = 0; i < 8; i++) {
                    unsigned __int128 hi = dist(rng);
                    unsigned __int128 lo = dist(rng);
                    *v ^= (hi << 64) | lo;
                }
            }
        }

        static inline uint64_t select64(uint64_t x, int64_t j) {
            assert(j < 64);
            const uint64_t y = _pdep_u64(UINT64_C(1) << j, x);
            return _tzcnt_u64(y);
        }

        inline uint64_t operator()(uint64_t key) const {
            auto res1 = (add_ + multiply_ * static_cast<decltype(multiply_)>(key)) >> 64;
            auto res2 = (add_2 + multiply_2 * static_cast<decltype(multiply_2)>(key)) >> 64;

            size_t index = _mm_popcnt_u64(key);
            size_t pos = select64(key, index / 2);
            if (pos & 1)
                return res1;
            return res2;
        }

        auto get_name() const -> std::string {
            return "IdioticHash";
        }
    };

    // See Patrascu and Thorup's "The Power of Simple Tabulation Hashing"
    class SimpleTabulation {
        uint64_t tables_[sizeof(uint64_t)][1 << CHAR_BIT];

    public:
        SimpleTabulation() {
            std::uint_least64_t seed;
            sysrandom(&seed, sizeof(seed));
            std::mt19937_64 rng(seed);
            std::uniform_int_distribution<uint64_t> dist(0, UINT64_MAX);
            for (unsigned i = 0; i < sizeof(uint64_t); ++i) {
                for (int j = 0; j < (1 << CHAR_BIT); ++j) {
                    tables_[i][j] = dist(rng);
                }
            }
        }

        uint64_t operator()(uint64_t key) const {
            uint64_t result = 0;
            for (unsigned i = 0; i < sizeof(key); ++i) {
                result ^= tables_[i][reinterpret_cast<uint8_t *>(&key)[i]];
            }
            return result;
        }
    };

    class SimpleMixSplit {

    public:
        uint64_t seed;

        SimpleMixSplit() {
            std::uint_least64_t seed;
            sysrandom(&seed, sizeof(seed));
            std::mt19937_64 rng(seed);
            std::uniform_int_distribution<uint64_t> dist(0, UINT64_MAX);
            seed = dist(rng);
            // seed <<= 32;
            // seed |= random();
        }

        inline static uint64_t murmur64(uint64_t h) {
            h ^= h >> 33;
            h *= UINT64_C(0xff51afd7ed558ccd);
            h ^= h >> 33;
            h *= UINT64_C(0xc4ceb9fe1a85ec53);
            h ^= h >> 33;
            return h;
        }

        inline uint64_t operator()(uint64_t key) const {
            return murmur64(key + seed);
        }
    };

    // class my_xxhash64 {
    //     uint64_t seed;

    // public:
    //     my_xxhash64() {
    //         seed = random();
    //     }
    //     inline uint64_t operator()(uint64_t key) const {
    //         return XXHash64::hash(&key, 8, seed);
    //     }
    //     auto get_name() const -> std::string {
    //         return "xxhash64";
    //     }
    // };

    // class my_wyhash64 {
    //     uint64_t seed;

    // public:
    //     my_wyhash64() {
    //         seed = random();
    //     }
    //     inline uint64_t operator()(uint64_t key) const {
    //         return wyhash64(key, seed);
    //     }

    //     auto get_name() const -> std::string {
    //         return "wyhash64";
    //     }
    // };

    // class my_BobHash {
    //     uint64_t seed1, seed2;

    // public:
    //     my_BobHash() {
    //         seed1 = random();
    //         seed2 = random();
    //     }


    //     inline uint64_t operator()(uint32_t s) const {
    //         uint32_t out1 = seed1, out2 = seed2;
    //         void BobHash(const void *buf, size_t length, uint32_t *idx1, uint32_t *idx2);
    //         BobJenkins::BobHash((void *) &s, 4, &out1, &out2);
    //         return ((uint64_t) out1 << 32ul) | ((uint64_t) out2);

    //         // return BobJenkins::BobHash((void *) &s, 4, seed);
    //     }

    //     // inline uint64_t operator()(uint64_t s) const {
    //     //     return BobJenkins::BobHash((void *) &s, 8, seed);
    //     // }

    //     auto get_name() const -> std::string {
    //         return "BobHash";
    //     }
    // };

    // inline uint32_t hashint(uint32_t a) {
    //     a = (a + 0x7ed55d16) + (a << 12);
    //     a = (a ^ 0xc761c23c) ^ (a >> 19);
    //     a = (a + 0x165667b1) + (a << 5);
    //     a = (a + 0xd3a2646c) ^ (a << 9);
    //     a = (a + 0xfd7046c5) + (a << 3);
    //     a = (a ^ 0xb55a4f09) ^ (a >> 16);
    //     return a;
    // }

    // inline uint32_t hashint(uint64_t a) {
    //     a = (a + 0x7ed55d16) + (a << 12);
    //     a = (a ^ 0xc761c23c) ^ (a >> 19);
    //     a = (a + 0x165667b1) + (a << 5);
    //     a = (a + 0xd3a2646c) ^ (a << 9);
    //     a = (a + 0xfd7046c5) + (a << 3);
    //     a = (a ^ 0xb55a4f09) ^ (a >> 16);
    //     return a;
    // }


}// namespace hashing

#endif// CUCKOO_FILTER_HASHUTIL_H_
