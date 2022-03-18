//
// Created by tomer on 15/06/2021.
//

#ifndef MULTI_LEVEL_HASH_SHIFT_OP_HPP
#define MULTI_LEVEL_HASH_SHIFT_OP_HPP

#include <cassert>
#include <climits>
#include <cstdint>

#include <algorithm>
#include <assert.h>
#include <immintrin.h>
#include <iomanip>
#include <iostream>
#include <limits.h>
#include <stdint.h>
#include <string.h>
#include <x86intrin.h>
#include <cmath>

// #ifdef TRACY_ENABLE
//#include "../../tracy/Tracy.hpp"
//#include "randomness.hpp"
// #endif


typedef __uint128_t u128;
typedef uint64_t u64;
typedef uint32_t u32;
typedef uint16_t u16;
typedef uint8_t u8;


#define PRINT_STATE (0)

namespace Shift_op {
    inline u64 extract_word(const uint64_t *a, size_t start4) {
        size_t byte_index = start4 / 2;
        auto mp = (const u8 *) a + byte_index;
        if (!(start4 & 1)) {
            u64 h1;
            memcpy(&h1, mp, 8);
            return h1;
        }

        u64 h0 = mp[0];
        u64 h1;
        memcpy(&h1, mp + 1, 8);
        h1 = (h1 << 4u) | (h0 >> 4u);
        return h1;
    }


    void shift_arr_4bits_left_att_wrapper(uint64_t *a, size_t begin, size_t end, size_t a_size);

    void shift_arr_4bits_right_att_wrapper(uint64_t *a, size_t begin, size_t end, size_t a_size);

    inline void shift_arr_4bits_right_inside_single_word_robuster(uint64_t *a, size_t begin, size_t end) {
        // end -= (end * 16 == a_size);
        if (begin >= end) return;
        // assert(begin < end);
        // assert(end % 16);
        constexpr unsigned slot_size = sizeof(a[0]) * CHAR_BIT;
        constexpr unsigned slot_mask = slot_size - 1u;
        constexpr unsigned shift = 4u;
        constexpr unsigned slot_sh_capacity = slot_size / shift;

        assert(end % slot_sh_capacity);
        size_t index = begin / slot_sh_capacity;
        assert(index == (end / slot_sh_capacity));


        size_t rel_begin = (begin * shift) & slot_mask;
        size_t rel_end = (end * shift) & slot_mask;
        uint64_t hi_mask = _bzhi_u64(-1, rel_end + shift);
        uint64_t hi = a[index] & ~hi_mask;

        if (rel_begin == 0) {
            // std::cout << "r0: " << std::endl;
            uint64_t old_lo = a[index] & _bzhi_u64(-1, shift);
            uint64_t lo = (a[index] << shift) & hi_mask;
            // uint64_t hi = a[index] & ~hi_mask;
            assert(!(lo & hi));
            assert(!(old_lo & hi));
            a[index] = old_lo | lo | hi;
            return;
        }
        // std::cout << "r1: " << std::endl;
        assert(rel_begin < rel_end);
        uint64_t lo_mask = _bzhi_u64(-1, rel_begin);
        uint64_t lo2 = a[index] & _bzhi_u64(-1, rel_begin + shift);
        uint64_t mi3 = ((a[index] & ~lo_mask) << shift) & hi_mask;

        assert(!(lo2 & mi3) and !(lo2 & hi) and !(mi3 & hi));
        a[index] = lo2 | mi3 | hi;
        // uint64_t lo = a[index] & lo_mask;
        // uint64_t mid = (a[index] << shift) & ((~lo_mask) & hi_mask);
        // uint64_t mi2 = (a[index] & ((~lo_mask) & hi_mask)) << shift;
        // std::cout << std::string(92, '=') << std::endl;
        // std::cout << "lo:   \t\t" << format_word_to_string(lo, 64);
        // std::cout << "lo2:   \t\t" << format_word_to_string(lo2, 64);
        // std::cout << "mid:  \t\t" << format_word_to_string(mid, 64);
        // std::cout << "mi2:  \t\t" << format_word_to_string(mi2, 64);
        // std::cout << "mi3:  \t\t" << format_word_to_string(mi3, 64);
        // std::cout << "hi:   \t\t" << format_word_to_string(hi, 64);
        // assert(!(lo & mid) and !(lo & hi) and !(mid & hi));
        // assert(!(lo & mi2) and !(lo & hi) and !(mi2 & hi));
        // assert(!(lo2 & mi3) and !(lo2 & hi) and !(mi3 & hi));
        // assert(!(lo2 & mi2) and !(lo2 & mid) and !(lo2 & hi));
        // assert(!(lo2 & mi2) and !(lo2 & hi));
        // assert(!(lo2 & mi3) and !(lo2 & hi) and !(mi3 & hi));
        // std::cout << std::string(92, '=') << std::endl;
        // a[index] = lo2 | mi2 | hi;
        // a[index] = lo | mi2 | hi;
        // a[index] = lo | mid | hi;
    }

    void shift_arr_1bit_left_att_wrapper(uint64_t *a, size_t begin, size_t end, size_t a_size);

    void shift_arr_1bit_right_att_wrapper(uint64_t *a, size_t begin, size_t end, size_t a_size);

    void update_byte(uint8_t *pointer, uint8_t rem4, bool should_update_hi);

    uint8_t read_4bits(const uint8_t *a, size_t index4, size_t a_size);

    uint8_t read_4bits(const uint64_t *a, size_t index4, size_t a_size);

    bool half_byte_cmp(const uint64_t *a, size_t half_byte_index, size_t length, uint8_t rem4);

    int half_byte_cmp_get_index_for_db(const uint64_t *a, size_t half_byte_index, size_t length, uint8_t rem4);

    void unpack_array(uint8_t *unpack_array, const uint8_t *packed_array, size_t packed_size);

    void pack_array(uint8_t *pack_array, const uint8_t *unpacked_array, size_t unpacked_size);

    void unpack_array8x2(uint8_t *unpacked_array, const uint8_t *pack_array, size_t packed_size);

    void pack_array8x2(uint8_t *pack_array, const uint8_t *unpacked_array, size_t unpacked_size);

    void unpack6x8(u8 *unpackArray, const u8 *packedArray, size_t packed_size);

    void pack6x8(u8 *packedArray, const u8 *unpackArray, size_t packed_size);

    bool test_pack_unpack(const uint8_t *pack_a, size_t pack_size);

    bool memcmp_1bit(const uint8_t *a, const uint8_t *b, size_t size1);

    bool memcmp_1bit(const uint64_t *a, const uint64_t *b, size_t size1);

    // bool

    void pack_array_gen_k(u8 *pack_array, const u32 *unpacked_array, size_t items, size_t k);

    void unpack_array_gen_k(u32 *unpack_array, const u32 *packed_array, size_t items, size_t k);

    void pack_array_gen_k_with_offset(u8 *pack_array, const u32 *unpacked_array, size_t items, size_t k, size_t offset);

    void unpack_array_gen_k_with_offset(u32 *unpack_array, const u8 *packed_array, size_t items, size_t k, size_t offset);
    // void unpack_array_gen_k_with_offset(u32 *unpack_array, const u32 *packed_array, size_t items, size_t k, size_t offset);
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * like link list insertion.
     * @param packedArray
     * @param packedSize
     * @param index
     * @param item
     * Using pack unpack.
     */
    void insert_push_4bit_ultra_naive(u8 *packedArray, size_t packedSize, size_t index, u8 item);

    void insert_push_4bit_by_shift(u8 *packedArray, size_t packedSize, size_t index4, u8 item);

    inline void fix_byte(u8 *mp, bool parity, u8 rem2) {
        if (parity) {
            mp[0] = (mp[0] & 0xf) | (rem2 << 4u);
        } else {
            mp[0] = (mp[0] & 0xf0) | rem2;
        }
    }

    inline void fix_byte2(u8 *mp, bool parity, u8 rem2) {
        u8 rem_twice = rem2 | (rem2 << 4u);
        u8 mask = 0xf << (parity * 4);
        mp[0] = (mp[0] & ~mask) | (rem_twice & mask);
    }

    void insert_push_4bit_disjoint_pair(u8 *packedArray, size_t packedSize, size_t index4, u8 rem1, u8 rem2);
    /**
     * Very similar to the previous function.
     * The only difference is that unpacked_size is now given, instead of being determined as 2 * packed_size.
     *
     * @param packedArray
     * @param packedSize
     * @param index4
     * @param item
     * @param unpackSize
     */
    void insert_push_4bit_ultra_naive_by_unpackSize(u8 *packedArray, size_t packedSize, size_t index4, u8 item,
                                                    size_t unpackSize);

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    template<typename T>
    void init_array(T *a, size_t a_size) {
        std::fill(a, a + a_size, 0);
    }


    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    __uint128_t get_4bits_cmp_vector(const uint64_t *a, size_t start4, size_t length, uint8_t rem4);

    u16 get_4bits_cmp_on_word3(const u8 word[8], size_t start4, uint8_t rem4, size_t length);

    u16 get_4bits_cmp16(const uint64_t *a, size_t start4, uint8_t rem4);

    u16 get_4bits_cmp_on_word(const u8 word[8], uint8_t rem4);
    /**
     * @brief if start4 is odd, compares only 15 items, where if it start4 is even, we compare 16 4 bits items.
     * 
     * @param a 
     * @param start4 
     * @param rem4 
     * @return u16 
     */
    inline u16 get_4bits_cmp16_ver2(const uint64_t *a, size_t start4, uint8_t rem4) {
        size_t byte_index = start4 / 2;
        auto mp = (const u8 *) a + byte_index;
        u16 cmp_mask = get_4bits_cmp_on_word(mp, rem4);
        return cmp_mask >> (start4 & 1);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    inline u8 flip2x4(u8 x) {
        return (x >> 4) | (x << 4);
    }

    void reverse_4_bits_array_naive(const u8 *a, u8 *rev_a, size_t packed_size);

    void reverse_4_bits_array(const u8 *a, u8 *rev_a, size_t packed_size);

    void reverse_4_bits_array_in_place(u8 *a, size_t packed_size);

    void insert_push_4bit_disjoint_pair_reversed_array_naive(u8 *packedArray, size_t packedSize, size_t index4, u8 rem1, u8 rem2);

    void insert_push_4bit_disjoint_pair_reversed_array_by_push(u8 *packedArray, size_t packedSize, size_t index4, u8 rem1, u8 rem2);

    void insert_push_4bit_disjoint_pair_reversed_array(u8 *packedArray, size_t packedSize, size_t index4, u8 rem1, u8 rem2);

    void shift_arr_4bits_right_att_wrapper8_un(uint8_t *a, size_t begin4, size_t end4, size_t a_size8);

    void shift_arr_4bits_left_att_wrapper8_un(uint8_t *a, size_t begin4, size_t end4, size_t a_size8);

    void shift_arr_4bits_left_att_wrapper8_sun(uint8_t *a, size_t begin4, size_t end4, size_t a_size8);

    void shift_arr_k_bits_right_att_wrapper(u8 *a, size_t begin, size_t end, size_t a8_size, size_t k);

    void shift_arr_1bit_right_att_wrapper8(uint8_t *a, size_t begin, size_t end, size_t a8_size);
}// namespace Shift_op


namespace bitsMani {
    constexpr u64 slot_size = 64u;

    inline bool is_single_bit_set(const uint64_t *a, size_t index1, size_t a_size) {
        assert(index1 < a_size * 64);
        if (a_size == 1) {
            bool att = a[0] & (1ULL << index1);
            bool val = _bextr_u64(a[0], index1, 1);
            assert(att == val);
            return att;
        }
        const size_t w_index = index1 / 64;
        const size_t j = index1 & 63u;
        bool att = a[w_index] & (1ULL << j);
        bool val = _bextr_u64(a[w_index], j, 1);
        assert(att == val);
        return att;
    }

    __attribute__((always_inline)) inline size_t pop64(u64 x) {
        return _mm_popcnt_u64(x);
    }

    inline size_t pop_array(const u64 *a, size_t size1) {
        constexpr u64 slot_mask = (slot_size - 1u);
        size_t size = 1 + (size1 - 1) / slot_size;
        size_t sum = 0;
        if (size1 & slot_mask) {
            const size_t rel_index = size1 & slot_mask;
            u64 temp_w = a[size - 1u] & _bzhi_u64(-1, rel_index);
            sum += _mm_popcnt_u64(temp_w);
            size -= 1u;
        }
        for (size_t i = 0; i < size; ++i) {
            sum += _mm_popcnt_u64(a[i]);
        }
        return sum;
    }

    inline size_t pop_array_with_limits(const u64 *a, size_t start_index1, size_t end_index1) {
        assert(start_index1 <= end_index1);
        constexpr u64 slot_mask = (slot_size - 1u);
        size_t const useful_end = end_index1 - 1;
        size_t shifted_first_word = a[start_index1 / slot_size] >> (start_index1 & slot_mask);
        size_t shifted_last_word = a[useful_end / slot_size] << (63 - (useful_end & slot_mask));

        size_t first_pop = _mm_popcnt_u64(shifted_first_word);
        size_t last_pop = _mm_popcnt_u64(shifted_last_word);
        size_t fixed_start = start_index1 / 64 * 64 + 64;
        size_t fixed_end = useful_end / 64 * 64;
        assert(!((fixed_start | fixed_end) & 63));
        size_t mid_s1 = fixed_end - fixed_start;
        size_t mid_pop = 0;
        for (size_t i = 1; i * 64 <= mid_s1; ++i) {
            mid_pop += _mm_popcnt_u64(a[i]);
        }

        return mid_pop + first_pop + last_pop;
    }

    /**
     * returns
     * @param x word
     * @param j index
     *
     * @return the position (starting from 0) of the jth set bit of x. Or 64 if pop64(x) <= j
     */
    __attribute__((always_inline)) inline size_t select64(u64 x, u64 j) {
        assert(j < 64);
        const uint64_t y = _pdep_u64(UINT64_C(1) << j, x);
        return _tzcnt_u64(y);
    }

    /**
     * Like select, just on arrays.
     * @param k
     * @param a
     * @param size1
     * @return
     */
    inline size_t select_arr(u64 k, const u64 *a, size_t size1) {
        size_t temp_k = k;
        assert(temp_k < 64 * size1);
        assert(size1);
        const size_t size64 = 1 + ((size1 - 1) / 64);
        for (size_t i = 0; i < size64 - 1; ++i) {
            uint64_t temp_word = a[i];
            auto temp_pop = pop64(temp_word);
            if (temp_k < temp_pop) {
                auto res = select64(temp_word, temp_k);
                return i * sizeof(a) * CHAR_BIT + res;
            }
            temp_k -= temp_pop;
        }
        uint64_t last_word = a[size64 - 1];
        assert(temp_k < pop64(last_word));
        auto res = select64(last_word, temp_k);
        return (size64 - 1) * sizeof(a) * CHAR_BIT + res;
    }

    inline size_t select_zero_arr(u64 k, const u64 *a, size_t size1) {
        size_t temp_k = k;
        assert(temp_k < 64 * size1);
        assert(size1);
        const size_t size64 = 1 + ((size1 - 1) / 64);
        for (size_t i = 0; i < size64 - 1; ++i) {
            uint64_t temp_word = ~a[i];
            auto temp_pop = pop64(temp_word);
            if (temp_k < temp_pop) {
                auto res = select64(temp_word, temp_k);
                return i * sizeof(a) * CHAR_BIT + res;
            }
            temp_k -= temp_pop;
        }
        uint64_t last_word = ~a[size64 - 1];
        assert(temp_k < pop64(last_word));
        auto res = select64(last_word, temp_k);
        return (size64 - 1) * sizeof(a) * CHAR_BIT + res;
    }

    inline void select_both_on_word(u64 x, size_t j, size_t *begin, size_t *end) {
        assert(j < 64);
        const uint64_t y = _pdep_u64(UINT64_C(3) << j, x);
        assert(_mm_popcnt_u64(y) == 2);
        /*
        if (y & (y >> 1u)) {
            assert(!Find_Ultra_Naive(quot, rem, pd));
            return false;
        }
    */
        // *begin = _tzcnt_u64(y) + 1;
        *begin = _tzcnt_u64(y);
        *end = _tzcnt_u64(_blsr_u64(y));
    }

    inline void select_both_arr(size_t k, const u64 *a, size_t size1, size_t *begin, size_t *end) {
        //    constexpr u64 slot_size = sizeof(a) * CHAR_BIT;
        assert(size1);
        const size_t size64 = 1 + ((size1 - 1) / 64);
        //        const size_t size64 = (size1 + 63) / 64;
        const size_t rel_index = size1 & 63;
        const u64 last_word_mask = (rel_index) ? (1ULL << rel_index) - 1u : UINT64_MAX;
        bool was_begin_set = false;
        // const size_t original_k = k;
        size_t temp_k = k;
        size_t i = 0;
        for (; i < size64 - 1; ++i) {
            uint64_t temp_word = a[i];
            auto temp_pop = pop64(temp_word);
            if (temp_k < temp_pop) {
                size_t offset = i * slot_size;
                if (temp_k + 1 < temp_pop) {
                    select_both_on_word(temp_word, temp_k, begin, end);
                    *begin += offset;
                    *end += offset;
                    assert(*begin <= *end);
                    return;
                }
                auto res = 63u - _lzcnt_u64(temp_word);
                assert(res == select64(temp_word, temp_k));

                // *begin = offset + res + 1;
                *begin = offset + res;
                was_begin_set = true;
                temp_k = (temp_k - temp_pop) + 1; /* temp_k++;*/
                i++;
                break;
            }
            temp_k -= temp_pop;
        }
        if (was_begin_set) {
            for (; i < size64 - 1; ++i) {
                uint64_t temp_word = a[i];
                auto temp_pop = pop64(temp_word);
                if (temp_k < temp_pop) {
                    auto res = select64(temp_word, temp_k);
                    *end = i * slot_size + res;
                    assert(*begin <= *end);
                    return;
                }
                temp_k -= temp_pop;
            }
            uint64_t last_word = a[i] & last_word_mask;
            assert(temp_k < pop64(last_word));
            auto res = select64(last_word, temp_k);
            *end = (size64 - 1) * slot_size + res;
            assert(*begin <= *end);
            return;
        }

        uint64_t temp_word = a[i] & last_word_mask;
        assert(temp_k + 1 < pop64(temp_word));
        const size_t offset = i * slot_size;
        select_both_on_word(temp_word, temp_k, begin, end);
        *begin += offset;
        *end += offset;
        assert(*begin <= *end);
    }

    /**
     *
     * @param k
     * @param a
     * @param size
     * @return the number of zeros between the k'th one, and the next one.
     */
    inline size_t count_zeros_between_consecutive_ones(u64 k, const u64 *a, size_t size1) {
        assert(k + 1 < pop_array(a, size1));
        return select_arr(k + 1, a, size1) - (select_arr(k, a, size1) + 1);
    }

    /**
     * Leading zero count on array, from given index.
     * @param pos_index
     * @param a
     * @param size
     * @return
     */
    inline size_t lzcnt_arr(size_t pos_index, const u64 *a, size_t size) {
        //        constexpr u64 slot_size = 64u;
        size_t index = (pos_index - 1) / slot_size;
        assert(index < size);
        if (pos_index & (slot_size - 1)) {
            size_t rel_index = pos_index & (slot_size - 1);
            assert(rel_index != 0);
            assert(rel_index < slot_size);
            u64 temp_word = a[index] & _bzhi_u64(-1, (u64) rel_index);
            if (temp_word) {
                auto abs_res = _lzcnt_u64(temp_word);
                auto res = abs_res - (64 - rel_index);
                return res;
            }

            size_t new_index = pos_index ^ rel_index;
            assert((new_index & (slot_size - 1u)) == 0);
            return rel_index + lzcnt_arr(new_index, a, size);
        }
        for (size_t i = 0; i <= index; ++i) {
            u64 temp_word = a[index - i];
            if (temp_word) {
                return i * 64 + _lzcnt_u64(temp_word);
            }
        }
        assert(0);
        return -1;
    }

    inline size_t leading_ones_count_arr(size_t pos_index, const u64 *a, size_t size) {
        //        constexpr u64 slot_size = 64u;
        size_t index = (pos_index - 1) / slot_size;
        assert(index < size);
        if (pos_index & (slot_size - 1)) {
            size_t rel_index = pos_index & (slot_size - 1);
            assert(rel_index != 0);
            assert(rel_index < slot_size);
            u64 temp_word = ~(a[index] & _bzhi_u64(-1, (u64) rel_index));
            if (temp_word != UINT64_MAX) {
                auto abs_res = _lzcnt_u64(temp_word);
                auto res = abs_res - (64 - rel_index);
                return res;
            }

            size_t new_index = pos_index ^ rel_index;
            assert((new_index & (slot_size - 1u)) == 0);
            return rel_index + lzcnt_arr(new_index, a, size);
        }
        for (size_t i = 0; i <= index; ++i) {
            u64 temp_word = ~a[index - i];
            if (temp_word != UINT64_MAX) {
                return i * 64 + _lzcnt_u64(temp_word);
            }
        }
        assert(0);
        return -1;
    }

    inline size_t tzcnt_arr(const u64 *a, size_t size64) {
        if (a[0])
            return _tzcnt_u64(a[0]);

        for (size_t i = 1; i < size64; ++i) {
            if (a[i])
                return i * 64 + _tzcnt_u64(a[i]);
        }
        assert(0);
        return -1;
    }


    inline size_t first_to_last_one_distance(u64 x) {
        assert(x);
        size_t last_one_index = 63 - _lzcnt_u64(x);
        size_t first_one_index = _tzcnt_u64(x);
        return last_one_index - first_one_index;
    }


    inline bool only_consecutive_ones_naive(u64 word) {
        auto pop = pop64(word);
        auto tz = _tzcnt_u64(word);
        auto lz = _lzcnt_u64(word);
        auto start = tz;
        auto end = 63 - lz;
        assert((start != end) or (pop == 1));
        bool res = (pop == (end - start + 1));
        return res;
    }

    bool zero0s_between_k_ones_word(size_t k, size_t range, u64 word);

    /*inline size_t zero0s_between_k_ones(size_t k, size_t range, const u64 *a, size_t size1) {
        size_t temp_k = k;
        assert(temp_k < 64 * size1);
        assert(size1);
        if (size1 <= 64) {
        }
        const size_t size64 = 1 + ((size1 - 1) / 64);
        for (size_t i = 0; i < size64 - 1; ++i) {
            uint64_t temp_word = a[i];
            auto temp_pop = pop64(temp_word);
            if (temp_k < temp_pop) {
                auto res = select64(temp_word, temp_k);
                return i * sizeof(a) * CHAR_BIT + res;
            }
            temp_k -= temp_pop;
        }
        uint64_t last_word = a[size64 - 1];
        assert(temp_k < pop64(last_word));
        auto res = select64(last_word, temp_k);
        return (size64 - 1) * sizeof(a) * CHAR_BIT + res;
    }
*/

    /**
     * @brief 
     * 
     * @param k 
     * @param range 
     * @return true If starting with the k`th 1, there are range consecutive 1, without any 0 between them.
     * @return false Otherwise.
     */
    // inline bool zero0s_between_k_ones(size_t k, size_t range, const u64 *a, size_t size) {
    // }

    template<typename T>
    void fill_array_with_ones(T *a, size_t ones_count, size_t a_size) {
        const T x = 0;
        const T y = x - 1;
        assert(y > x);// validating T is unsigned.

        if (ones_count == 0)
            return;
        constexpr uint64_t slotSize = sizeof(T) * CHAR_BIT;
        assert(slotSize > 1);
        size_t full_ones_words = ones_count / slotSize;
        assert(full_ones_words <= a_size);
        size_t i = 0;
        for (; i < full_ones_words; i++) {
            a[i] = y;
        }
        size_t ones_remainder = ones_count & (slotSize - 1u);
        if (ones_remainder) {
            T mask = (((T) 1) << ones_remainder) - 1u;
            assert(i < a_size);
            a[i] = mask;
        }
        // size_t full_ones_words = (ones_count + slotSize - 1u)
    }

    inline bool compare_k_packed_items(u64 word, u8 rem, size_t rem_length, size_t items) {
        assert(rem_length <= 8);
        const u64 mask = (1ULL << rem_length) - 1;
        assert(rem <= mask);
        for (size_t i = 0; i < items; i++) {
            if ((word & mask) == rem)
                return true;
            word >>= rem_length;
        }
        return false;
    }
    inline bool cmp_bits_inside_un_aligned_single_word(const u64 *a, u8 rem, size_t rem_length, size_t start_index1, size_t end_index1) {
        assert(((end_index1 - start_index1) % rem_length) == 0);
        const size_t items = (end_index1 - start_index1) / rem_length;
        auto mp = (const u8 *) a;
        const size_t total_bits_to_compare = end_index1 - start_index1;
        const size_t offset = start_index1 & 7;
        assert(total_bits_to_compare + offset <= 64);

        u64 word = 0;
        memcpy(&word, mp + (start_index1 / 8), 8);
        word >>= offset;
        return compare_k_packed_items(word, rem, rem_length, items);
    }

    bool compare_bits_ranged(const u64 *a, u8 rem, size_t rem_length, size_t start_index1, size_t end_index1);

    bool compare_bits(const u64 *a, u8 rem, size_t rem_length, size_t index1);

    u64 get_compare_mask(const u8 *a, u32 rem, size_t rem_length, size_t start_index1, size_t end_index1);

    __attribute__((always_inline)) inline uint32_t reduce32(uint32_t hash, uint32_t n) {
        // http://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/
        return (uint32_t) (((uint64_t) hash * n) >> 32);
    }

    u64 extract_bits(const u8 *a, size_t start1, size_t end1);

    void update_bits(u8 *a, size_t start1, size_t end1, u64 value);

    void update_bits_inside_8bytes_boundaries_safer(u8 *a, size_t start1, size_t k, u64 value);
}// namespace bitsMani


namespace Shift_op::check {
    bool test_rev4_bits_arr(const u8 *a, size_t packed_size);
    bool test_rev4_bits_in_place(const u8 *a, size_t packed_size);

    bool test_shift4_right_r();

    bool test_shift4_left_r();

    /**
     * Validates the functionality of an insert_push function, based on the original input, and the result.
     * @param pre_a The original input
     * @param post_a The output
     * @param packed_size number of 4 items in pre_a.
     * @param index4 the index in which we wanted to insert new item.
     * @param item the new item value. (4 bits).
     * @return
     */
    bool validate_insert_push_4bit(u8 *pre_a, u8 *post_a, size_t packed_size, size_t index4, u8 item);

    bool validate_insert_push_4bit_disjoint_pair(u8 *pre_a, u8 *post_a, size_t packed_size, size_t index4, u8 lo_rem1, u8 lo_rem2);

    void insert_push_4bit_disjoint_pair_reversed_array(u8 *packedArray, size_t packedSize, size_t index4, u8 rem1, u8 rem2);

    bool test_pack_unpack_6x8(const uint8_t *pack_a, size_t pack_size);

    bool test_pack_unpack_pdep0();

    bool test_pack_unpack_pdep();

    bool test_pack_unpack_array_gen_k(const uint32_t *pack_a, size_t items, size_t k);

    void comp_test0_shift_arr_k_bits_right_att_wrapper();

    bool comp_test_shift_arr_k_bits_right_att_wrapper();
}// namespace Shift_op::check

namespace bitsMani::test {
    bool zero0s_between_k_ones_word_naive(size_t k, size_t range, u64 word);
    bool val_zero0s_between_k_ones_word(size_t k, size_t range, u64 word);
    bool val_zero0s_between_k_ones_word_rand(size_t reps);
    void prt_zeros_failed(size_t k, size_t range, u64 word);

    bool wrap_extract_update_bits();
}// namespace bitsMani::test
namespace str_bitsMani {
    inline auto to_bin(uint64_t x, size_t length) -> std::string {
        assert(length <= 64);
        uint64_t b = 1ULL;
        std::string res;
        for (size_t i = 0; i < length; i++) {
            res += (b & x) ? "1" : "0";
            b <<= 1ul;
        }
        return res;
    }

    inline auto space_string(const std::string &s) -> std::string {
        std::string new_s;
        for (size_t i = 0; i < s.size(); i += 4) {
            if (i) {
                if (i % 16 == 0) {
                    new_s += "|";
                } else if (i % 4 == 0) {
                    new_s += ".";
                }
            }
            new_s += s.substr(i, 4);
        }
        return new_s;
    }

    inline auto format_word_to_string(uint64_t x, size_t length = 64) -> std::string {
        std::string res = to_bin(x, length);
        return space_string(res);
        // std::cout << space_string(res) << std::endl;
    }
    inline auto format_128word_to_string(__uint128_t x, size_t length = 128) -> std::string {
        assert(length >= 64);
        u64 hi = x >> 64;
        std::string lo_s = to_bin(x, 64);
        std::string hi_s = to_bin(hi, length - 64);
        std::string res = lo_s + hi_s;
        return space_string(res);
        // std::cout << space_string(res) << std::endl;
    }

    inline auto format_2words_and_xor(uint64_t x, uint64_t y, size_t length = 64) -> std::string {
        std::stringstream ss;
        ss << format_word_to_string(x, length) << std::endl;
        ss << format_word_to_string(y, length) << std::endl;
        ss << format_word_to_string(x ^ y, length) << std::endl;
        return ss.str();
    }

    inline std::string str_array_as_memory_no_delim(const uint8_t *a, size_t size8) {
        std::string res;
        size_t temp_size = size8;
        size_t byte_index = 0;
        while (temp_size >= 8) {
            uint64_t h = 0;
            memcpy(&h, a + byte_index, 8);
            std::string temp_s = to_bin(h, 64);
            res += temp_s;
            byte_index += 8;
            temp_size -= 8;
        }
        if (temp_size) {
            uint64_t h = 0;
            memcpy(&h, a + byte_index, temp_size);
            std::string temp_s = to_bin(h, temp_size * 8);
            res += temp_s;
        }
        return res;
    }

    inline std::string str_array_as_memory(const uint8_t *a, size_t size8) {
        std::stringstream s_res;
        if (size8 <= 8) {
            uint64_t h = 0;
            memcpy(&h, a, size8);
            auto s = format_word_to_string(h, size8 * 8);
            // s_res << std::string(80, '*') << std::endl;
            s_res << s << std::endl;
            // s_res << std::string(80, '*') << std::endl;
            return s_res.str();
        }
        size_t size64 = (size8 + 7) / 8;
        uint64_t a64[size64];
        Shift_op::init_array(a64, size64);
        memcpy(a64, a, size8);
        s_res << std::string(80, '*') << std::endl;
        for (size_t i = 0; i < size64; i++) {
            s_res << i << ":\t" << format_word_to_string(a64[i], 64);
            s_res << "\t|\t" << _mm_popcnt_u64(a64[i]) << std::endl;
        }
        s_res << std::string(80, '*') << std::endl;
        return s_res.str();
    }

    inline std::string str_array_half_byte_aligned(const uint8_t *a, size_t packed_size) {
        if (packed_size == 0) {
            return "Empty!";
        }
        std::stringstream ss;
        assert(packed_size);
        //        std::cout << std::endl;
        ss << std::string(86, '-') << std::endl;

        ss << "0:\t";
        auto temp = (uint16_t) a[0];
        u16 lo0 = temp & 0xf;
        u16 hi0 = temp >> 4u;
        ss << std::left << std::setw(4) << (lo0);
        ss << ", " << std::left << std::setw(4) << (hi0);
        for (size_t i = 1; i < packed_size; i++) {
            bool cond = ((i % 4) == 0);
            if (!cond) {
                ss << ", ";
            }

            temp = ((uint16_t) a[i]);
            u16 lo = temp & 0xf;
            u16 hi = temp >> 4u;
            ss << std::left << std::setw(4) << (lo);
            ss << ", " << std::left << std::setw(4) << (hi);

            if (i % 4 == 3) {
                ss << std::endl;
                ss << ((i + 1) * 2) << ":\t";
            }
        }
        ss << std::endl;
        ss << std::string(86, '-') << std::endl;
        return ss.str();
    }

    template<typename T>
    inline std::string str_array_with_line_numbers(const T *a, size_t items, bool to_hex = false) {
        auto base = std::dec;
        if (to_hex) {
            base = std::hex;
        }
        if (items == 0) {
            std::stringstream s_res;
            s_res << std::endl;
            s_res << std::string(86, '-') << std::endl;
            s_res << "Empty!";
            s_res << std::string(86, '-') << std::endl;
            return s_res.str();
        }

        std::stringstream s_res;
        s_res << std::endl;
        s_res << std::string(86, '-') << std::endl;

        u64 s0 = a[0];

        s_res << "0:\t" << std::left << std::setw(3) << base << s0;
        for (size_t i = 1; i < items; i++) {
            bool cond = ((i % 8) == 0);
            if (!cond) {
                s_res << ", ";
            }
            u64 temp_s = a[i];
            s_res << std::left << std::setw(3) << base << temp_s;
            if (i % 8 == 7) {
                s_res << std::endl;
                s_res << (i + 1) << ":\t";
            }
        }
        s_res << std::endl;
        s_res << std::string(86, '-') << std::endl;
        return s_res.str();
    }

    inline std::string str_array_with_line_numbers(const uint8_t *a, size_t size) {
        if (size == 0) {
            std::stringstream s_res;
            s_res << std::endl;
            s_res << std::string(86, '-') << std::endl;
            s_res << "Empty!";
            s_res << std::string(86, '-') << std::endl;
            return s_res.str();
        }

        assert(size);
        std::stringstream s_res;
        s_res << std::endl;
        s_res << std::string(86, '-') << std::endl;

        auto s0 = (uint16_t) a[0];
        s_res << "0:\t" << std::left << std::setw(3) << s0;
        for (size_t i = 1; i < size; i++) {
            bool cond = ((i % 8) == 0);
            if (!cond) {
                s_res << ", ";
            }
            auto temp_s = (uint16_t) a[i];
            s_res << std::left << std::setw(3) << temp_s;
            if (i % 8 == 7) {
                s_res << std::endl;
                s_res << (i + 1) << ":\t";
            }
        }
        s_res << std::endl;
        s_res << std::string(86, '-') << std::endl;
        return s_res.str();
    }

    template<typename T>
    std::string get_first_k_bits_of_each_item(const T *a, size_t items, size_t k) {
        std::stringstream s_res;
        //        s_res << format_word_to_string(a[0], k);
        for (size_t i = 0; i < items; ++i) {
            s_res << i << ":\t" << format_word_to_string(a[i], k) << std::endl;
        }
        return s_res.str();
    }
    template<typename T>
    std::string format_qr(T qr) {
        assert(sizeof(T) == 2);
        std::string a = std::to_string(qr >> 8u);
        if (a.length() < 2) {
            a += " ";
        }
        std::string b = std::to_string(qr & 0xff);
        if (b.length() < 3) {
            b = b + std::string(3 - b.length(), ' ');
        }
        std::string tp = "(" + a + ", " + b + ")";
        return tp;
    }

    inline std::string format_qr_by_width(u64 qr, size_t long_length, bool to_hex = false) {
        std::string a = std::to_string(qr >> long_length);
        size_t digits = std::ceil(log10((double) (1ULL << long_length)));
        while (a.length() < 2) {
            a += " ";
        }

        auto rem = qr & _bzhi_u64(-1, long_length);
        std::string b;
        if (to_hex) {
            std::stringstream ss;
            ss << std::hex << rem;
            b = ss.str();
        } else {
            b = std::to_string(rem);
        }
        if (b.length() < digits) {
            b += std::string(digits - b.length(), ' ');
        }
        std::string tp = "(" + a + ", " + b + ")";
        return tp;
    }

    inline void print_array_as_tuples_with_line_numbers(const uint16_t *a, size_t size) {
        assert(size);
        std::cout << std::endl;
        std::cout << std::string(86, '-') << std::endl;
        auto s0 = format_qr(a[0]);
        std::cout << "0:\t" << s0;
        for (size_t i = 1; i < size; i++) {
            bool cond = ((i % 8) == 0);
            if (!cond) {
                std::cout << ", ";
            }
            auto temp_s = format_qr(a[i]);
            std::cout << temp_s;
            if (i % 8 == 7) {
                std::cout << std::endl;
                std::cout << (i + 1) << ":\t";
            }
        }
        std::cout << std::endl;
        std::cout << std::string(86, '-') << std::endl;
    }

    inline std::string str_array_as_tuples_with_line_numbers(const uint16_t *a, size_t size) {
        assert(size);
        std::stringstream ss;
        ss << std::endl;
        ss << std::string(86, '-') << std::endl;
        auto s0 = format_qr(a[0]);
        ss << "0:\t" << s0;
        for (size_t i = 1; i < size; i++) {
            bool cond = ((i % 8) == 0);
            if (!cond) {
                ss << ", ";
            }
            auto temp_s = format_qr(a[i]);
            ss << temp_s;
            if (i % 8 == 7) {
                ss << std::endl;
                ss << (i + 1) << ":\t";
            }
        }
        ss << std::endl;
        ss << std::string(86, '-') << std::endl;
        auto res = ss.str();
        return res;
    }

    inline std::string str_array_as_tuples_for_long_rems(const u64 *a, size_t size, size_t l_len, bool to_hex = false) {
        assert(size);
        std::stringstream ss;
        ss << std::endl;
        ss << std::string(86, '-') << std::endl;
        auto s0 = format_qr_by_width(a[0], l_len, to_hex);
        ss << "0:\t" << s0;
        for (size_t i = 1; i < size; i++) {
            bool cond = ((i % 8) == 0);
            if (!cond) {
                ss << ", ";
            }
            auto temp_s = format_qr_by_width(a[i], l_len, to_hex);
            ss << temp_s;
            if (i % 8 == 7) {
                ss << std::endl;
                ss << (i + 1) << ":\t";
            }
        }
        ss << std::endl;
        ss << std::string(86, '-') << std::endl;
        auto res = ss.str();
        return res;
    }
    //    struct qr32 {
    //        u32 quot;
    //        u32 rem;
    //
    //        qr32() : quot(0), rem(0) {}
    //        qr32(u32 q, u32 r) : quot(q), rem(r) {}
    //
    //        qr32(u64 qr_pair) : quot(qr_pair >> 32), rem(qr_pair & 0xffff) {}
    //    };

    inline std::string str_unpack_print_array(const u8 *a, size_t start_index1, size_t items, size_t items_len, bool to_hex = false) {
        assert(items <= 256);
        u64 temp_array[items];
        Shift_op::init_array(temp_array, items);
        size_t temp_start = start_index1;
        for (size_t i = 0; i < items; ++i) {
            auto value = bitsMani::extract_bits(a, temp_start, temp_start + items_len);
            temp_array[i] = value;
            temp_start += items_len;
        }
        auto s = str_array_with_line_numbers(temp_array, items, to_hex);
        return s;
    }

}// namespace str_bitsMani

namespace Shift_pd {

    /**
     * @brief right is move bits to higher address. << ( which is actually left)
     * 
     * @tparam k2 
     * @param pd 
     * @param start1 
     * @param end1 bits after this index don't change.
     * @return size_t 
     */
    template<size_t k>
    inline void shift_by_k_right(__m512i *pd, size_t start1, size_t end1) {
        assert(end1 <= 512);
        assert(start1 <= end1);// this is not a must
        if (start1 >= 512 - 64) {
            //FIXME:: the case of end = 512 needs to be handled differently.
            if (start1 + k == 512) {
                //                std::cout << "start1 + k == 512" << std::endl;
                return;
            }
            assert(start1 + k < 512);
            u64 word = 0;
            memcpy(&word, (u8 *) pd + 64 - 8, 8);
            size_t new_start = start1 - (512 - 64);
            size_t new_end = end1 - (512 - 64);
            const u64 lo_mask = (1ULL << (new_start + k)) - 1u;
            const u64 hi_mask = _bzhi_u64(-1, new_end);
            assert(hi_mask);
            u64 lo = word & lo_mask;
            u64 hi = word & ~hi_mask;
            u64 mid_to_shift = word & ~_bzhi_u64(-1, new_start);
            // u64 mid = (word << k) & (hi_mask & ~lo_mask);
            u64 mid = (mid_to_shift << k) & hi_mask;
            assert(!(lo & mid));
            assert(!(lo & hi));
            assert(!(hi & mid));
            u64 new_word = lo | mid | hi;
            memcpy((u8 *) pd + 64 - 8, &new_word, 8);
            //            std::cout << "case_m1: " << std::endl;
            return;
        }
        //        const size_t items = (end1 - start1) / k;
        auto mp = (u8 *) pd;
        const size_t total_bits_to_compare = end1 - start1;
        const size_t offset = start1 & 7;
        if (total_bits_to_compare + offset <= 64) {
            u64 word = 0;
            const size_t bytes_to_copy = (offset + total_bits_to_compare + 7) / 8;
            assert((start1 / 8) + bytes_to_copy <= 64);
            memcpy(&word, mp + (start1 / 8), bytes_to_copy);
            const u64 lo_mask = (1ULL << (offset + k)) - 1u;
            assert(lo_mask);// this can't be 0;
            // const u64 lo_mask = (1ULL << (offset + 1)) - 1u;
            // const u64 hi_mask = (1ULL << (offset + total_bits_to_compare)) - 1u;
            const u64 hi_mask = _bzhi_u64(-1, offset + total_bits_to_compare);
            assert(hi_mask);
            u64 lo = word & lo_mask;
            u64 hi = word & ~hi_mask;
            u64 mid_to_shift = word & ~_bzhi_u64(-1, offset);
            u64 mid = (mid_to_shift << k) & hi_mask;
            // u64 mid = (word << k) & (hi_mask & ~_bzhi_u64(-1, offset + k));
            // u64 mid = (word << k) & (hi_mask & ~lo_mask);
            assert(!(lo & mid));
            assert(!(lo & hi));
            assert(!(hi & mid));
            u64 new_word = lo | mid | hi;
            /* if (k == 2) { 
                std::cout << std::string(80, '=') << std::endl;
                std::cout << "start1: \t" << start1 << std::endl;
                std::cout << "end1:   \t" << end1 << std::endl;
                std::cout << "offset: \t" << offset << std::endl;
                std::cout << std::string(80, '~') << std::endl;
                std::cout << "lo:       \t" << str_bitsMani::format_word_to_string(lo) << std::endl;
                std::cout << "mid:      \t" << str_bitsMani::format_word_to_string(mid) << std::endl;
                std::cout << "hi:       \t" << str_bitsMani::format_word_to_string(hi) << std::endl;
                std::cout << "word:     \t" << str_bitsMani::format_word_to_string(word) << std::endl;
                std::cout << "new_word: \t" << str_bitsMani::format_word_to_string(new_word) << std::endl;
                std::cout << std::string(80, '=') << std::endl;
            }
            */
            memcpy(mp + (start1 / 8), &new_word, bytes_to_copy);
            //            std::cout << "case0: " << std::endl;
            return;
        }
        u64 *pd64 = (u64 *) pd;
        const size_t first_word_index = start1 / 64;
        const size_t last_word_index = (end1 - 1) / 64;
        // size_t temp_index = last_word_index;
        const u64 first_word = pd64[first_word_index];
        // const u64 second_word = pd64[first_word_index + 1];
        const u64 last_word = pd64[last_word_index];

        for (int i = (int) last_word_index; i > (int) first_word_index; i--) {
            assert(i > 0);
            pd64[i] = (pd64[i] << k) | (pd64[i - 1] >> (64 - k));
        }

        //fix first word
        if ((start1 & 63) + k > 64) {
            //            std::cout << "rel-start1 + k:\t" << (start1 & 63) + k << std::endl;
            size_t rel_start = (start1 & 63);
            size_t shift_by = rel_start + k - 64;
            //            u64 temp_w1 = first_word & ~_bzhi_u64(-1, rel_start - 1);
            //            u64 temp_w1 = first_word & ~_bzhi_u64(-1, rel_start); //FIXME!!!
            u64 temp_w1 = (first_word >> rel_start);//FIXME!!!
                                                    //            u64 lo_w1 = temp_w1 << (shift_by - 1);
            u64 lo_w1 = temp_w1 << shift_by;
#ifndef NDEBUG
            u64 bits_to_move = _bextr_u64(first_word, rel_start, 64 - rel_start);
            u64 shifted_bits = bits_to_move << shift_by;
            if (lo_w1 != shifted_bits) {
                std::cout << "lo_w1: \t" << lo_w1 << std::endl;
                std::cout << "shifted_bits: \t" << shifted_bits << std::endl;
                assert(0);
            }
#endif//!NDEBUG
            const size_t rel_index = (start1 + k) & 63;
            const u64 mask = _bzhi_u64(-1, rel_index);
            // u64 new_lo = (first_word &  >> (64 - k)) & mask;

            u64 temp_lo = temp_w1 >> (64 - k);
#ifndef NDEBUG
            u64 masked_temp_lo = temp_lo & mask;
            if (temp_lo != masked_temp_lo) {
                std::cout << "temp_lo:   \t" << temp_lo << std::endl;
                std::cout << "m_temp_lo: \t" << masked_temp_lo << std::endl;
                std::cout << std::string(80, '-') << std::endl;
                std::cout << "temp_lo:   \t" << str_bitsMani::format_word_to_string(temp_lo) << std::endl;
                std::cout << "m_temp_lo: \t" << str_bitsMani::format_word_to_string(masked_temp_lo) << std::endl;
                assert(0);
            }

#endif//!NDEBUG                                 \
        // u64 lo = temp_w1 >> (64 - k) & mask; \
        // u64 lo = pd64[first_word_index + 1] & mask;
            u64 hi = pd64[first_word_index + 1] & ~mask;
            assert(!(temp_lo & hi));
            pd64[first_word_index + 1] = temp_lo | hi;
        } else {
            //            std::cout << "start != 63." << std::endl;
            u64 lo = first_word & _bzhi_u64(-1, (start1 & 63) + k);
            u64 mid = (first_word & ~_bzhi_u64(-1, start1 & 63)) << k;
            assert(!(lo & mid));
            pd64[first_word_index] = lo | mid;
            /* const u64 mask1 = _bzhi_u64(-1, (start1 + k) & 63);
            const u64 w1_shifted = first_word << k;
            pd64[first_word_index] = (first_word & mask1) | (w1_shifted & ~mask1); */
        }
        //fix last word
        const u64 mask2 = _bzhi_u64(-1, end1 & 63);
        if ((end1 & 63) == 0) {
            //            std::cout << "end is aligned." << std::endl;
            return;
        }
        //        std::cout << "***end is not aligned.***" << std::endl;
        pd64[last_word_index] = (pd64[last_word_index] & mask2) | (last_word & ~mask2);
    }


    template<size_t k>
    u64 shift_by_k_left_inside_word(u64 word, size_t start1, size_t end1) {
        assert(start1 + k <= end1);
        assert(start1 < 64);
        assert(end1 <= 64);
        assert(end1 > 0);
        const u64 lo_mask = (1ULL << start1) - 1u;
        assert(end1 > k);
        const u64 hi_mask = _bzhi_u64(-1, end1 - k);
        assert(hi_mask);
        const u64 mid_mask = _bzhi_u64(-1, end1 - start1 - k) << start1;
        u64 lo = word & lo_mask;
        u64 hi = word & ~hi_mask;
        // u64 mid_to_shift = word & (~(lo_mask << k) & );
        // u64 mid = (mid_to_shift >> k) & hi_mask;
        // u64 ns_mid = word & mid_mask;
        u64 mid = (word >> k) & mid_mask;
        assert(!(lo & mid));
        assert(!(lo & hi));
        assert(!(hi & mid));
        u64 new_word = lo | mid | hi;
        /*std::cout << std::string(80, '=') << std::endl;
        std::cout << "start1: \t" << start1 << std::endl;
        std::cout << "end1:   \t" << end1 << std::endl;
        // std::cout << "offset: \t" << offset << std::endl;
        std::cout << std::string(80, '~') << std::endl;
        std::cout << "lo:       \t" << str_bitsMani::format_word_to_string(lo) << std::endl;
        std::cout << "ns_mid:   \t" << str_bitsMani::format_word_to_string(ns_mid) << std::endl;
        std::cout << "mid:      \t" << str_bitsMani::format_word_to_string(mid) << std::endl;
        std::cout << "hi:       \t" << str_bitsMani::format_word_to_string(hi) << std::endl;
        std::cout << "word:     \t" << str_bitsMani::format_word_to_string(word) << std::endl;
        std::cout << "new_word: \t" << str_bitsMani::format_word_to_string(new_word) << std::endl;
        std::cout << std::string(80, '=') << std::endl;*/

        return new_word;
        // memcpy(mp + (start1 / 8), &new_word, bytes_to_copy);
        // return;
    }

    template<size_t k>
    void shift_by_k_left(__m512i *pd, size_t start1, size_t end1) {
        // constexpr size_t end1 = 512;
        assert(end1 <= 512);
        assert(start1 <= end1);// this is not a must
        if (start1 + k == end1)
            return;
        if (start1 >= 512 - 64) {
            assert(start1 + k < 512);
            u64 word = 0;
            memcpy(&word, (u8 *) pd + 64 - 8, 8);
            size_t new_start = start1 - (512 - 64);
            size_t new_end = end1 - (512 - 64);
            u64 new_word = shift_by_k_left_inside_word<k>(word, new_start, new_end);
            memcpy((u8 *) pd + 64 - 8, &new_word, 8);
            //            std::cout << "m0: " << std::endl;
            return;

            /* const u64 lo_mask = (1ULL << (new_start + k)) - 1u;
            const u64 hi_mask = _bzhi_u64(-1, new_end);
            assert(hi_mask);
            u64 lo = word & lo_mask;
            u64 hi = word & ~hi_mask;
            u64 mid_to_shift = word & ~_bzhi_u64(-1, new_start);
            // u64 mid = (word << k) & (hi_mask & ~lo_mask);
            u64 mid = (mid_to_shift << k) & hi_mask;
            assert(!(lo & mid));
            assert(!(lo & hi));
            assert(!(hi & mid));
            u64 new_word = lo | mid | hi;
            memcpy((u8 *) pd + 64 - 8, &new_word, 8);
            //            std::cout << "case_m1: " << std::endl;
            return; */
        }
        //        const size_t items = (end1 - start1) / k;
        auto mp = (u8 *) pd;
        const size_t total_bits_to_compare = end1 - start1;
        const size_t offset = start1 & 7;
        if (total_bits_to_compare + offset <= 64) {
            u64 word = 0;
            const size_t bytes_to_copy = (offset + total_bits_to_compare + 7) / 8;
            assert((start1 / 8) + bytes_to_copy <= 64);
            memcpy(&word, mp + (start1 / 8), bytes_to_copy);

            u64 new_word = shift_by_k_left_inside_word<k>(word, offset, offset + total_bits_to_compare);
            /* const u64 lo_mask = (1ULL << offset) - 1u;
            // assert(lo_mask);// this can't be 0;
            assert(offset + total_bits_to_compare > k);
            const u64 hi_mask = _bzhi_u64(-1, offset + total_bits_to_compare - k);
            assert(hi_mask);
            u64 lo = word & lo_mask;
            u64 hi = word & ~hi_mask;
            u64 mid_to_shift = word & _bzhi_u64(-1, offset + total_bits_to_compare);
            u64 mid = (mid_to_shift >> k) & hi_mask;
            assert(!(lo & mid));
            assert(!(lo & hi));
            assert(!(hi & mid));
            u64 new_word = lo | mid | hi; */
            memcpy(mp + (start1 / 8), &new_word, bytes_to_copy);
            //            std::cout << "m1: " << std::endl;

            /* std::cout << std::string(80, '=') << std::endl;
            std::cout << "start1: \t" << start1 << std::endl;
            std::cout << "end1:   \t" << end1 << std::endl;
            std::cout << "offset: \t" << offset << std::endl;
            std::cout << std::string(80, '~') << std::endl;
            std::cout << "lo:       \t" << str_bitsMani::format_word_to_string(lo) << std::endl;
            std::cout << "mid:      \t" << str_bitsMani::format_word_to_string(mid) << std::endl;
            std::cout << "hi:       \t" << str_bitsMani::format_word_to_string(hi) << std::endl;
            std::cout << "word:     \t" << str_bitsMani::format_word_to_string(word) << std::endl;
            std::cout << "new_word: \t" << str_bitsMani::format_word_to_string(new_word) << std::endl;
            std::cout << std::string(80, '=') << std::endl; */
            return;
        }
        //        assert(0);
        u64 *pd64 = (u64 *) pd;
        const size_t first_word_index = start1 / 64;
        const size_t last_word_index = (end1 - 1) / 64;
        // size_t temp_index = last_word_index;
        const u64 first_word = pd64[first_word_index];
        // const u64 second_word = pd64[first_word_index + 1];
        // const u64 last_word = pd64[last_word_index];

        for (int i = (int) first_word_index; i < (int) last_word_index; i++) {
            pd64[i] = (pd64[i] >> k) | (pd64[i + 1] << (64 - k));
        }

        // size_t rel_end = ((end1 - 1) & 63) + 1;
        /*std::cout << "m2" << std::endl;
        std::cout << "start1:     \t" << start1 << std::endl;
        std::cout << "end1:       \t" << end1 << std::endl;
        std::cout << "rel_start1: \t" << (start1 & 63) << std::endl;
        std::cout << "rel_end1:   \t" << rel_end << std::endl;
        std::cout << "i0:         \t" << first_word_index << std::endl;
        std::cout << "i_end:      \t" << last_word_index << std::endl;*/
        //fix first word
        const u64 start_mask = _bzhi_u64(-1, start1 & 63);
        const u64 hi = (pd64[first_word_index] & ~start_mask);
        const u64 lo = (first_word & start_mask);

        /*std::cout << "start_mask:       \t" << str_bitsMani::format_word_to_string(start_mask) << std::endl;
        std::cout << "old w0:          \t" << str_bitsMani::format_word_to_string(first_word) << std::endl;
        std::cout << "new w0:          \t" << str_bitsMani::format_word_to_string(pd64[first_word_index]) << std::endl;
        std::cout << "hi:              \t" << str_bitsMani::format_word_to_string(hi) << std::endl;
        std::cout << "lo:              \t" << str_bitsMani::format_word_to_string(lo) << std::endl;
*/
        pd64[first_word_index] = lo | hi;

        //        std::cout << "last_word_index: \t" << last_word_index << std::endl;
        //fix last word
        const size_t end_mask_index = ((end1 - 1) & 63) + 1;
        u64 end_mask = _bzhi_u64(-1, end_mask_index);
        assert(end_mask);
        u64 e_lo = ((pd64[last_word_index] & end_mask) >> k);
        u64 e_hi = (pd64[last_word_index] & ~(end_mask >> k));
        assert(!(e_lo & e_hi));
        u64 new_last_word = e_lo | e_hi;

        //        std::cout << "start_mask:       \t" << str_bitsMani::format_word_to_string(start_mask) << std::endl;
        //        std::cout << "old w1:          \t" << str_bitsMani::format_word_to_string(pd64[last_word_index]) << std::endl;
        //        std::cout << "new w1:          \t" << str_bitsMani::format_word_to_string(new_last_word) << std::endl;
        //        std::cout << "e_hi:            \t" << str_bitsMani::format_word_to_string(e_hi) << std::endl;
        //        std::cout << "e_lo:            \t" << str_bitsMani::format_word_to_string(e_lo) << std::endl;
        pd64[last_word_index] = new_last_word;
    }


    inline void shift_by_k_right_no_template(__m512i *pd, size_t start1, size_t end1, size_t k) {
        assert(end1 <= 512);
        assert(start1 <= end1);// this is not a must
        if (start1 >= 512 - 64) {
            assert(0);
            u64 word;
            memcpy(&word, (u8 *) pd + 64 - 8, 8);
            size_t new_start = start1 - (512 - 64);
            size_t new_end = end1 - (512 - 64);
            const u64 lo_mask = (1ULL << (new_start + k)) - 1u;
            const u64 hi_mask = _bzhi_u64(-1, new_end);
            assert(hi_mask);
            u64 lo = word & lo_mask;
            u64 hi = word & ~hi_mask;
            u64 mid_to_shift = word & ~_bzhi_u64(-1, new_start);
            // u64 mid = (word << k) & (hi_mask & ~lo_mask);
            u64 mid = (mid_to_shift << k) & hi_mask;
            assert(!(lo & mid));
            assert(!(lo & hi));
            assert(!(hi & mid));
            u64 new_word = lo | mid | hi;
            memcpy((u8 *) pd + 64 - 8, &new_word, 8);
            //            std::cout << "case_m1: " << std::endl;
            return;
        }
        //        const size_t items = (end1 - start1) / k;
        auto mp = (u8 *) pd;
        const size_t total_bits_to_compare = end1 - start1;
        const size_t offset = start1 & 7;
        if (total_bits_to_compare + offset <= 64) {
            u64 word = 0;
            const size_t bytes_to_copy = (offset + total_bits_to_compare + 7) / 8;
            assert((start1 / 8) + bytes_to_copy <= 64);
            memcpy(&word, mp + (start1 / 8), bytes_to_copy);

            const u64 lo_mask = (1ULL << (offset + k)) - 1u;
            // const u64 lo_mask = (1ULL << (offset + 1)) - 1u;
            // const u64 hi_mask = (1ULL << (offset + total_bits_to_compare)) - 1u;
            const u64 hi_mask = _bzhi_u64(-1, offset + total_bits_to_compare);
            assert(hi_mask);
            u64 lo = word & lo_mask;
            u64 hi = word & ~hi_mask;
            u64 mid_to_shift = word & ~_bzhi_u64(-1, offset);
            u64 mid = (mid_to_shift << k) & hi_mask;
            // u64 mid = (word << k) & (hi_mask & ~_bzhi_u64(-1, offset + k));
            // u64 mid = (word << k) & (hi_mask & ~lo_mask);
            assert(!(lo & mid));
            assert(!(lo & hi));
            assert(!(hi & mid));
            u64 new_word = lo | mid | hi;
            /* if (k == 2) {
                std::cout << std::string(80, '=') << std::endl;
                std::cout << "start1: \t" << start1 << std::endl;
                std::cout << "end1:   \t" << end1 << std::endl;
                std::cout << "offset: \t" << offset << std::endl;
                std::cout << std::string(80, '~') << std::endl;
                std::cout << "lo:       \t" << str_bitsMani::format_word_to_string(lo) << std::endl;
                std::cout << "mid:      \t" << str_bitsMani::format_word_to_string(mid) << std::endl;
                std::cout << "hi:       \t" << str_bitsMani::format_word_to_string(hi) << std::endl;
                std::cout << "word:     \t" << str_bitsMani::format_word_to_string(word) << std::endl;
                std::cout << "new_word: \t" << str_bitsMani::format_word_to_string(new_word) << std::endl;
                std::cout << std::string(80, '=') << std::endl;
            }
            */
            memcpy(mp + (start1 / 8), &new_word, bytes_to_copy);
            //            std::cout << "case0: " << std::endl;
            return;
        }
        u64 *pd64 = (u64 *) pd;
        const size_t first_word_index = start1 / 64;
        const size_t last_word_index = (end1 - 1) / 64;
        // size_t temp_index = last_word_index;
        const u64 first_word = pd64[first_word_index];
        // const u64 second_word = pd64[first_word_index + 1];
        const u64 last_word = pd64[last_word_index];

        for (int i = (int) last_word_index; i > (int) first_word_index; i--) {
            assert(i > 0);
            pd64[i] = (pd64[i] << k) | (pd64[i - 1] >> (64 - k));
        }

        //fix first word
        if ((start1 & 63) + k > 64) {
            //            std::cout << "rel-start1 + k:\t" << (start1 & 63) + k << std::endl;
            size_t rel_start = (start1 & 63);
            size_t shift_by = rel_start + k - 64;
            //            u64 temp_w1 = first_word & ~_bzhi_u64(-1, rel_start - 1);
            //            u64 temp_w1 = first_word & ~_bzhi_u64(-1, rel_start); //FIXME!!!
            u64 temp_w1 = (first_word >> rel_start);//FIXME!!!
            //            u64 lo_w1 = temp_w1 << (shift_by - 1);
            u64 lo_w1 = temp_w1 << shift_by;
#ifndef NDEBUG
            u64 bits_to_move = _bextr_u64(first_word, rel_start, 64 - rel_start);
            u64 shifted_bits = bits_to_move << shift_by;
            if (lo_w1 != shifted_bits) {
                std::cout << "lo_w1: \t" << lo_w1 << std::endl;
                std::cout << "shifted_bits: \t" << shifted_bits << std::endl;
                assert(0);
            }
#endif//!NDEBUG
            const size_t rel_index = (start1 + k) & 63;
            const u64 mask = _bzhi_u64(-1, rel_index);
            // u64 new_lo = (first_word &  >> (64 - k)) & mask;

            u64 temp_lo = temp_w1 >> (64 - k);
#ifndef NDEBUG
            u64 masked_temp_lo = temp_lo & mask;
            if (temp_lo != masked_temp_lo) {
                std::cout << "temp_lo:   \t" << temp_lo << std::endl;
                std::cout << "m_temp_lo: \t" << masked_temp_lo << std::endl;
                std::cout << std::string(80, '-') << std::endl;
                std::cout << "temp_lo:   \t" << str_bitsMani::format_word_to_string(temp_lo) << std::endl;
                std::cout << "m_temp_lo: \t" << str_bitsMani::format_word_to_string(masked_temp_lo) << std::endl;
                assert(0);
            }
#endif//!NDEBUG

            // u64 lo = temp_w1 >> (64 - k) & mask;
            // u64 lo = pd64[first_word_index + 1] & mask;
            u64 hi = pd64[first_word_index + 1] & ~mask;
            assert(!(temp_lo & hi));
            pd64[first_word_index + 1] = temp_lo | hi;
        } else {
            //            std::cout << "start != 63." << std::endl;
            u64 lo = first_word & _bzhi_u64(-1, (start1 & 63) + k);
            u64 mid = (first_word & ~_bzhi_u64(-1, start1 & 63)) << k;
            assert(!(lo & mid));
            pd64[first_word_index] = lo | mid;
            /* const u64 mask1 = _bzhi_u64(-1, (start1 + k) & 63);
            const u64 w1_shifted = first_word << k;
            pd64[first_word_index] = (first_word & mask1) | (w1_shifted & ~mask1); */
        }
        //fix last word
        const u64 mask2 = _bzhi_u64(-1, end1 & 63);
        if ((end1 & 63) == 0) {
            //            std::cout << "end is aligned." << std::endl;
            return;
        }
        //        std::cout << "***end is not aligned.***" << std::endl;
        pd64[last_word_index] = (pd64[last_word_index] & mask2) | (last_word & ~mask2);
    }
    // void insert_push_4bit_by_shift(u8 *packedArray, size_t packedSize, size_t index4, u8 item);
    template<size_t k>
    void insert_push_k_bits_item_by_shift(__m512i *pd, size_t start1, size_t end1, u32 value) {
        assert(value <= _bzhi_u64(-1, k));
        assert(end1 <= 512);
        assert(start1 + k <= end1);

        shift_by_k_right<k>(pd, start1, end1);
        bitsMani::update_bits_inside_8bytes_boundaries_safer((u8 *) pd, start1, k, value);
    }

    template<size_t k>
    void insert_pull_k_bits_item_by_shift(__m512i *pd, size_t start1, size_t end1, u32 value) {
        assert(value <= _bzhi_u64(-1, k));
        assert(end1 <= 512);
        assert(start1 + k <= end1);

        shift_by_k_left<k>(pd, start1, end1);
        bitsMani::update_bits_inside_8bytes_boundaries_safer((u8 *) pd, end1 - k, k, value);
    }
    /* template<size_t two_k>
    void insert_push_2k_bit_by_shift(u8 *packedArray, size_t packedSize, size_t packed_index, u8 item) {
        //FIXME.
        assert(two_k < 8);// use memmove.
        assert((two_k & 1) == 0);
        assert(item <= _bzhi_u32(-1, two_k));
        assert(packed_index * two_k < packedSize);
        assert(2 <= packedSize);

#ifndef NDEBUG
        u8 backup_a[packedSize];
        memcpy(backup_a, packedArray, packedSize);
#endif
        size_t start4 = index4;
        size_t end4 = packedSize * 2 - 1;
        size_t size64 = (packedSize + 7) / 8;
        shift_arr_4bits_right_att_wrapper((u64 *) packedArray, start4, end4, size64);
        size_t byte_index = index4 / 2;
        if (!(index4 & 1)) {
            //            A[1]++;
            packedArray[byte_index] = (packedArray[byte_index] & 0xf0) | item;
        } else {
            //            A[2]++;
            packedArray[byte_index] = (packedArray[byte_index] & 0xf) | (item << 4u);
        }

        assert(check::validate_insert_push_4bit(backup_a, packedArray, packedSize, index4, item));
        return;
    } */

    inline void pack_array_gen_k_with_offset(__m512i *pd, const u32 *unpacked_array, size_t start1, size_t items, size_t k) {
        auto pd8 = (u8 *) pd;
        size_t first_byte_index = start1 / 8;
        u8 backup = pd8[first_byte_index];
        Shift_op::pack_array_gen_k(pd8 + first_byte_index, unpacked_array, items, k);
        const size_t offset = start1 & 7;
        shift_by_k_right_no_template(pd, start1 / 8 * 8, start1 + items * k, k);
        u8 mask = (1 << offset) - 1;
        pd8[first_byte_index] = (backup & mask) | (pd8[first_byte_index] & ~mask);
        //        bitsMani::update_bits_inside_8bytes_boundaries_safer(pd8 + first_byte_index, 0, )
    }
    template<typename T, typename S>
    size_t min_failed_memcmp_index(const T *a, const S *b, size_t bytes_to_compare) {
        auto a8 = (const u8 *) a;
        auto b8 = (const u8 *) b;
        for (size_t i = 0; i < bytes_to_compare; ++i) {
            if (a8[i] != b8[i])
                return i;
        }
        return bytes_to_compare;
    }

    inline u64 extract_64bits_safe(const __m512i *pd, size_t start1, size_t k) {
        assert(k <= 64);
        const size_t length = k;
        const u64 mask = _bzhi_u64(-1, length);
        const size_t offset = start1 & 7;
        u64 word;
        const size_t first_byte = start1 / 8;
        assert(first_byte < 64);
        const size_t bytes_to_copy = (first_byte <= 64 - 8) ? 8 : 64 - first_byte;
        assert(bytes_to_copy <= 8);
        //        std::cout << "bytes_to_copy: \t" << bytes_to_copy << std::endl;
        memcpy(&word, (u8 *) pd + first_byte, bytes_to_copy);
        word >>= offset;
        if (offset + length <= 64) {
            return word & mask;
        } else {
            u64 hi = ((u8 *) pd)[first_byte] << (64 - offset);
            return mask | hi;
        }
    }

    inline void update_bits_inside_u64_boundaries(__m512i *pd, size_t start1, size_t length, u64 value) {
        assert(length <= 64);
        assert(value <= _bzhi_u64(-1, length));
        const u64 mask = _bzhi_u64(-1, length);
        const size_t word_index = start1 / 64;
        const size_t offset = start1 & 63;
        u64 *mp = (u64 *) pd + word_index;
        u64 word = mp[0];
        u64 shifted_mask = mask << offset;
        u64 mid = value << offset;
        #ifndef NDEBUG
            u64 inner = mid;
            u64 outer = (word & ~shifted_mask);
            assert(!(inner & outer));
            assert((inner & shifted_mask) == inner);
        #endif //!NDEBUG
        u64 new_word = (word & ~shifted_mask) | mid;
        mp[0] = new_word;
    }

    inline void update_bits_inside_8bytes_boundaries(__m512i *pd, size_t start1, size_t length, u64 value) {
        assert(start1 <= (512 - 64));
        //        if (start1 >= (512 - 64)) {
        //            return update_bits_inside_u64_boundaries(pd, start1, length, value);
        //        }
        assert(length <= 64);
        assert(value <= _bzhi_u64(-1, length));
        const u64 mask = _bzhi_u64(-1, length);
        const size_t offset = start1 & 7;
        const size_t byte_index = start1 / 8;
        u64 word;
        //        assert()
        auto mp = (u8 *) pd + byte_index;
        memcpy(&word, mp, 8);
        u64 shifted_mask = mask << offset;
        u64 mid = value << offset;
        u64 new_word = (word & ~shifted_mask) | mid;
        memcpy(mp, &new_word, 8);
    }

    inline void update_64bits_safe(__m512i *pd, size_t start1, size_t length, u64 value) {
        assert(length <= 64);
        assert(start1 + length <= 512);
        assert(value <= _bzhi_u64(-1, length));
        if (start1 >= 512 - 64) {
            return update_bits_inside_u64_boundaries(pd, start1, length, value);
        }
        if (length + (start1 & 7) <= 64)
            return update_bits_inside_8bytes_boundaries(pd, start1, length, value);
        else {
            const size_t sub_len1 = length - 8;
            const u64 val1 = value & _bzhi_u64(-1, sub_len1);
            const u64 val2 = value >> sub_len1;
            update_bits_inside_8bytes_boundaries(pd, start1, sub_len1, val1);
            update_bits_inside_8bytes_boundaries(pd, start1 + sub_len1, 8, val2);
        }
    }


    inline bool compare_bits_ranged_safe(const __m512i *pd, u8 rem, size_t rem_length, size_t start_index1, size_t end_index1) {
        assert(((end_index1 - start_index1) % rem_length) == 0);
        const size_t items = (end_index1 - start_index1) / rem_length;
        if (start_index1 >= 512 - 64) {
            size_t offset = start_index1 - (512 - 64);
            u64 word = ((const u64 *) pd)[7];
            word >>= offset;
            return bitsMani::compare_k_packed_items(word, rem, rem_length, items);
        }
        //        const size_t items = (end_index1 - start_index1) / rem_length;
        auto mp = (const u8 *) pd;
        const size_t total_bits_to_compare = end_index1 - start_index1;
        const size_t offset = start_index1 & 7;
        if (total_bits_to_compare + offset <= 64) {
            u64 word = 0;
            memcpy(&word, mp + (start_index1 / 8), 8);
            word >>= offset;
            return bitsMani::compare_k_packed_items(word, rem, rem_length, items);
        }
        size_t first_part_bits = (64 - offset) / rem_length * rem_length;
        assert(first_part_bits);
        auto temp = bitsMani::cmp_bits_inside_un_aligned_single_word((const u64 *) pd, rem, rem_length, start_index1, start_index1 + first_part_bits);
        auto rest = (start_index1 + first_part_bits < end_index1) && compare_bits_ranged_safe(pd, rem, rem_length, start_index1 + first_part_bits, end_index1);
        return temp or rest;
    }
}// namespace Shift_pd

namespace Shift_pd::check {

    inline void random_filler(__m512i *pd) {
        for (size_t i = 0; i < (512 / 32); i++) {
            ((u32 *) pd)[i] = random();
        }
    }

    inline void print_helper(const __m512i *pd0, const __m512i *pd1, size_t beg, size_t end, size_t k, size_t backup_index, size_t s_byte, size_t offset, size_t reps) {

        // size_t s_byte = beg / 8;
        size_t bytes0 = std::min<size_t>(64u - s_byte, 16);
        // size_t bytes = ((64 - s_byte) >= 16) ? 16 : (64 - s_byte);
        // assert(bytes == bytes0);
        size_t word_index = backup_index / 64;
        std::string w_m1_after = str_bitsMani::format_word_to_string(((const u64 *) pd1)[word_index - 1]);
        std::string w0_after = str_bitsMani::format_word_to_string(((const u64 *) pd1)[word_index]);
        std::string w1_after = str_bitsMani::format_word_to_string(((const u64 *) pd1)[word_index + 1]);

        std::string w_m1_before = str_bitsMani::format_word_to_string(((const u64 *) pd0)[word_index - 1]);
        std::string w0_before = str_bitsMani::format_word_to_string(((const u64 *) pd0)[word_index]);
        std::string w1_before = str_bitsMani::format_word_to_string(((const u64 *) pd0)[word_index + 1]);

        assert(word_index * 64 <= backup_index);
        size_t offset_from_first_printed_bit_to_error = backup_index - word_index * 64;
        std::cout << std::string(80, '=') << std::endl;
        std::cout << "first_p_index: \t" << word_index * 64 << std::endl;
        std::cout << "offset_index:  \t" << offset_from_first_printed_bit_to_error << std::endl;
        std::cout << std::string(80, '-') << std::endl;
        std::cout << "w_m1_before: \t" << w_m1_before << std::endl;
        std::cout << "w_m1_after:  \t" << w_m1_after << std::endl;
        std::cout << "" << std::endl;
        std::cout << "w0_before:   \t" << w0_before << std::endl;
        std::cout << "w0_after:    \t" << w0_after << std::endl;
        std::cout << "" << std::endl;
        std::cout << "w1_before:   \t" << w1_before << std::endl;
        std::cout << "w1_after:    \t" << w1_after << std::endl;
        std::cout << std::string(80, '-') << std::endl;
        std::cout << "beg:    \t" << beg << std::endl;
        std::cout << "end:    \t" << end << std::endl;
        std::cout << "reps:   \t" << reps << std::endl;
        std::cout << std::string(80, '=') << std::endl;


        auto s_after = str_bitsMani::str_array_as_memory_no_delim((const u8 *) pd1 + s_byte, bytes0);
        auto s_backup = str_bitsMani::str_array_as_memory_no_delim((const u8 *) pd0 + s_byte, bytes0);

        auto s_after_del = str_bitsMani::str_array_as_memory((const u8 *) pd1 + s_byte, bytes0);
        auto s_backup_del = str_bitsMani::str_array_as_memory((const u8 *) pd0 + s_byte, bytes0);

        std::cout << "first_printed_bit: \t" << s_byte * 8 << std::endl;
        std::cout << "s_byte: \t" << s_byte << std::endl;
        std::cout << "offset: \t" << offset << std::endl;
        std::cout << "beg:    \t" << beg << std::endl;
        std::cout << "end:    \t" << end << std::endl;
        std::cout << "backup_index:  \t" << backup_index << std::endl;
        std::cout << "shift_k:\t" << k << std::endl;
        std::cout << "reps:   \t" << reps << std::endl;
        std::cout << std::string(80, '=') << std::endl;
        std::cout << "s_backup:    " << s_backup << std::endl;
        std::cout << "s_after:     " << s_after << std::endl;
        std::cout << "s_backup_del: \n"
                  << s_backup_del << std::endl;
        std::cout << "s_after_del:  \n"
                  << s_after_del << std::endl;
        assert(0);
    }

    template<size_t shift_k>
    inline void test_shift_by_2_right() {
        __m512i pd = {0};
        for (size_t i = 0; i < (512 / 32); i++) {
            ((u32 *) &pd)[i] = random();
        }

        const __m512i pd0 = pd;
        for (size_t reps = 0; reps < (1ULL << 16); reps++) {
            size_t beg = 0;
            size_t end = 0;
            while (true) {
                beg = random() % (512 + 1);
                end = random() % (512 + 1);
                // bool cond1 = (beg + shift_k <= end);
                // bool cond2 = (end - beg <= 96);// remove this condition;
                bool cond3 = (beg + shift_k < end);
                // if (cond1 and cond2 and cond3) {
                if (cond3) {
                    break;
                }
            }

            /* std::cout << std::string(80, '*') << std::endl;
            std::cout << "beg: " << beg << std::endl;
            std::cout << "end: " << end << std::endl;
            std::cout << std::string(80, '*') << std::endl; */
            pd = pd0;
            assert(memcmp(&pd, &pd0, 64) == 0);
            // constexpr size_t shift_k = 1;
            shift_by_k_right<shift_k>(&pd, beg, end);

            for (size_t i = 0; i < beg; i++) {
                bool a_bit = bitsMani::is_single_bit_set((const u64 *) &pd, i, 8);
                bool val_bit = bitsMani::is_single_bit_set((const u64 *) &pd0, i, 8);

                assert(a_bit == val_bit);

                /*if (a_bit == val_bit)
                    continue;
                size_t s_byte = i / 8;
                size_t bytes0 = std::min<size_t>(64u - s_byte, 16);
                assert(bytes0 <= 16);
                auto s0 = str_bitsMani::str_array_as_memory_no_delim((const u8 *) &pd + s_byte, bytes0);
                auto s1 = str_bitsMani::str_array_as_memory_no_delim((const u8 *) &pd0 + s_byte, bytes0);

                std::cout << "s_byte: \t" << s_byte << std::endl;
                std::cout << "offset: \t" << i - s_byte * 8 << std::endl;
                std::cout << "i:      \t" << i << std::endl;
                std::cout << "beg:    \t" << beg << std::endl;
                std::cout << "end:    \t" << end << std::endl;
                std::cout << "shift_k:\t" << shift_k << std::endl;
                std::cout << "reps:   \t" << reps << std::endl;
                std::cout << std::string(80, '=') << std::endl;
                std::cout << "s1: " << s1 << std::endl;
                std::cout << "s0: " << s0 << std::endl;
                assert(0);*/
            }
            const size_t shift_interval = (end - beg) - shift_k;
            assert(shift_interval <= 512);

            /*
             * Testing the first shift_k bits after begin. Let's say those bits value is not defined, although making them zeo will be preferred.
             *
             * those bits value is
             * for (size_t j = 0; j < shift_k; j++) {
                if (beg + j + shift_k >= end) {
                    break;
                }
                bool a_bit = bitsMani::is_single_bit_set((const u64 *) &pd, beg + j + shift_k, 8);
                bool val_bit = bitsMani::is_single_bit_set((const u64 *) &pd, beg + j, 8);
                //The error is not actually an error. this is a cyclic problem. if x == x+ 2. then this will imply x + 2 == x+ 4, which is not necessarily true. In other words, this specific error is in the test, and not in the implementation.
                // assert(a_bit == val_bit);

                if (a_bit == val_bit)
                    continue;


                size_t s_byte = beg / 8;
                size_t offset = beg - s_byte * 8;
                print_helper(&pd0, &pd, beg, end, shift_k, beg + j, s_byte, offset, reps);
                */
            /* size_t bytes0 = std::min<size_t>(64u - s_byte, 16);
                size_t bytes = ((64 - s_byte) >= 16) ? 16 : (64 - s_byte);
                assert(bytes == bytes0);
                auto s_after = str_bitsMani::str_array_as_memory_no_delim((const u8 *) &pd + s_byte, bytes);
                auto s_backup = str_bitsMani::str_array_as_memory_no_delim((const u8 *) &pd0 + s_byte, bytes);

                auto s_after_del = str_bitsMani::str_array_as_memory((const u8 *) &pd + s_byte, bytes);
                auto s_backup_del = str_bitsMani::str_array_as_memory((const u8 *) &pd0 + s_byte, bytes);

                std::cout << "s_byte: \t" << s_byte << std::endl;
                std::cout << "first_printed_bit: \t" << s_byte * 8 << std::endl;
                std::cout << "offset: \t" << beg - s_byte * 8 << std::endl;
                std::cout << "beg:    \t" << beg << std::endl;
                std::cout << "end:    \t" << end << std::endl;
                std::cout << "j:      \t" << j << std::endl;
                std::cout << "shift_k:\t" << shift_k << std::endl;
                std::cout << "reps:   \t" << reps << std::endl;
                std::cout << std::string(80, '=') << std::endl;
                std::cout << "s_backup:    " << s_backup << std::endl;
                std::cout << "s_after:     " << s_after << std::endl;
                std::cout << "s_backup_del: \n"
                          << s_backup_del << std::endl;
                std::cout << "s_after_del:  \n"
                          << s_after_del << std::endl;
                assert(0); */
            /*

                assert(a_bit == val_bit);
            }*/
            for (size_t j = 0; j < shift_interval; j++) {
                bool a_bit = bitsMani::is_single_bit_set((const u64 *) &pd, beg + j + shift_k, 8);
                bool val_bit = bitsMani::is_single_bit_set((const u64 *) &pd0, beg + j, 8);
                assert(a_bit == val_bit);

                /* if (a_bit == val_bit)
                    continue;

                size_t s_byte;
                size_t offset;
                if (j < 8) {
                    s_byte = beg / 8;
                    offset = beg & 7;
                } else {
                    s_byte = (beg + j) / 8;
                    offset = (beg + j) - (s_byte * 8);
                }
                print_helper(&pd0, &pd, beg, end, shift_k, beg + j, s_byte, offset, reps);*/
                /* // size_t first_byte_index
                auto s0 = str_bitsMani::str_array_as_memory((const u8 *) &pd + beg / 8 * 8, (shift_interval + 23) / 8);
                auto s1 = str_bitsMani::str_array_as_memory((const u8 *) &pd0 + beg / 8 * 8, (shift_interval + 23) / 8);

                std::cout << "beg: \t" << beg << std::endl;
                std::cout << "end: \t" << end << std::endl;
                std::cout << "j:   \t" << j << std::endl;
                std::cout << std::string(80, '=') << std::endl;
                std::cout << "s0: " << s0 << std::endl;
                std::cout << "s1: " << s1 << std::endl;
                assert(0); */
            }
            for (size_t j = end; j < 512; j++) {
                bool a_bit = bitsMani::is_single_bit_set((const u64 *) &pd, j, 8);
                bool val_bit = bitsMani::is_single_bit_set((const u64 *) &pd0, j, 8);

                assert(a_bit == val_bit);

                /* if (a_bit == val_bit)
                    continue;

                size_t s_byte = j / 8 * 8;
                auto s0 = str_bitsMani::str_array_as_memory((const u8 *) &pd + s_byte, 12);
                auto s1 = str_bitsMani::str_array_as_memory((const u8 *) &pd0 + s_byte, 12);

                std::cout << "beg: \t" << beg << std::endl;
                std::cout << "end: \t" << end << std::endl;
                std::cout << "j:   \t" << j << std::endl;
                std::cout << std::string(80, '=') << std::endl;
                std::cout << "s0: " << s0 << std::endl;
                std::cout << "s1: " << s1 << std::endl;
                assert(0); */
            }
            //            std::cout << "reps: " << reps << std::endl;
        }
        std::cout << "Done with : \t" << shift_k << std::endl;
    }

    template<size_t shift_k>
    inline void test_shift_by_k_left() {
        __m512i pd = {0};
        for (size_t i = 0; i < (512 / 32); i++) {
            ((u32 *) &pd)[i] = random();
        }

        const __m512i pd0 = pd;
        for (size_t reps = 0; reps < (1ULL << 16); reps++) {
            size_t beg = 0;
            size_t end = 0;
            while (true) {
                beg = random() % (512 + 1);
                end = random() % (512 + 1);
                // bool cond1 = (beg + shift_k <= end);
                //                bool cond2 = ((end - beg + (beg & 7)) <= 64);
                //                bool cond2 = (end - beg <= 96);
                //                bool cond2 = true;
                bool cond3 = (beg + shift_k < end);
                //                bool cond4 = (beg + shift_k < end);
                // if (cond1 and cond2 and cond3) {
                if (cond3) {
                    break;
                }
            }

            /* std::cout << std::string(80, '*') << std::endl;
            std::cout << "beg: " << beg << std::endl;
            std::cout << "end: " << end << std::endl;
            std::cout << std::string(80, '*') << std::endl; */
            pd = pd0;
            assert(memcmp(&pd, &pd0, 64) == 0);
            // constexpr size_t shift_k = 1;
            shift_by_k_left<shift_k>(&pd, beg, end);
            for (size_t i = 0; i < beg; i++) {
                bool a_bit = bitsMani::is_single_bit_set((const u64 *) &pd, i, 8);
                bool val_bit = bitsMani::is_single_bit_set((const u64 *) &pd0, i, 8);
                //                assert(a_bit == val_bit);

                if (a_bit == val_bit)
                    continue;
                size_t s_byte = i / 8;
                size_t bytes0 = std::min<size_t>(64u - s_byte, 16);
                assert(bytes0 <= 16);
                auto s0 = str_bitsMani::str_array_as_memory_no_delim((const u8 *) &pd0 + s_byte, bytes0);
                auto s1 = str_bitsMani::str_array_as_memory_no_delim((const u8 *) &pd + s_byte, bytes0);

                std::cout << "s_byte: \t" << s_byte << std::endl;
                std::cout << "offset: \t" << i - s_byte * 8 << std::endl;
                std::cout << "i:      \t" << i << std::endl;
                std::cout << "beg:    \t" << beg << std::endl;
                std::cout << "end:    \t" << end << std::endl;
                std::cout << "shift_k:\t" << shift_k << std::endl;
                std::cout << "reps:   \t" << reps << std::endl;
                std::cout << std::string(80, '=') << std::endl;
                std::cout << "s0: " << s0 << std::endl;
                std::cout << "s1: " << s1 << std::endl;
                assert(0);
            }
            const size_t shift_interval = (end - beg) - shift_k;
            assert(shift_interval <= 512);
            for (size_t j = 0; j < shift_interval; j++) {

                // bool a_bit = bitsMani::is_single_bit_set((const u64 *) &pd0, (end - shift_k) - j, 8);
                // bool val_bit = bitsMani::is_single_bit_set((const u64 *) &pd, end - j, 8);
                bool a_bit = bitsMani::is_single_bit_set((const u64 *) &pd0, beg + shift_k + j, 8);
                bool val_bit = bitsMani::is_single_bit_set((const u64 *) &pd, beg + j, 8);
                // assert(a_bit == val_bit);
                if (a_bit == val_bit)
                    continue;

                size_t s_byte = (beg + j) / 8;
                size_t offset = (beg + j) & 7;
                // u64 w0, w2;
                // memcpy(&w2, (const u8 *) &pd + s_byte, 8);
                // memcpy(&w0, (const u8 *) &pd0 + s_byte, 8);
                // auto s = str_bitsMani::format_2words_and_xor(w0, w2);
                auto s0 = str_bitsMani::str_array_as_memory((const u8 *) &pd0 + s_byte, 8);
                auto s1 = str_bitsMani::str_array_as_memory((const u8 *) &pd + s_byte, 8);
                size_t rel_beg = beg - s_byte * 8;
                size_t rel_end = end - s_byte * 8;

                std::cout << "beg:    \t" << beg << std::endl;
                std::cout << "end:    \t" << end << std::endl;
                std::cout << std::string(80, '~') << std::endl;
                std::cout << "rel_beg:\t" << rel_beg << std::endl;
                std::cout << "rel_end:\t" << rel_end << std::endl;
                std::cout << std::string(80, '~') << std::endl;
                std::cout << "offset: \t" << offset << std::endl;
                std::cout << "s_byte: \t" << s_byte << std::endl;
                std::cout << "j:      \t" << j << std::endl;
                std::cout << std::string(80, '=') << std::endl;
                // std::cout << s << std::endl;
                std::cout << "s0: \t\t" << s0;// << std::endl;
                std::cout << "s1: \t\t" << s1 << std::endl;
                assert(0);
            }
            for (size_t j = end; j < 512; j++) {
                bool a_bit = bitsMani::is_single_bit_set((const u64 *) &pd, j, 8);
                bool val_bit = bitsMani::is_single_bit_set((const u64 *) &pd0, j, 8);

                assert(a_bit == val_bit);
            }
        }
        std::cout << "Done with : \t" << shift_k << std::endl;
    }


    template<size_t k>
    bool test_insert_push_k_bits_item_ultra_naive(__m512i *pd, size_t start1, size_t end1, u32 value) {
        assert(value <= _bzhi_u32(-1, k));
        assert(end1 <= 512);
        assert(start1 + k <= end1);


        const __m512i pd0 = *pd;

        // u8 backup_a[packedSize];
        // memcpy(backup_a, (const u8*)pd, 64);

        const size_t total_bits = (end1 - start1);
        assert(!(total_bits % k));
        const size_t items = total_bits / k;

        u32 val_up_arr[items + 1];
        std::fill(val_up_arr, val_up_arr + items + 1, 0);
        assert((start1 + items * k) == end1);
        Shift_op::unpack_array_gen_k_with_offset(val_up_arr + 1, (const u8 *) &pd0, items, k, start1);
        val_up_arr[0] = value;

        //
        insert_push_k_bits_item_by_shift<k>(pd, start1, end1, value);
        u32 att_up_arr[items + (items == 1)];
        Shift_op::unpack_array_gen_k_with_offset(att_up_arr, (const u8 *) pd, items, k, start1);

        for (size_t i = 0; i < items; i++) {
            u32 a_res = att_up_arr[i];
            u32 v_res = val_up_arr[i];
            if (att_up_arr[i] != val_up_arr[i]) {
                std::cout << "start1: \t" << start1 << std::endl;
                std::cout << "end1:   \t" << end1 << std::endl;
                std::cout << std::string(80, '~') << std::endl;
                std::cout << "i: " << i << std::endl;
                std::cout << "a_res: " << a_res << std::endl;
                std::cout << "v_res: " << v_res << std::endl;
                if (!i) {
                    std::cout << "value: " << value << std::endl;
                }
                assert(0);
            }
        }

        return true;
        /* // size_t first_byte = start1 / 8;
        // size_t end_lim = (end + 7) / 8 * 8;
        // size_t last_byte_p1 = end_lim / 8;
        // size_t bytes_range = last_byte_p1 - first_byte;
        // const size_t unpack_size = bytes_range; */
    }

    template<size_t k>
    void test_insert_push_k_bit() {
        // size_t reps = 512;
        __m512i pd = {0};
        random_filler(&pd);
        // const __m512i pd0 = pd;
        for (size_t i = 0; i < 256; i++) {
            for (size_t j = 0; j < 256; j++) {
                __m512i att_pd = pd;

                size_t beg = 0;
                size_t end = 0;
                while (true) {
                    beg = random() % 512;
                    end = random() % 512 + 1;
                    bool cond1 = (beg + k <= end);
                    bool cond2 = ((end - beg) % k) == 0;
                    if (cond1 and cond2) {
                        break;
                    }
                }

                u32 value = random() & _bzhi_u32(-1, k);
                bool res = test_insert_push_k_bits_item_ultra_naive<k>(&att_pd, beg, end, value);
                assert(res);
            }
        }
    }


}// namespace Shift_pd::check

// template<typename T>
// void my_stable_sort(T* a, size_t size, f_cmp)

#endif//MULTI_LEVEL_HASH_SHIFT_OP_HPP