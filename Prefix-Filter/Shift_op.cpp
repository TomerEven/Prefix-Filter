//
// Created by tomer on 15/06/2021.
//


#include "Shift_op.hpp"

namespace Shift_op {
    /**
     * @brief
     *      index = begin * shift / slot_size;
     *      r_begin = (begin * shift) % slot_size
     *      r_end = (begin * shift) % slot_size
     *      set a[index] = a[index][:r_begin - shift] + (a[index][r_begin:r_end - shift] + a[index][r_end - shift:r_end]) + a[index][r_end:];
     *
     *      1, 2, 3, b, 5, 6, 7, 8, e, 9, 0
     *
     *      1, 2, b, 5, 6, 7, 8, e, e, 9, 0
     *
     *
     * @b {word Overwrite a[index][r_begin - shift :r_begin] (I think).}
     * @param a
     * @param begin shift max_cap index - If each slot in the array @param a is of max_cap @param shift, then begin is
     * an index of an item in the array.
     * Stated differently, move bits between [begin * shift, end * shift) left.
     * @param end
     * @param a_size
     */
    void shift_arr_4bits_left_inside_single_word_robuster(uint64_t *a, size_t begin, size_t end, size_t a_size) {
        if (begin >= end) return;
        assert(begin % 16);
        constexpr unsigned slot_size = sizeof(a[0]) * CHAR_BIT;
        constexpr unsigned shift = 4u;
        constexpr unsigned slot_sh_capacity = slot_size / shift;

        assert((end % 16 == 0) or (begin & _bzhi_u64(-1, shift)) <= (end & _bzhi_u64(-1, shift)));
        size_t index = begin / slot_sh_capacity;
        assert(index == ((end - 1) / slot_sh_capacity));

        size_t rel_begin = (begin * shift) & (slot_size - 1);
        size_t rel_end = (end * shift) & (slot_size - 1);
        uint64_t lo_mask = _bzhi_u64(-1, rel_begin - shift);
        uint64_t lo = a[index] & lo_mask;

        if (rel_end == 0) {
            uint64_t mid = (a[index] >> shift) & (~lo_mask);
            assert(!(lo & mid));
            a[index] = lo | mid;
            return;
        }
        assert(rel_begin < rel_end);

        uint64_t hi_mask = _bzhi_u64(-1, rel_end - shift);
        uint64_t mid = (a[index] >> shift) & ((~lo_mask) & hi_mask);
        uint64_t hi = a[index] & ~hi_mask;
        assert(!(lo & mid) and !(lo & hi) and !(mid & hi));
        a[index] = lo | mid | hi;
    }

    void shift_arr_4bits_left_att_helper(uint64_t *a, size_t begin, size_t end, size_t a_size) {
        if (begin == end) return;
        constexpr unsigned slot_size = sizeof(a[0]) * CHAR_BIT;
        constexpr unsigned shift = 4u;
        constexpr unsigned slot_sh_capacity = slot_size / shift;
        assert(slot_sh_capacity * shift == slot_size);

        size_t begin_word_index = begin >> shift;
        assert(begin_word_index == (begin / slot_sh_capacity));
        size_t rel_begin = begin & (slot_sh_capacity - 1u);
        assert(rel_begin);
        size_t end_m1_word_index = (end - 1u) >> shift;


        if (begin_word_index == end_m1_word_index) {
            // std::cout << "hh0" << std::endl;
            return shift_arr_4bits_left_inside_single_word_robuster(a, begin, end, a_size);
        }

        size_t temp_i = begin_word_index;
        size_t last_i = end_m1_word_index;
        uint64_t first_word_mask = _bzhi_u64(-1, (rel_begin - 1) * shift);
        uint64_t first_word_lo = a[temp_i] & first_word_mask;

        for (; temp_i < last_i; temp_i++) {
            auto lo = a[temp_i] >> shift;
            assert(temp_i + 1 < a_size);
            auto hi = (a[temp_i + 1] << (slot_size - shift));
            assert((lo & hi) == 0);
            a[temp_i] = lo | hi;
        }

        // Dealing with the last word.
        size_t new_begin = end_m1_word_index * slot_sh_capacity + 1;
        shift_arr_4bits_left_inside_single_word_robuster(a, new_begin, end, a_size);

        /**Dealing with the first word. */
        uint64_t first_word_hi = a[begin_word_index] & ~first_word_mask;
        assert((first_word_lo & first_word_hi) == 0);
        a[begin_word_index] = first_word_lo | first_word_hi;
    }

    /**
     * @brief 
     * I think I have the following inconsistently:
     * sometimes the last item is doubled, and sometimes the second appearance is masked out (to zeros).
     * It might be related to the fact that that end is """equal""" to a_size (end * 4 == a_size * 64);
     * 
     * @param a 
     * @param begin 
     * @param end 
     * @param a_size 
     */
    void shift_arr_4bits_left_att_wrapper(uint64_t *a, size_t begin, size_t end, size_t a_size) {
        // ZoneScopedN("Shift-4Left");
        assert(end * 4 <= a_size * 64);
        assert(begin <= end);
        if (begin == end) return;
        constexpr unsigned slot_size = sizeof(a[0]) * CHAR_BIT;
        constexpr unsigned shift = 4u;
        constexpr unsigned slot_sh_capacity = slot_size / shift;
        assert(slot_sh_capacity * shift == slot_size);

        //todo: begin += !begin;

        size_t rel_begin = begin & (slot_sh_capacity - 1);
        if (rel_begin) {
            // std::cout << "h0" << std::endl;
            return shift_arr_4bits_left_att_helper(a, begin, end, a_size);
        }

        if (begin == 0) {
            // std::cout << "h1" << std::endl;
            return shift_arr_4bits_left_att_helper(a, begin + 1, end, a_size);
        }

        // std::cout << "h2" << std::endl;
        size_t begin_word_index = begin / slot_sh_capacity;
        uint64_t last_bits = a[begin_word_index];
        shift_arr_4bits_left_att_helper(a, begin + 1, end, a_size);
        uint64_t lo = a[begin_word_index - 1] & _bzhi_u64(-1, slot_size - shift);
        uint64_t hi = last_bits << (slot_size - shift);
        assert(!(lo & hi));
        a[begin_word_index - 1] = lo | hi;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    void shift_arr_1bit_left_inside_single_word_robuster(uint64_t *a, size_t begin, size_t end) {
        if (begin >= end) return;
        assert(begin % 64);
        constexpr unsigned slot_size = 64;
        constexpr unsigned slot_mask = 63;
        constexpr unsigned shift = 1u;
        constexpr unsigned items_in_slots = slot_size / shift;

        /** Assert on one of the following:
         * 1) rel_begin <= rel_end
         * 2) end points to the next word.
        */
        assert((!(end & slot_mask)) or ((begin & slot_mask) <= (end & slot_mask)));
        size_t index = begin / items_in_slots;
        assert(index == ((end - 1) / items_in_slots));

        size_t rel_begin = (begin * shift) & slot_mask;
        size_t rel_end = (end * shift) & slot_mask;
        uint64_t lo_mask = _bzhi_u64(-1, rel_begin - shift);
        uint64_t lo = a[index] & lo_mask;

        if (rel_end == 0) {
            uint64_t mid = (a[index] >> shift) & (~lo_mask);
            assert(!(lo & mid));
            a[index] = lo | mid;
            return;
        }
        assert(rel_begin < rel_end);

        uint64_t hi_mask = _bzhi_u64(-1, rel_end);
        //        uint64_t mid = (a[index] >> shift) & ((~lo_mask) & hi_mask);
        uint64_t mid = (a[index] & ((~lo_mask) & hi_mask)) >> shift;
        uint64_t hi = a[index] & ~hi_mask;
        assert(!(lo & mid) and !(lo & hi) and !(mid & hi));
        a[index] = lo | mid | hi;
    }

    void shift_arr_1bit_left_att_helper(uint64_t *a, size_t begin, size_t end, size_t a_size) {
        if (begin == end) return;
        constexpr unsigned slot_size = sizeof(a[0]) * CHAR_BIT;
        constexpr unsigned shift = 1u;
        constexpr unsigned slot_mask = slot_size - 1u;
        constexpr unsigned items_in_slots = slot_size;
        //        assert(items_in_slots * shift == slot_size);

        size_t begin_word_index = begin / slot_size;
        size_t rel_begin = begin & slot_mask;
        assert(rel_begin);
        size_t end_m1_word_index = (end - 1u) / slot_size;


        if (begin_word_index == end_m1_word_index) {
            // std::cout << "hh0" << std::endl;
            return shift_arr_1bit_left_inside_single_word_robuster(a, begin, end);
        }

        size_t temp_i = begin_word_index;
        size_t last_i = end_m1_word_index;
        uint64_t first_word_mask = _bzhi_u64(-1, (rel_begin - 1) * shift);
        uint64_t first_word_lo = a[temp_i] & first_word_mask;

        for (; temp_i < last_i; temp_i++) {
            //FIXME:
            auto lo = a[temp_i] >> shift;
            assert(temp_i + 1 < a_size);
            auto hi = (a[temp_i + 1] << (slot_size - shift));
            assert((lo & hi) == 0);
            a[temp_i] = lo | hi;
        }

        // Dealing with the last word.
        size_t new_begin = end_m1_word_index * items_in_slots + 1;
        shift_arr_1bit_left_inside_single_word_robuster(a, new_begin, end);

        /**Dealing with the first word. */
        uint64_t first_word_hi = a[begin_word_index] & ~first_word_mask;
        assert((first_word_lo & first_word_hi) == 0);
        a[begin_word_index] = first_word_lo | first_word_hi;
    }

    void shift_arr_1bit_left_att_wrapper(uint64_t *a, size_t begin, size_t end, size_t a_size) {
        assert(end <= a_size * 64);
        assert(begin <= end);
        if (begin == end) return;
        constexpr unsigned slot_size = sizeof(a[0]) * CHAR_BIT;
        constexpr unsigned shift = 1u;
        constexpr unsigned slot_mask = slot_size - 1u;
        constexpr unsigned items_in_slots = slot_size;
        //        assert(items_in_slots * shift == slot_size);

        size_t rel_begin = begin & slot_mask;
        if (rel_begin) {
            // std::cout << "h0" << std::endl;
            return shift_arr_1bit_left_att_helper(a, begin, end, a_size);
        }

        if (begin == 0) {
            // std::cout << "h1" << std::endl;
            //            a[0] >>= 1u;
            return shift_arr_1bit_left_att_helper(a, begin + 1, end, a_size);
        }

        /** The last item in the word before @param begin_word_index needs to be updated to the value of
         * @param begin_word_index first item
         */
        // std::cout << "h2" << std::endl;
        size_t begin_word_index = begin / items_in_slots;
        uint64_t last_bits = a[begin_word_index];
        shift_arr_1bit_left_att_helper(a, begin + 1, end, a_size);
        uint64_t lo = a[begin_word_index - 1] & _bzhi_u64(-1, slot_size - shift);
        //        uint64_t lo = (a[begin_word_index - 1] << 1u) >> 1u;
        uint64_t hi = last_bits << (slot_size - shift);
        assert(!(lo & hi));
        a[begin_word_index - 1] = lo | hi;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    void shift_arr_1bit_right_inside_single_word_robuster(uint64_t *a, size_t begin, size_t end, size_t a_size) {
        // end -= (end * 16 == a_size);
        if (begin >= end) return;
        // assert(end % 16);
        constexpr unsigned slot_size = sizeof(a[0]) * CHAR_BIT;
        constexpr unsigned shift = 1u;
        constexpr unsigned slot_mask = 63;
        constexpr unsigned items_in_slots = slot_size / shift;

        assert(end % items_in_slots);
        size_t index = begin / items_in_slots;
        assert(index == (end / items_in_slots));


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
    void shift_arr_1bit_right_inside_single_word_robuster8(uint8_t *a, size_t begin, size_t end) {

        if (begin >= end) return;
        constexpr unsigned slot_size = sizeof(a[0]) * CHAR_BIT;
        constexpr unsigned shift = 1u;
        constexpr unsigned slot_mask = slot_size - 1;
        constexpr unsigned items_in_slots = slot_size / shift;

        //        assert(end % items_in_slots);
        size_t index = begin / items_in_slots;
        assert(index == (end / items_in_slots));


        size_t rel_begin = begin & slot_mask;
        size_t rel_end = end & slot_mask;
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
    }

    /**
    * @brief We assume ((end % 16) != 0).
    *
    * @param a
    * @param begin
    * @param end
    * @param a_size
    */
    void shift_arr_1bit_right_att_helper(uint64_t *a, size_t begin, size_t end, size_t a_size) {
        if (begin == end) return;
        constexpr unsigned slot_size = sizeof(a[0]) * CHAR_BIT;
        constexpr unsigned shift = 1u;
        constexpr unsigned slot_mask = 63;
        constexpr unsigned items_in_slots = slot_size / shift;
        assert(items_in_slots * shift == slot_size);

        size_t begin_word_index = begin / slot_size;
        // size_t rel_begin = begin & (items_in_slots - 1u);
        size_t rel_end = end & (items_in_slots - 1u);
        assert(rel_end);
        size_t end_m1_word_index = end / slot_size;


        if (begin_word_index == end_m1_word_index) {
            // std::cout << "rr0" << std::endl;
            return shift_arr_1bit_right_inside_single_word_robuster(a, begin, end, a_size);
        }


        size_t i = end_m1_word_index;
        size_t lim = begin_word_index;
        uint64_t last_word_backup = a[end_m1_word_index];

        for (; i > lim; i--) {
            auto hi = (a[i] << shift);
            auto lo = a[i - 1] >> (slot_size - shift);
            assert(i >= 1);
            assert((lo & hi) == 0);
            a[i] = lo | hi;
        }

        shift_arr_1bit_right_inside_single_word_robuster(a, begin, begin | slot_mask, a_size);

        if (rel_end + 1 != items_in_slots) {
            // std::cout << "rr1" << std::endl;
            uint64_t last_mask = _bzhi_u64(-1, (rel_end + 1) * shift);
            uint64_t last_lo = a[end_m1_word_index] & last_mask;
            uint64_t last_hi = last_word_backup & ~last_mask;
            assert(!(last_lo & last_hi));
            a[end_m1_word_index] = last_lo | last_hi;
        }
    }

    void shift_arr_1bit_right_att_wrapper(uint64_t *a, size_t begin, size_t end, size_t a_size) {
        assert(end <= a_size * 64);
        assert(begin <= end);
        if (begin == end) return;
        if (a_size == 1) {
            return shift_arr_1bit_right_inside_single_word_robuster(a, begin, end, a_size);
        }
        constexpr unsigned slot_size = sizeof(a[0]) * CHAR_BIT;
        constexpr unsigned shift = 1u;
        // constexpr unsigned slot_mask = 63;
        constexpr unsigned items_in_slots = slot_size / shift;
        assert(items_in_slots * shift == slot_size);

        // size_t rel_begin = begin & (items_in_slots - 1);
        size_t rel_end = end & (items_in_slots - 1);
        if (rel_end) {
            // std::cout << "h0" << std::endl;
            return shift_arr_1bit_right_att_helper(a, begin, end, a_size);
        }
        size_t end_m1_word_index = (end - 1) / items_in_slots;

        if (end_m1_word_index + 1 == a_size) {
            // std::cout << "h1" << std::endl;
            assert(end / items_in_slots == a_size);
            return shift_arr_1bit_right_att_helper(a, begin, end - 1, a_size);
        }
        assert(end / items_in_slots != a_size);
        // std::cout << "h2" << std::endl;
        uint64_t last_bits = a[end_m1_word_index];
        shift_arr_1bit_right_att_helper(a, begin, end - 1, a_size);
        uint64_t lo = last_bits >> (slot_size - shift);
        uint64_t hi = a[end_m1_word_index + 1] & ~(15ULL);
        assert(!(lo & hi));
        a[end_m1_word_index + 1] = lo | hi;
    }

    void shift_arr_1bit_right_att_wrapper8(uint8_t *a, size_t begin, size_t end, size_t a8_size) {
        assert(end <= a8_size * 8);
        assert(begin <= end);
        if (begin == end)
            return;

        size_t s64 = (end - begin + 63) / 64 + 2;
        assert(s64 < 64);
        u64 temper[s64];
        std::fill(temper, temper + s64, 0);
        size_t bytes_to_copy = ((end - 1) / 8) - (begin / 8) + 1;
        const size_t first_byte = (begin / 8);
        memcpy(temper, a + first_byte, bytes_to_copy);
        size_t new_start = begin & 7;
        size_t new_end = end - (first_byte * 8);
        shift_arr_1bit_right_att_wrapper(temper, new_start, new_end, s64);
        memcpy(a + first_byte, temper, bytes_to_copy);
        return;
    }


    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    void shift_arr_k_bits_left_att_wrapper(uint64_t *a, size_t begin, size_t end, size_t a_size, size_t k) {
        assert(end * k <= a_size * 64);
        assert(begin <= end);
        if (begin == end) return;
        constexpr unsigned slot_size = sizeof(a[0]) * CHAR_BIT;
        const unsigned shift = k;
        const unsigned slot_sh_capacity = slot_size / shift;
        assert(slot_sh_capacity * shift == slot_size);

        size_t rel_begin = begin & (slot_sh_capacity - 1u);
        if (rel_begin) {
            // std::cout << "h0" << std::endl;
            return shift_arr_4bits_left_att_helper(a, begin, end, a_size);
        }

        if (begin == 0) {
            // std::cout << "h1" << std::endl;
            return shift_arr_4bits_left_att_helper(a, begin + 1, end, a_size);
        }

        // std::cout << "h2" << std::endl;
        size_t begin_word_index = begin / slot_sh_capacity;
        uint64_t last_bits = a[begin_word_index];
        shift_arr_4bits_left_att_helper(a, begin + 1, end, a_size);
        uint64_t lo = a[begin_word_index - 1] & _bzhi_u64(-1, slot_size - shift);
        uint64_t hi = last_bits << (slot_size - shift);
        assert(!(lo & hi));
        a[begin_word_index - 1] = lo | hi;
    }

    void shift_arr_k_bits_right_att_wrapper(u8 *a, size_t begin, size_t end, size_t a8_size, size_t k) {
        // size_t temp_k = k;
        for (size_t i = 0; i < k; ++i) {
            shift_arr_1bit_right_att_wrapper8(a, begin + i, end + i, a8_size);
        }
    }

    /**
    * @brief We assume ((end % 16) != 0).
    *
    * @param a
    * @param begin
    * @param end
    * @param a_size
    */
    void shift_arr_4bits_right_att_helper(uint64_t *a, size_t begin, size_t end, size_t a_size) {
        // assert(begin < end); //optimization.
        if (begin == end) return;
        constexpr unsigned slot_size = sizeof(a[0]) * CHAR_BIT;
        constexpr unsigned shift = 4u;
        constexpr unsigned slot_sh_capacity = slot_size / shift;
        assert(slot_sh_capacity * shift == slot_size);

        size_t begin_word_index = begin >> shift;
        //        size_t rel_begin = begin & (slot_sh_capacity - 1u);
        size_t rel_end = end & (slot_sh_capacity - 1u);
        assert(rel_end);
        size_t end_m1_word_index = end >> shift;


        if (begin_word_index == end_m1_word_index) {
            // std::cout << "rr0" << std::endl;
            return shift_arr_4bits_right_inside_single_word_robuster(a, begin, end);
        }

        size_t i = end_m1_word_index;
        size_t lim = begin_word_index;
        uint64_t last_word_backup = a[end_m1_word_index];

        for (; i > lim; i--) {
            auto hi = (a[i] << shift);
            auto lo = a[i - 1] >> (slot_size - shift);
            assert(i >= 1);
            assert((lo & hi) == 0);
            a[i] = lo | hi;
        }

        shift_arr_4bits_right_inside_single_word_robuster(a, begin, begin | 15);

        if (rel_end + 1 != slot_sh_capacity) {
            // std::cout << "rr1" << std::endl;
            uint64_t last_mask = _bzhi_u64(-1, (rel_end + 1) * shift);
            uint64_t last_lo = a[end_m1_word_index] & last_mask;
            uint64_t last_hi = last_word_backup & ~last_mask;
            assert(!(last_lo & last_hi));
            a[end_m1_word_index] = last_lo | last_hi;
        }
    }

    void shift_arr_4bits_right_att_wrapper(uint64_t *a, size_t begin, size_t end, size_t a_size) {
        // ZoneScopedN("Shift-4Right");
        assert(end * 4 <= a_size * 64);
        assert(begin <= end);
        assert(begin < end);//optimization.
        // if (begin == end) {
        //     return;
        // }
        constexpr unsigned slot_size = sizeof(a[0]) * CHAR_BIT;
        constexpr unsigned shift = 4u;
        constexpr unsigned slot_sh_capacity = slot_size / shift;
        assert(slot_sh_capacity * shift == slot_size);

        // size_t rel_begin = begin & (slot_sh_capacity - 1);
        size_t rel_end = end & (slot_sh_capacity - 1);
        if (rel_end) {
            // std::cout << "h0" << std::endl;
            return shift_arr_4bits_right_att_helper(a, begin, end, a_size);
        }
        size_t end_m1_word_index = (end - 1) / slot_sh_capacity;

        if (end_m1_word_index + 1 == a_size) {
            // std::cout << "h1" << std::endl;
            assert(end / slot_sh_capacity == a_size);
            return shift_arr_4bits_right_att_helper(a, begin, end - 1, a_size);
        }
        assert(end / slot_sh_capacity != a_size);
        // std::cout << "h2" << std::endl;
        uint64_t last_bits = a[end_m1_word_index];
        shift_arr_4bits_right_att_helper(a, begin, end - 1, a_size);
        uint64_t lo = last_bits >> (slot_size - shift);
        uint64_t hi = a[end_m1_word_index + 1] & ~(15ULL);
        assert(!(lo & hi));
        a[end_m1_word_index + 1] = lo | hi;
    }


    void shift_arr_4bits_right_att_wrapper8_un(uint8_t *a, size_t begin4, size_t end4, size_t a_size8) {
        assert(end4 <= a_size8 * 2);
        assert(begin4 <= end4);
        assert(begin4 < end4);

        constexpr unsigned slot_size = sizeof(a[0]) * CHAR_BIT;
        constexpr unsigned shift = 4u;
        constexpr unsigned slot_sh_capacity = slot_size / shift;
        static_assert(slot_sh_capacity == 2);
        assert(slot_sh_capacity * shift == slot_size);

        // size_t rel_begin = begin4 & (slot_sh_capacity - 1);
        // size_t rel_end = end4 & (slot_sh_capacity - 1);
        // size_t w_begin_index = begin4 / slot_sh_capacity;
        // size_t w_end_index = end4 / slot_sh_capacity;

        size_t counter = end4 - begin4;
        while (counter) {
            size_t write_index = end4 / slot_sh_capacity;
            size_t read_index = (end4 - 1) / slot_sh_capacity;
            // u8 temp = (end4 & 1) ? a[read_index] & 0xf : (a[read_index] >> 4u);
            if (end4 & 1) {
                u8 tag = a[read_index] & 0xf;
                a[write_index] = tag | (tag << 4u);
            } else {
                u8 tag = a[read_index] >> 4u;
                a[write_index] = (a[write_index] & 0xf0) | tag;
            }
            counter--;
            end4--;
        }
    }

    void shift_arr_4bits_left_att_wrapper8_sun(uint8_t *a, size_t begin4, size_t end4, size_t a_size8) {
        assert(end4 <= a_size8 * 2);
        assert(begin4 <= end4);
        assert(begin4 < end4);

        constexpr unsigned slot_size = sizeof(a[0]) * CHAR_BIT;
        constexpr unsigned shift = 4u;
        constexpr unsigned slot_sh_capacity = slot_size / shift;
        static_assert(slot_sh_capacity == 2);
        assert(slot_sh_capacity * shift == slot_size);

        begin4 += !begin4;
        // size_t rel_begin = begin4 & (slot_sh_capacity - 1);
        // size_t rel_end = end4 & (slot_sh_capacity - 1);
        // size_t w_begin_index = begin4 / slot_sh_capacity;
        // size_t w_end_index = end4 / slot_sh_capacity;


        size_t counter = end4 - begin4;
        while (counter) {
            size_t write_index = (begin4 - 1) / slot_sh_capacity;
            size_t read_index = begin4 / slot_sh_capacity;
            // u8 temp = (end4 & 1) ? a[read_index] & 0xf : (a[read_index] >> 4u);
            if (begin4 & 1) {
                u8 tag = a[read_index] >> 4u;
                // a[write_index] = tag | (tag << 4u);
                a[write_index] = (a[write_index] & 0xf0) | tag;
            } else {
                u8 tag = a[read_index] << 4u;
                a[write_index] = (a[write_index] & 0xf) | tag;
            }
            counter--;
            begin4++;
        }
    }
    void shift_arr_4bits_left_att_wrapper8_un(uint8_t *a, size_t begin4, size_t end4, size_t a_size8) {
        assert(end4 <= a_size8 * 2);
        assert(begin4 <= end4);
        assert(begin4 < end4);
        if (begin4 == end4) return;

        constexpr unsigned slot_size = sizeof(a[0]) * CHAR_BIT;
        constexpr unsigned shift = 4u;
        constexpr unsigned slot_sh_capacity = slot_size / shift;
        static_assert(slot_sh_capacity == 2);
        assert(slot_sh_capacity * shift == slot_size);

        // size_t rel_begin = begin4 & (slot_sh_capacity - 1);
        // size_t rel_end = end4 & (slot_sh_capacity - 1);
        // size_t w_begin_index = begin4 / slot_sh_capacity;
        // size_t w_end_index = end4 / slot_sh_capacity;

        size_t counter = end4 - begin4;
        size_t byte_index = begin4 / 2;
        if (begin4 & 1) {
            u8 temp = a[byte_index];
            u8 new_tag = (temp & 0xf0) | (temp >> 4u);
            a[byte_index] = new_tag;

            // a[byte_index] = ((a[byte_index]) & 0xf) | (a[byte_index + 1] << 4u);
            byte_index++;
            counter--;
        }
        size_t lim = counter / 2;
        for (size_t i = 0; i < lim; ++i) {
            u8 prev_data = a[byte_index + i];
            u8 new_data = (prev_data >> 4u) | (a[byte_index + i + 1] << 4u);
            a[byte_index] = new_data;
        }
        if (counter & 1) {
            assert(end4 & 1);
            u8 temp = a[byte_index + lim];
            u8 new_tag = (temp & 0xf0) | (temp >> 4u);
            a[byte_index + lim] = new_tag;
        }
    }

    void shift_4bits_right(uint8_t *array, uint16_t size) {
        // https://stackoverflow.com/a/29514/5381404
        // I call left shift a right shift.
        int i;
        uint8_t shifted = 0x00;
        uint8_t overflow = (0xF0 & array[0]) >> 4;

        for (i = (size - 1); i >= 0; i--) {
            shifted = (array[i] << 4) | overflow;
            overflow = (0xF0 & array[i]) >> 4;
            array[i] = shifted;
        }
    }


    void shift_arr_4bits_right_att_wrapper8_un2(uint8_t *a, size_t begin4, size_t end4, size_t a_size8) {
        assert(end4 <= a_size8 * 2);
        assert(begin4 <= end4);
        assert(begin4 < end4);

        begin4 += !begin4;
        size_t fixed_begin = (begin4 & 1) ? begin4 + 1 : begin4;
        size_t fixed_end = (end4 & 1) ? end4 - 1 : end4;
        size_t fixed_size = (fixed_end - fixed_begin) / 2;
        shift_4bits_right(a + (fixed_begin / 2), fixed_size);

        assert(0);
    }

    bool half_byte_read(const uint64_t *a, size_t half_byte_index) {
        size_t byte_index = half_byte_index / 2;
        if (half_byte_index & 1u)
            return (((uint8_t const *) a)[byte_index]) >> 4u;
        else {
            return (((uint8_t const *) a)[byte_index]) & 0xf;
        }
    }

    /**
     * @brief reads @a length items, each of max_cap 4 bits, starting from the @a half_byte_index, and compares them to @a rem4;
     *
     * @param a
     * @param half_byte_index
     * @param length
     * @param rem4
     * @return true
     * @return false
     */
    bool half_byte_cmp(const uint64_t *a, size_t half_byte_index, size_t length, uint8_t rem4) {
        //FIXME: to validate.
        assert(rem4 <= 0xF);
        size_t byte_index = half_byte_index / 2;
        if (half_byte_index & 1u) {
            uint8_t first = (((uint8_t const *) a)[byte_index]) >> 4u;
            if (first == rem4) return true;
            byte_index++;
            length--;
        }
        uint8_t const *pointer = (uint8_t const *) a + byte_index;
        size_t lim = length / 2;
        for (size_t i = 0; i < lim; i++) {
            uint8_t hi = (pointer[i] >> 4u) & 0xf;
            uint8_t lo = pointer[i] & 0xf;
            if (rem4 == hi) return true;
            if (rem4 == lo) return true;
        }
        if (length & 1) {
            uint8_t last_val = (pointer[lim] & 0xf);
            return last_val == rem4;
        }

        return false;
    }

    int half_byte_cmp_get_index_for_db(const uint64_t *a, size_t half_byte_index, size_t length, uint8_t rem4) {
        size_t counter = 0;
        assert(rem4 <= 0xF);
        size_t byte_index = half_byte_index / 2;
        if (half_byte_index & 1u) {
            uint8_t first = (((uint8_t const *) a)[byte_index]) >> 4u;
            if (first == rem4) return 0;
            counter++;
            byte_index++;
            length--;
        }
        uint8_t const *pointer = (uint8_t const *) a + byte_index;
        size_t lim = length / 2;
        for (size_t i = 0; i < lim; i++) {
            uint8_t hi = (pointer[i] >> 4u) & 0xf;
            uint8_t lo = pointer[i] & 0xf;
            if (rem4 == lo) return counter;
            counter++;
            if (rem4 == hi) return counter;
            counter++;
        }
        if (length & 1) {
            uint8_t last_val = (pointer[lim] & 0xf);
            if (last_val == rem4) return counter;
        }

        return -1;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    void update_byte(uint8_t *pointer, uint8_t rem4, bool should_update_hi) {
        assert(rem4 < 16);
        uint8_t new_byte = pointer[0];
        if (should_update_hi) {
            uint8_t lo = new_byte & 0xf;
            uint8_t hi = (rem4 << 4u);
            pointer[0] = lo | hi;
        } else {
            uint8_t hi = new_byte & 0xf0;
            uint8_t lo = rem4;
            pointer[0] = lo | hi;
        }
    }


    uint8_t read_4bits(const uint64_t *a, size_t index4, size_t a_size) {
        assert(index4 < a_size * 16);
        if (a_size == 1) {
            uint8_t res = (a[0] << (4u * index4)) & 0xf;
            return res;
        }
        const size_t w_index = (index4 + 15) / 16;
        const size_t rel_index = index4 & 15u;
        uint8_t res = (a[w_index] << (4u * rel_index)) & 0xf;
        return res;
    }

    uint8_t read_4bits(const uint8_t *a, size_t index4, size_t a_size) {
        assert(index4 < a_size * 2);
        if (a_size == 1) {
            uint8_t res = (a[0] << (2 * index4)) & 0xf;
            return res;
        }
        const size_t w_index = (index4 + 1) / 2;
        const size_t rel_index = index4 & 1u;
        uint8_t res = (a[w_index] << (2u * rel_index)) & 0xf;
        return res;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    void unpack_array(uint8_t *unpack_array, const uint8_t *packed_array, size_t packed_size) {
        for (size_t i = 0; i < packed_size; ++i) {
            uint8_t lo = packed_array[i] & 0xf;
            uint8_t hi = (packed_array[i] & 0xf0) >> 4u;
            unpack_array[i * 2] = lo;
            unpack_array[i * 2 + 1] = hi;
        }
    }

    void pack_array(uint8_t *pack_array, const uint8_t *unpacked_array, size_t unpacked_size) {
        for (size_t i = 0; i < unpacked_size; i += 2) {
            size_t index = i / 2;
            uint8_t lo = unpacked_array[i];
            uint8_t hi = unpacked_array[i + 1];
            assert(lo < 16);
            assert(hi < 16);
            uint8_t val = (hi << 4u) | lo;
            pack_array[index] = val;
        }
    }

    void unpack_array8x2(uint8_t *unpacked_array, const uint8_t *pack_array, size_t packed_size) {
        for (size_t i = 0; i < packed_size; ++i) {
            size_t index = i * 4;
            for (size_t j = 0; j < 4; ++j) {
                unpacked_array[index + j] = ((pack_array[i] >> (2 * j)) & 3);
            }
        }
    }

    void pack_array8x2(uint8_t *pack_array, const uint8_t *unpacked_array, size_t unpacked_size) {
        for (size_t i = 0; i < unpacked_size; i += 4) {
            size_t index = i / 4;
            const u8 *mp = unpacked_array + i;
            u8 val = mp[0] | (mp[1] << 2) | (mp[2] << 4) | (mp[3] << 6);
            pack_array[index] = val;
        }
    }

    void memcpy_for_vec(uint8_t *pack_array, const std::vector<bool> *b_vec) {
        size_t i = 0;
        const size_t V1_size = b_vec->size();
        const size_t V8_size = (b_vec->size()) / 8;

        for (; i < V8_size; i++) {
            size_t index = i * 8;
            u8 temp_word = 0;
            for (int j = 7; j >= 0; j--) {
                temp_word <<= 1;
                temp_word |= b_vec->at(index + j);
            }
            pack_array[i] = temp_word;
        }

        if (!(V1_size & 7))
            return;

        size_t index = V8_size * 8;
        size_t offset = V1_size & 7;
        u8 temp_word = 0;
        for (int j = offset; j >= 0; j--) {
            temp_word <<= 1;
            temp_word |= b_vec->at(index + j);
        }
        pack_array[i] = temp_word;
    }

    void pack_array_gen_k(u8 *pack_array, const u32 *unpacked_array, size_t items, size_t k) {
        const size_t V8_size = (k * items + 7) / 8;
        std::vector<bool> b_vec(V8_size * 8, false);
        for (size_t i = 0; i < items; i++) {
            size_t index = i * k;
            u32 temp_word = unpacked_array[i];
            for (size_t j = 0; j < k; j++) {
                b_vec.at(index + j) = temp_word & 1;
                temp_word >>= 1;
            }
        }
        // auto mp = &b_vec[0];
        memcpy_for_vec(pack_array, &b_vec);
    }
    void pack_array_gen_k_with_offset(u8 *pack_array, const u32 *unpacked_array, size_t items, size_t k, size_t offset) {
        u8 backup = pack_array[0];
        pack_array_gen_k(pack_array, unpacked_array, items, k);
        std::cout << "Here!1551: " << std::endl;
        // shift_arr_k_bits_left_att_wrapper(pack_array, 0, items * k, k);
    }

    

    void unpack_array_gen_k_with_offset(u32 *unpack_array, const u8 *packed_array, size_t items, size_t k, size_t offset) {
        assert(k <= 32);

        size_t start = offset;
        for (size_t i = 0; i < items; i++) {
            u32 temp_item = bitsMani::extract_bits(packed_array, start, start + k);
            unpack_array[i] = temp_item;
            start += k;
        }
    }

    void unpack_array_gen_k(u32 *unpack_array, const u32 *packed_array, size_t items, size_t k) {
        const size_t V1_size_exact = k * items;
        const size_t V8_size_round = (k * items + 7) / 8;
        std::vector<bool> b_vec(V8_size_round * 8, false);
        for (size_t i = 0; i < V1_size_exact; ++i) {
            bool bit = packed_array[i / 32] & (1 << (i & 31));
            bool bit2 = _bextr_u64(packed_array[i / 32], i & 31, 1);
            assert(bit == bit2);
            b_vec.at(i) = bit;
        }
        /*for (size_t i = 0; i < V1_size_exact; ++i) {
            std::cout << b_vec.at(i);
        }
        std::cout << std::endl;*/
        /*for (size_t i = 0; i < items; i++) {
            size_t index = i * k;
            u32 temp_word = 0;
            for (int j = k - 1; j >= 0; j--) {
                temp_word <<= 1;
                temp_word |= b_vec.at(index + j);
            }
            unpack_array[i] = temp_word;
        }*/
        for (size_t i = 0; i < items; i++) {
            size_t index = i * k;
            u32 temp_word = 0;
            u64 b = 1ULL;
            for (size_t h = 0; h < k; h++) {
                bool temp_val = b_vec.at(index + h);
                if (temp_val) temp_word |= b;
                b <<= 1u;
            }
            unpack_array[i] = temp_word;
        }
    }

    bool test_pack_unpack(const uint8_t *pack_a, size_t pack_size) {
        size_t unpack_size = pack_size * 2;
        uint8_t unpack_of_a[unpack_size];
        init_array(unpack_of_a, unpack_size);
        unpack_array(unpack_of_a, pack_a, pack_size);
        uint8_t pack_res[pack_size];
        init_array(pack_res, pack_size);
        pack_array(pack_res, unpack_of_a, unpack_size);

        int cmp_res = memcmp(pack_a, pack_res, pack_size);
        return cmp_res == 0;
    }


    /*  u32 pack4x8_as_4x6(const u8 *unpackArray) {
        assert(*std::max_element(unpackArray, unpackArray + 4) <= 63);

        u32 r1 = unpackArray[0] | ((u32) unpackArray[1] << 6u);
        u32 r2 = unpackArray[2] | ((u32) unpackArray[3] << 6u);
        u32 res = r1 | (r2 << 12u);

        assert(res <= _bzhi_u64(-1, 6 * 4));
        return res;
    }
    u32 pack7x6_as_5x8(const u8 *unpackArray) {
        assert(*std::max_element(unpackArray, unpackArray + 4) <= 63);

        u64 r1 = unpackArray[0] | ((u32) unpackArray[1] << 6u);
        u64 r2 = unpackArray[2] | ((u32) unpackArray[3] << 6u);
        u64 r3 = unpackArray[2] | ((u32) unpackArray[3] << 6u);
        u64 r4 = unpackArray[2] | ((u32) unpackArray[3] << 6u);
        u64 r5 = unpackArray[2] | ((u32) unpackArray[3] << 6u);
        u32 res = r1 | (r2 << 12u);

        assert(res <= _bzhi_u64(-1, 6 * 4));
        return res;
    }
 */

    u32 unpack4x6_as_4x8(u32 packed) {
        //   assert(*std::max_element(a, a + 4) <= 63);

        u32 x1 = packed & 63;
        u32 x2 = ((packed >> 6) & 63) << 8;
        u32 x3 = ((packed >> 12) & 63) << 16;
        u32 x4 = ((packed >> 18) & 63) << 24;

        return x1 | x2 | x3 | x4;
    }

    u64 unpack8x6_as_8x8(u64 packed) {
        //   assert(*std::max_element(a, a + 4) <= 63);
        // constexpr uint64_t mask0 = 0x3f3f'3f3f;
        constexpr uint64_t mask = 0x3f3f'3f3f'3f3f'3f3f;
        u64 res = _pdep_u64(packed, mask);
        // u64 res0 = _pdep_u64(packed, mask);
        return res;
        // return _pext_u64(packed, mask);
    }

    u64 pack8x8_as_8x6(u64 unpacked) {
        constexpr uint64_t mask = 0x3f3f'3f3f'3f3f'3f3f;
        static_assert(__builtin_popcountll(mask) == 48);
        return _pdep_u64(unpacked, mask);
        // return _pext_u64(packed, mask);
    }

    u32 pack4x8_as_4x6(const u8 unpackArray[4]) {

        u32 r1 = (unpackArray[0] & 63) | (((u32) unpackArray[1] & 63) << 6u);
        u32 r2 = (unpackArray[2] & 63) | (((u32) unpackArray[3] & 63) << 6u);
        u32 res = r1 | (r2 << 12u);

        assert(res <= _bzhi_u64(-1, 6 * 4));
        return res;
    }

    void pack6x8(u8 *packedArray, const u8 *unpackArray, size_t packed_size) {
        //        const size_t items = packed_size + packed_size / 2;
        const size_t items = packed_size * 4 / 3;
        const size_t reps = (items + 3) / 4;
        size_t p_index = 0;
        size_t up_index = 0;
        for (size_t i = 0; i < reps; i++) {
            u32 packed_value = pack4x8_as_4x6(unpackArray + up_index);
            memcpy(packedArray + p_index, &packed_value, 3);

            p_index += 3;
            up_index += 4;
        }

        size_t items_left = packed_size & 3;
        u8 temp_arr[4] = {0};
        memcpy(temp_arr, unpackArray + up_index, items_left);
        u32 packed_value = pack4x8_as_4x6(temp_arr);
        memcpy(packedArray + p_index, &packed_value, (items_left * 6 + 7) / 8);
    }

    void unpack6x8(u8 *unpackArray, const u8 *packedArray, size_t packed_size) {
        //        const size_t items = packed_size + packed_size / 2;
        const size_t items = packed_size * 4 / 3;
        const size_t reps = (items + 3) / 4;
        size_t p_index = 0;
        size_t up_index = 0;
        for (size_t i = 0; i < reps; i++) {
            u32 packed_value = 0;
            memcpy(&packed_value, packedArray + p_index, 3);
            u32 unpacked_value = unpack4x6_as_4x8(packed_value);
            memcpy(unpackArray + up_index, &unpacked_value, 4);

            p_index += 3;
            up_index += 4;
        }
        size_t items_left = packed_size & 3;
        u32 packed_value = 0;
        memcpy(&packed_value, packedArray + p_index, (items_left * 6 + 7) / 8);
        u32 unpacked_value = unpack4x6_as_4x8(packed_value);
        memcpy(unpackArray + up_index, &unpacked_value, items_left);
    }


    bool memcmp_1bit(const uint64_t *a, const uint64_t *b, size_t size1) {
        return memcmp_1bit((const uint8_t *) a, (const uint8_t *) b, size1);
    }

    bool memcmp_1bit(const uint8_t *a, const uint8_t *b, size_t size1) {
        assert(size1 > 0);
        size_t size8 = (size1 + 7) / 8;
        if ((size1 & 7) == 0) {
            return memcmp(a, b, size8) == 0;
        }

        bool temp = (memcmp(a, b, size8 - 1u) == 0);
        if (!temp) return false;


        size_t rel_index = size1 & 7;
        uint8_t mask = (1u << rel_index) - 1u;

        uint8_t h1 = a[size8 - 1];
        uint8_t h2 = b[size8 - 1];

        return (h1 & mask) == (h2 & mask);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    
    void insert_push_4bit_ultra_naive(u8 *packedArray, size_t packedSize, size_t index4, u8 item) {
        return insert_push_4bit_by_shift(packedArray, packedSize, index4, item);
        assert(item < 16u);
        assert(index4 < packedSize * 2);
        assert(2 <= packedSize);

        u8 backup_a[packedSize];
        memcpy(backup_a, packedArray, packedSize);

        const size_t unpack_size = packedSize * 2;
        u8 unpack_arr[unpack_size];
        init_array(unpack_arr, unpack_size);
        unpack_array(unpack_arr, packedArray, packedSize);

        //        const u8 index_byte = packedArray[packedSize - 1];
        // u8 prev_last = index_byte >> 4u;

        u8 *mp = unpack_arr + index4;
        size_t btc = unpack_size - (index4 + 1);
        memmove(mp + 1, mp, btc);
        mp[0] = item;
        //        assert(unpack_arr[unpack_size - 1] == prev_last); // this was commented, because I tried to use a trick in swap-split-insert easy case.
        pack_array(packedArray, unpack_arr, unpack_size);
        assert(check::validate_insert_push_4bit(backup_a, packedArray, packedSize, index4, item));
    }

    void insert_push_4bit_by_shift(u8 *packedArray, size_t packedSize, size_t index4, u8 item) {
        //        static int A[4] = {0};
        //        A[0]++;
        assert(item < 16u);
        assert(index4 < packedSize * 2);
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
    }


    void insert_push_4bit_disjoint_pair(u8 *packedArray, size_t packedSize, size_t index4, u8 rem1, u8 rem2) {
        //    static int A[4] = {0};
        //    A[0]++;
        assert((rem1 < 16u) and (rem2 < 16));
        assert(index4 < packedSize * 2);
        assert(index4 != 0);//it might work, but that was not the intention, and this case should be treated differently, (more efficient).
        assert(2 <= packedSize);


#ifndef NDEBUG
        u8 backup_a[packedSize];
        memcpy(backup_a, packedArray, packedSize);
#endif
        size_t byte_index = index4 / 2;
        u8 *mp = packedArray + byte_index;
        size_t bytes_to_move = packedSize - (byte_index + 1);
        memmove(mp + 1, mp, bytes_to_move);
#ifndef NDEBUG
        u8 dont_change = mp[1];
#endif
        fix_byte2(packedArray + byte_index, index4 & 1, rem2);

        size_t size64 = (packedSize + 7) / 8;
        shift_arr_4bits_right_att_wrapper((u64 *) packedArray, 0, index4 + 1, size64);
        packedArray[0] = (packedArray[0] & 0xf0) | rem1;
        assert((mp[1] & 0xf0) == (dont_change & 0xf0));

        assert(check::validate_insert_push_4bit_disjoint_pair(backup_a, packedArray, packedSize, index4, rem1, rem2));
        // assert(test::validate_insert_push_4bit_disjoint_pair(backup_a, packedArray, packedSize, index4 + 1, rem1, rem2));
    }

    void insert_push_4bit_disjoint_pair_reversed_array(u8 *packedArray, size_t packedSize, size_t index4, u8 rem1, u8 rem2) {
        //        static int A[4] = {0};
        //        A[0]++;
        assert((rem1 < 16u) and (rem2 < 16));
        assert(index4 < packedSize * 2);
        assert(index4 != 0);//it might work, but that was not the intention, and this case should be treated differently, (more efficient).
        assert(2 <= packedSize);
        auto s0 = str_bitsMani::str_array_half_byte_aligned(packedArray, packedSize);
        const size_t size_m1 = packedSize - 1;
        const size_t byte_index = index4 / 2;
        const size_t rev_byte_index = size_m1 - byte_index;
#ifndef NDEBUG
        u8 backup_a[packedSize];
        memcpy(backup_a, packedArray, packedSize);
#endif

        size_t end4 = packedSize * 2 - 2;
        size_t begin4 = (packedSize - 1) * 2 - index4;
        shift_arr_4bits_right_att_wrapper8_un(packedArray, begin4, end4, packedSize);
        auto s1 = str_bitsMani::str_array_half_byte_aligned(packedArray, packedSize);
        fix_byte2(packedArray + rev_byte_index, (index4 & 1), rem2);
        auto s2 = str_bitsMani::str_array_half_byte_aligned(packedArray, packedSize);

        std::cout << std::string(80, 'x') << std::endl;
        std::cout << "s0: \n"
                  << s0 << std::endl;
        std::cout << "s1: \n"
                  << s1 << std::endl;
        std::cout << "s2: \n"
                  << s2 << std::endl;

        std::cout << "arguments:" << std::endl;
        std::cout << "begin4: \t" << begin4 << std::endl;
        std::cout << "end4:   \t" << end4 << std::endl;
        std::cout << "rem1:   \t" << (u16) rem1 << std::endl;
        std::cout << "rem2:   \t" << (u16) rem2 << std::endl;
        std::cout << "index4: \t" << index4 << std::endl;
        std::cout << "size:   \t" << packedSize << std::endl;
        std::cout << "" << std::endl;
        std::cout << std::string(80, 'x') << std::endl;

        //        packedArray[size_m1] = (packedArray[size_m1] & 0xf0) | rem1;
        assert((packedArray[size_m1] & 0xf0) == (rem1 << 4u));
    }

    void insert_push_4bit_disjoint_pair_reversed_array_by_push(u8 *packedArray, size_t packedSize, size_t index4, u8 rem1, u8 rem2) {
        assert((rem1 < 16u) and (rem1 < 16u));
        assert(index4 < packedSize * 2);
        assert(2 <= packedSize);
        assert(packedSize <= 1024);
        u8 backup_a[packedSize];
        memcpy(backup_a, packedArray, packedSize);

        const size_t unpack_size = packedSize * 2;
        // const size_t unpack_last = unpack_size - 1;
        u8 unpack_arr[unpack_size];
        init_array(unpack_arr, unpack_size);
        unpack_array(unpack_arr, packedArray, packedSize);

        u8 *mp = unpack_arr + (unpack_size - 2) - index4;
        size_t btc = index4;
        int res_m1 = (unpack_arr + unpack_size) - (mp + 1 + btc);
        assert(res_m1 == 1);
        memmove(mp + 1, mp, btc);
        mp[0] = rem2;
        unpack_arr[(unpack_size - 1)] = rem1;
        pack_array(packedArray, unpack_arr, unpack_size);
        // assert(test::validate_insert_push_4bit(backup_a, packedArray, packedSize, index4, item));
    }

    void insert_push_4bit_disjoint_pair_reversed_array_naive(u8 *packedArray, size_t packedSize, size_t index4, u8 rem1, u8 rem2) {
        //FIXME::this is a naive implementation only.
        assert((rem1 < 16u) and (rem2 < 16));
        assert(index4 < packedSize * 2);
        assert(index4 != 0);//it might work, but that was not the intention, and this case should be treated differently, (more
        assert(2 <= packedSize);
        const size_t rev_index4 = (packedSize - 1) * 2 - index4;
#ifndef NDEBUG
        u8 backup_a[packedSize];
        memcpy(backup_a, packedArray, packedSize);
#endif

        std::cout << "input:" << std::endl;
        std::cout << str_bitsMani::str_array_half_byte_aligned(packedArray, packedSize);

        reverse_4_bits_array_in_place(packedArray, packedSize);
        std::cout << "reversed:" << std::endl;
        std::cout << str_bitsMani::str_array_half_byte_aligned(packedArray, packedSize);

        std::cout << "arguments:" << std::endl;
        std::cout << "rem1:   \t" << (u16) rem1 << std::endl;
        std::cout << "rem2:   \t" << (u16) rem2 << std::endl;
        std::cout << "index4: \t" << index4 << std::endl;
        std::cout << "Rindex4:\t" << rev_index4 << std::endl;
        std::cout << "size:   \t" << packedSize << std::endl;
        std::cout << "" << std::endl;


        insert_push_4bit_disjoint_pair(packedArray, packedSize, rev_index4, rem1, rem2);

        std::cout << "after:" << std::endl;
        std::cout << str_bitsMani::str_array_half_byte_aligned(packedArray, packedSize);


        reverse_4_bits_array_in_place(packedArray, packedSize);
        std::cout << "re-reversed:" << std::endl;
        std::cout << str_bitsMani::str_array_half_byte_aligned(packedArray, packedSize);
        // assert(test::validate_insert_push_4bit_disjoint_pair(backup_a, packedArray, packedSize, index4, rem1, rem2));
        // assert(test::validate_insert_push_4bit_disjoint_pair(backup_a, packedArray, packedSize, index4 + 1, rem1, rem2));
    }


    void delete_pull_4bit_ultra_naive_small_size_array(u8 *a, size_t index4, u8 item) {
        assert(item < 16u);
        if (index4 == 0) {
            u8 masked_val = (a[0] >> 4u) << 4u;
            a[0] = (masked_val | item);
            return;
        }
        u8 masked_val = a[0] & 0xf;
        a[0] = (masked_val | (item << 4u));
        return;
    }

    void delete_pull_4bit_ultra_naive(u8 *packedArray, size_t packedSize, size_t index4, u8 item) {
        assert(item < 16u);
        assert(index4 < packedSize * 2);
        // if (packedSize == 1)
        //     return delete_pull_4bit_ultra_naive_small_size_array(packedArray, index4, item);

        if (index4 < 2) {
            return delete_pull_4bit_ultra_naive_small_size_array(packedArray, index4, item);
        }

        u8 backup_a[packedSize];
        memcpy(backup_a, packedArray, packedSize);

        const size_t unpack_size = packedSize * 2;
        u8 unpack_arr[unpack_size];
        init_array(unpack_arr, unpack_size);
        unpack_array(unpack_arr, packedArray, packedSize);

        // const u8 index_byte = packedArray[packedSize - 1];
        // u8 prev_last = index_byte >> 4u;

        u8 *mp = unpack_arr + index4;
        size_t btc = unpack_size - (index4 + 1);
        memmove(unpack_arr, unpack_arr + 1, btc);
        mp[0] = item;
        //        assert(unpack_arr[unpack_size - 1] == prev_last); // this was commented, because I tried to use a trick in swap-split-insert easy case.
        pack_array(packedArray, unpack_arr, unpack_size);
        assert(check::validate_insert_push_4bit(backup_a, packedArray, packedSize, index4, item));
    }

    void insert_push_4bit_ultra_naive_by_unpackSize(u8 *packedArray, size_t packedSize, size_t index4, u8 item,
                                                    size_t unpackSize) {
        assert(item < 16u);
        assert(index4 < packedSize * 2);
        assert(packedSize < 2);
        u8 backup_a[packedSize];
        memcpy(backup_a, packedArray, packedSize);

        const size_t unpack_size = unpackSize;
        u8 unpack_arr[unpack_size];
        init_array(unpack_arr, unpack_size);
        unpack_array(unpack_arr, packedArray, packedSize);

        //        const u8 index_byte = packedArray[packedSize - 1]; // Keep this ->
        //        u8 prev_last = index_byte >> 4u;
        u8 *mp = unpack_arr + index4;
        size_t btc = unpack_size - (index4 + 1);
        memmove(mp + 1, mp, btc);
        mp[0] = item;
        //        assert(unpack_arr[unpack_size - 1] == prev_last); // this was commented, because I tried to use a trick in swap-split-insert easy case.
        pack_array(packedArray, unpack_arr, unpack_size);
        assert(check::validate_insert_push_4bit(backup_a, packedArray, packedSize, index4, item));
    }


    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    __uint128_t get_4bits_cmp_vector(const uint64_t *a, size_t start4, size_t length, uint8_t rem4) {
        assert(length < UINT64_MAX);
        __uint128_t res = 0;
        __uint128_t b = 1;

        // size_t counter = 0;
        assert(rem4 <= 0xF);
        size_t byte_index = start4 / 2;
        if (start4 & 1u) {
            uint8_t first = (((uint8_t const *) a)[byte_index]) >> 4u;
            if (first == rem4) res |= b;
            // counter++;
            byte_index++;
            length--;
            b <<= 1;
        }
        uint8_t const *pointer = (uint8_t const *) a + byte_index;
        size_t lim = length / 2;
        for (size_t i = 0; i < lim; i++) {
            uint8_t hi = (pointer[i] >> 4u) & 0xf;
            uint8_t lo = pointer[i] & 0xf;

            if (rem4 == lo) res |= b;
            b <<= 1;

            if (rem4 == hi) res |= b;
            b <<= 1;
        }
        if (length & 1) {
            uint8_t last_val = (pointer[lim] & 0xf);
            if (last_val == rem4) res |= b;
        }

        return res;
    }


    u16 get_4bits_cmp_on_word(const u8 word[8], uint8_t rem4) {
        u16 res = 0;
        u64 b = 1ULL;
        u8 lo_rem = rem4;
        u8 hi_rem = rem4 << 4u;

        /* bool f;        // conditional flag
        unsigned int m;// the bit mask
        unsigned int w;// the word to modify:  if (f) w |= m; else w &= ~m;

        w ^= (-f ^ w) & m;*/

        for (size_t i = 0; i < 8; i++) {
            if ((word[i] & 0xf) == lo_rem) res |= b;
            b <<= 1;
            if ((word[i] & 0xf0) == hi_rem) res |= b;
            b <<= 1;
        }
        return res;
    }

    u16 get_4bits_cmp_on_word3(const u8 word[8], uint8_t rem4, size_t length) {
        u16 res = 0;
        u64 b = 1ULL;
        u8 lo_rem = rem4;
        u8 hi_rem = rem4 << 4u;

        /* bool f;        // conditional flag
        unsigned int m;// the bit mask
        unsigned int w;// the word to modify:  if (f) w |= m; else w &= ~m;

        w ^= (-f ^ w) & m;*/

        for (size_t i = 0; i < length; i++) {
            if ((word[i] & 0xf) == lo_rem) res |= b;
            b <<= 1;
            if ((word[i] & 0xf0) == hi_rem) res |= b;
            b <<= 1;
        }
        return res;
    }


    u16 get_4bits_cmp16(const uint64_t *a, size_t start4, uint8_t rem4) {
        u64 h1 = extract_word(a, start4);
        return get_4bits_cmp_on_word((const u8 *) (&h1), rem4);
    }


    u16 get_4bits_cmp16_ranged1(const uint64_t *a, size_t start4, uint8_t rem4, size_t length) {
        assert(length < 16);
        assert(length < 15);
        size_t byte_index = start4 / 2;
        auto mp = (const u8 *) a + byte_index;
        u64 h1;
        memcpy(&h1, mp, 8);
        if (start4 & 1) h1 >>= 4u;

        auto temp_arr = (const u8 *) (&h1);
        // u16 res_mask = get_4bits_cmp_on_word3(temp_arr, rem4, length / 2);
        assert(0);
        return -1;

    }

    u16 get_4bits_cmp16_ranged2(const uint64_t *a, size_t start4, uint8_t rem4, size_t length) {
        assert(length < 16);
        size_t byte_index = start4 / 2;
        // auto mp = (const u8 *) a + byte_index;
        u16 cmp_mask = get_4bits_cmp_on_word((const u8 *) a + byte_index, rem4);
        u8 shift = 4 * (start4 & 1);
        return (cmp_mask >> shift) & _bzhi_u64(-1, length);
    }

    u16 get_4bits_cmp16_ranged3(const uint64_t *a, size_t start4, uint8_t rem4, size_t length) {
        assert(length < 16);
        size_t byte_index = start4 / 2;
        // auto mp = (const u8 *) a + byte_index;
        u16 cmp_mask = get_4bits_cmp_on_word((const u8 *) a + byte_index, rem4);
        return cmp_mask >> (start4 & 1);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    void reverse_4_bits_array_naive(const u8 *a, u8 *rev_a, size_t packed_size) {
        const size_t unpack_size = packed_size * 2;
        assert(unpack_size < (1ULL << 16));// sanity check
        u8 *unpack_a = new u8[unpack_size];

        unpack_array(unpack_a, a, packed_size);
        std::reverse(unpack_a, unpack_a + unpack_size);

        pack_array(unpack_a, rev_a, unpack_size);
        delete[] unpack_a;
    }

    void reverse_4_bits_array(const u8 *a, u8 *rev_a, size_t packed_size) {
        assert(a != rev_a);
        const size_t last_i = packed_size - 1;
        for (size_t i = 0; i < packed_size; i++) {
            rev_a[i] = flip2x4(a[last_i - i]);
        }
    }

    void reverse_4_bits_array_in_place(u8 *a, size_t packed_size) {
        if (packed_size == 0) {
            std::cout << "rev4 in place with size zero." << std::endl;
            return;
        }
        if (packed_size == 1) {
            a[0] = flip2x4(a[0]);
            return;
        }

        const size_t last_i = packed_size - 1;
        size_t i = 0;
        for (; i + i < last_i; i++) {
            u8 x = a[i];
            u8 y = a[last_i - i];
            a[i] = flip2x4(y);
            a[last_i - i] = flip2x4(x);
        }
        if (i + i == last_i)
            a[i] = flip2x4(a[i]);
    }

}// namespace Shift_op


namespace Shift_op::check {

    bool test_rev4_bits_arr(const u8 *a, size_t packed_size) {
        u8 *rev_a1 = new u8[packed_size];
        u8 *rev_a2 = new u8[packed_size];

        reverse_4_bits_array(a, rev_a1, packed_size);
        reverse_4_bits_array(rev_a1, rev_a2, packed_size);

        int res = memcmp(rev_a2, a, packed_size);
        if (res == 0) {
            delete[] rev_a1;
            delete[] rev_a2;
            return true;
        }

        auto s0 = str_bitsMani::str_array_half_byte_aligned(a, packed_size);
        auto s1 = str_bitsMani::str_array_half_byte_aligned(rev_a1, packed_size);
        auto s2 = str_bitsMani::str_array_half_byte_aligned(rev_a2, packed_size);
        delete[] rev_a1;
        delete[] rev_a2;
        assert(0);
        return false;
    }

    bool test_rev4_bits_in_place(const u8 *a, size_t packed_size) {
        u8 *rev_a1 = new u8[packed_size];
        u8 *rev_a2 = new u8[packed_size];
        u8 *rev_a3 = new u8[packed_size];


        memcpy(rev_a3, a, packed_size);
        reverse_4_bits_array_in_place(rev_a3, packed_size);

        auto s_a = str_bitsMani::str_array_half_byte_aligned(a, packed_size);
        auto s3 = str_bitsMani::str_array_half_byte_aligned(rev_a3, packed_size);

        reverse_4_bits_array(a, rev_a1, packed_size);
        bool cmp1 = (memcmp(rev_a3, rev_a1, packed_size) == 0);
        if (!cmp1) {
            auto s1 = str_bitsMani::str_array_half_byte_aligned(rev_a1, packed_size);
            std::cout << "sa: \n"
                      << s_a << std::endl;

            std::cout << "s1: \n"
                      << s1 << std::endl;

            std::cout << "s3: \n"
                      << s3 << std::endl;
        }
        assert(cmp1);
        reverse_4_bits_array(rev_a1, rev_a2, packed_size);

        reverse_4_bits_array_in_place(rev_a3, packed_size);
        bool cmp2 = (memcmp(rev_a3, rev_a2, packed_size) == 0);
        assert(cmp2);

        bool cmp3 = (memcmp(rev_a3, a, packed_size) == 0);
        assert(cmp3);

        delete[] rev_a1;
        delete[] rev_a2;
        delete[] rev_a3;
        return true;
    }

    bool test_shift4_right_r() {
        constexpr size_t a_size = 64;
        u64 a[a_size] = {0};
        for (size_t i = 0; i < a_size; i++) {
            a[i] = random() ^ random();
        }

        u64 val[a_size] = {0};
        u8 att[a_size * 8] = {0};

        constexpr size_t reps = 1ULL << 12u;
        size_t range = a_size / 4;
        for (size_t i = 0; i < reps; i++) {
            memcpy(val, a, a_size * 8);
            memcpy(att, a, a_size * 8);
            assert(memcmp(att, val, a_size * 8) == 0);
            size_t begin4 = 0;
            size_t end4 = 0;
            while (end4 <= begin4) {
                begin4 = random() % range;
                end4 = random() % range;
            }


            shift_arr_4bits_right_att_wrapper(val, begin4, end4, a_size);
            shift_arr_4bits_right_att_wrapper8_un(att, begin4, end4, a_size * 8);
            assert(memcmp(att, val, a_size * 8) == 0);
        }
        return true;
    }

    bool test_shift4_left_r() {
        constexpr size_t a_size = 64;
        u64 a[a_size] = {0};
        for (size_t i = 0; i < a_size; i++) {
            a[i] = random() ^ random();
        }

        u64 val[a_size] = {0};
        u8 att[a_size * 8] = {0};

        constexpr size_t reps = 1ULL << 12u;
        size_t range = a_size;
        for (size_t i = 0; i < reps; i++) {
            memcpy(val, a, a_size * 8);
            memcpy(att, a, a_size * 8);
            assert(memcmp(att, val, a_size * 8) == 0);
            size_t begin4 = 0;
            size_t end4 = 0;
            // while ((end4 <= begin4) or (begin4 == 0)) {
            while (end4 <= begin4) {
                begin4 = random() % range;
                end4 = random() % range;
            }


            shift_arr_4bits_left_att_wrapper(val, begin4, end4, a_size);
            shift_arr_4bits_left_att_wrapper8_sun(att, begin4, end4, a_size * 8);
            bool ok = memcmp(att, val, a_size * 8) == 0;

            if (ok)
                continue;


            size_t offset = begin4 / 2;
            if ((offset & 1) and (offset > 1)) offset--;
            size_t interval = (end4 - begin4 + 1) / 2;
            auto s0 = str_bitsMani::str_array_half_byte_aligned((const u8 *) a + offset, interval);
            auto s1 = str_bitsMani::str_array_half_byte_aligned(att + offset, interval);
            auto s2 = str_bitsMani::str_array_half_byte_aligned((const u8 *) val + offset, interval);
            auto s0_full = str_bitsMani::str_array_half_byte_aligned((const u8 *) a, a_size);
            auto s1_full = str_bitsMani::str_array_half_byte_aligned(att, a_size);
            auto s2_full = str_bitsMani::str_array_half_byte_aligned((const u8 *) val, a_size);

            std::cout << std::string(80, '=') << std::endl;
            std::cout << "i:  \t" << i << std::endl;
            std::cout << "begin4:  \t" << begin4 << std::endl;
            std::cout << "end4:    \t" << end4 << std::endl;
            std::cout << "offset:  \t" << offset << std::endl;
            std::cout << "interval:\t" << interval << std::endl;
            std::cout << std::string(80, '~') << std::endl;
            std::cout << s0 << std::endl;
            std::cout << s1 << std::endl;
            std::cout << s2 << std::endl;
            std::cout << std::string(80, '~') << std::endl;
            std::cout << s0_full << std::endl;
            std::cout << s1_full << std::endl;
            std::cout << s2_full << std::endl;
            std::cout << std::string(80, '=') << std::endl;

            // std::cout << std::string(80, '=') << std::endl;
            // std::cout << std::string(80, '=') << std::endl;

            // auto s3 = str_bitsMani::str_array_half_byte_aligned((const u8 *) a, a_size * 8);
            // auto s4 = str_bitsMani::str_array_half_byte_aligned(att, a_size * 8);
            // auto s5 = str_bitsMani::str_array_half_byte_aligned((const u8 *) val, a_size * 8);


            assert(memcmp(att, val, a_size * 8) == 0);
        }
        return true;
    }

    /* void insert_push_4bit_disjoint_pair_reversed_array(u8 *pre_a, u8 *post_a, size_t packed_size, size_t index4, u8 first_in_queue, u8 other_rem) {
        size_t unpack_size = packed_size * 2;

        u8 pre_unpack[unpack_size];
        init_array(pre_unpack, unpack_size);
        unpack_array(pre_unpack, pre_a, packed_size);

        u8 post_unpack[unpack_size];
        init_array(post_unpack, unpack_size);
        unpack_array(post_unpack, post_a, packed_size);

        post_unpack[0] ==

                size_t ri = packed_size - 1;
        for (size_t i = 0; i < index4; i++) {
        }


        for (size_t i = 0; i < index4; i++) {
            if (pre_unpack[i] != post_unpack[i]) {
                return false;
            }
        }

        if (post_unpack[index4] != item)
            return false;

        for (size_t i = index4; i < unpack_size - 1; i++) {
            if (pre_unpack[i] != post_unpack[i + 1]) {
                return false;
            }
        }
        return true;
    }
 */
    /**
     * Validates the functionality of an insert_push function, based on the original input, and the result.
     * @param pre_a The original input
     * @param post_a The output
     * @param packed_size number of 4 items in pre_a.
     * @param index4 the index in which we wanted to insert new item.
     * @param item the new item value. (4 bits).
     * @return
     */
    bool validate_insert_push_4bit(u8 *pre_a, u8 *post_a, size_t packed_size, size_t index4, u8 item) {
        size_t unpack_size = packed_size * 2;

        u8 pre_unpack[unpack_size];
        init_array(pre_unpack, unpack_size);
        unpack_array(pre_unpack, pre_a, packed_size);

        u8 post_unpack[unpack_size];
        init_array(post_unpack, unpack_size);
        unpack_array(post_unpack, post_a, packed_size);

        for (size_t i = 0; i < index4; i++) {
            if (pre_unpack[i] != post_unpack[i]) {
                return false;
            }
        }

        if (post_unpack[index4] != item)
            return false;

        for (size_t i = index4; i < unpack_size - 1; i++) {
            if (pre_unpack[i] != post_unpack[i + 1]) {
                return false;
            }
        }
        return true;
    }

    bool validate_insert_push_4bit_disjoint_pair(u8 *pre_a, u8 *post_a, size_t packed_size, size_t index4, u8 lo_rem1, u8 lo_rem2) {
        size_t unpack_size = packed_size * 2;

        u8 pre_unpack[unpack_size];
        init_array(pre_unpack, unpack_size);
        unpack_array(pre_unpack, pre_a, packed_size);

        u8 post_unpack[unpack_size];
        init_array(post_unpack, unpack_size);
        unpack_array(post_unpack, post_a, packed_size);

        assert(post_unpack[0] == lo_rem1);
        for (size_t i = 0; i < index4; i++) {
            if (pre_unpack[i] != post_unpack[i + 1]) {
                std::cout << "index4:      \t" << index4 << std::endl;
                std::cout << "unpack_size: \t" << unpack_size << std::endl;
                auto s0 = str_bitsMani::str_array_with_line_numbers(pre_unpack, index4 + 3);
                auto s1 = str_bitsMani::str_array_with_line_numbers(post_unpack, index4 + 3);
                std::cout << std::endl;
                std::cout << s0 << std::endl;
                std::cout << std::string(80, '=') << std::endl;
                std::cout << s1 << std::endl;
                return false;
            }
        }

        if (post_unpack[index4 + 1] != lo_rem2) {
            std::cout << "index4:      \t" << index4 << std::endl;
            std::cout << "unpack_size: \t" << unpack_size << std::endl;
            auto s0 = str_bitsMani::str_array_with_line_numbers(pre_unpack, index4 + 3);
            auto s1 = str_bitsMani::str_array_with_line_numbers(post_unpack, index4 + 3);
            std::cout << std::endl;
            std::cout << s0 << std::endl;
            std::cout << std::string(80, '=') << std::endl;
            std::cout << s1 << std::endl;
            return false;
        }

        for (size_t i = index4; i < unpack_size - 2; i++) {
            if (pre_unpack[i] != post_unpack[i + 2]) {
                std::cout << "index4:      \t" << index4 << std::endl;
                std::cout << "unpack_size: \t" << unpack_size << std::endl;
                auto s0 = str_bitsMani::str_array_with_line_numbers(pre_unpack, unpack_size);
                auto s1 = str_bitsMani::str_array_with_line_numbers(post_unpack, unpack_size);
                std::cout << std::endl;
                std::cout << s0 << std::endl;
                std::cout << std::string(80, '=') << std::endl;
                std::cout << s1 << std::endl;
                return false;
            }
        }
        return true;
    }


    bool test_pack_unpack_6x8(const uint8_t *pack_a, size_t pack_size) {
        auto s0 = str_bitsMani::str_array_as_memory(pack_a, pack_size);
        size_t items = pack_size + pack_size / 2;
        size_t unpack_size = items;
        uint8_t unpack_of_a[unpack_size];
        init_array(unpack_of_a, unpack_size);
        unpack6x8(unpack_of_a, pack_a, pack_size);
        auto us = str_bitsMani::str_array_as_memory(unpack_of_a, unpack_size);

        uint8_t pack_res[pack_size];
        init_array(pack_res, pack_size);
        pack6x8(pack_res, unpack_of_a, pack_size);

        bool res = memcmp(pack_a, pack_res, pack_size) == 0;
        if (res) return true;

        size_t index = 0;
        while (pack_a[index] == pack_res[index]) { index++; }


        std::cout << s0 << std::endl;
        std::cout << us << std::endl;
        auto s1 = str_bitsMani::str_array_as_memory(pack_res, pack_size);
        // std::cout << s0 << std::endl;
        std::cout << s1 << std::endl;

        std::cout << "first wrong index:" << index << std::endl;
        std::cout << "pack_size: " << pack_size << std::endl;

        assert(0);
        return false;
    }

    bool test_pack_unpack_pdep0() {
        constexpr u64 bz_mask = (1ull << 48u) - 1u;
        // constexpr size_t p_size = 8;
        u64 packed = 0x207185'103081;// {1,2,3,4,5,6,7,8} (each 6 bits)
        u64 true_unpacked = 0x0807'0605'0403'0201;

        u64 lo_up = unpack4x6_as_4x8(packed);
        u64 hi_up = unpack4x6_as_4x8(packed >> 24);
        u64 hi_up2 = unpack4x6_as_4x8(0x207185);
        if (hi_up != hi_up2) {
            std::cout << "hi_hp differ. " << std::endl;
            auto s_diff = str_bitsMani::format_2words_and_xor(hi_up, hi_up2);
            std::cout << s_diff << std::endl;
            std::cout << std::string(80, '=') << std::endl;
        }
        u64 val_up = lo_up | (hi_up << 32u);

        bool cmp0 = ((val_up & bz_mask) == (true_unpacked & bz_mask));
        if (!cmp0) {
            std::cout << str_bitsMani::format_word_to_string(val_up) << std::endl;
            std::cout << str_bitsMani::format_word_to_string(true_unpacked) << std::endl;
            std::cout << str_bitsMani::format_word_to_string(val_up ^ true_unpacked) << std::endl;
            assert(0);
        }
        return true;
    }


    bool test_pack_unpack_pdep() {
        constexpr size_t p_size = 8;
        u8 packed[p_size] = {0};

        for (size_t i = 0; i < p_size; i++) {
            packed[i] = random() ^ random() ^ random() ^ random() ^ 0x20739156;
        }

        // constexpr size_t temp_size = 6;

        // u8 unpack_temp[p_size] = {0};
        u32 lo_p, hi_p;
        memcpy(&lo_p, packed, 3);
        memcpy(&hi_p, packed + 3, 3);
        u64 packed64 = lo_p | (((u64) hi_p) << 24);
        u64 lo_up = unpack4x6_as_4x8(lo_p);
        u64 hi_up = unpack4x6_as_4x8(hi_p);
        u64 val_up = lo_up | (hi_up << 32u);

        u64 att_up = unpack8x6_as_8x8(packed64);
        // constexpr u64 bz_mask = (1ull << 48u) - 1u;
        constexpr u64 bz_mask = -1;
        bool res = ((att_up & bz_mask) == (val_up & bz_mask));

        if (res) return 1;

        constexpr uint64_t mask = 0x3f3f'3f3f'3f3f'3f3f;

        auto s0 = str_bitsMani::str_array_as_memory(packed, 8);
        auto s1 = str_bitsMani::format_word_to_string(val_up);
        auto s2 = str_bitsMani::format_word_to_string(val_up & bz_mask);
        auto s3 = str_bitsMani::format_word_to_string(att_up);
        auto s4 = str_bitsMani::format_word_to_string(att_up & bz_mask);
        auto s5 = str_bitsMani::format_word_to_string(att_up ^ val_up);
        auto s6 = str_bitsMani::format_word_to_string((att_up ^ val_up) & bz_mask);


        std::cout << "mem:    \t" << s0;
        std::cout << "val:    \t" << s1 << std::endl;
        std::cout << "valm:   \t" << s2 << std::endl;
        std::cout << "att:    \t" << s3 << std::endl;
        std::cout << "attm:   \t" << s4 << std::endl;
        std::cout << "xor:    \t" << s5 << std::endl;
        std::cout << "xorm:   \t" << s6 << std::endl;
        std::cout << "mask:   \t" << str_bitsMani::format_word_to_string(mask) << std::endl;
        std::cout << "val_up: \t" << val_up << std::endl;
        std::cout << "att_up: \t" << att_up << std::endl;


        assert(0);
        return false;
    }

    bool test_pack_unpack_array_gen_k(const uint32_t *pack_a, size_t items, size_t k) {
        const size_t pack_size = (k * items + 7) / 8;
        auto s0 = str_bitsMani::str_array_as_memory((const u8 *) pack_a, pack_size);
        const size_t unpack_size = items;
        uint32_t unpack_of_a[unpack_size];
        init_array(unpack_of_a, unpack_size);
        unpack_array_gen_k(unpack_of_a, pack_a, items, k);
        auto us = str_bitsMani::get_first_k_bits_of_each_item(unpack_of_a, items, k);

        uint8_t pack_res[pack_size];
        init_array(pack_res, pack_size);
        pack_array_gen_k(pack_res, unpack_of_a, items, k);

        size_t dec = ((k * items & 7) != 0);
        size_t pack_size_m1 = pack_size - dec;
        bool res0 = memcmp(pack_a, pack_res, pack_size_m1) == 0;
        bool res = memcmp(pack_a, pack_res, pack_size) == 0;
        if (res) return true;

        if (res0) {
            size_t offset = (k * items) & 7;
            const size_t index = pack_size - 1;
            const u64 mask = _bzhi_u64(-1, offset);
            assert(index < pack_size);
            bool fix_res = (((const u8 *) pack_a)[index] & mask) == (pack_res[index] & mask);
            if (fix_res)
                return true;
        }
        size_t index = 0;
        while (((const u8 *) pack_a)[index] == pack_res[index]) { index++; }


        std::cout << s0 << std::endl;
        std::cout << us << std::endl;
        auto s1 = str_bitsMani::str_array_as_memory(pack_res, pack_size);
        // std::cout << s0 << std::endl;
        std::cout << s1 << std::endl;

        std::cout << "k:                 \t" << k << std::endl;
        std::cout << "items:             \t" << items << std::endl;
        std::cout << "first wrong index: \t" << index << std::endl;
        std::cout << "pack_size:         \t" << pack_size << std::endl;

        //        assert(0);
        return false;
    }

    void comp_test0_shift_arr_k_bits_right_att_wrapper() {
        constexpr size_t a_size = 2;
        constexpr size_t range = 56;
        u64 a[64];
        for (size_t i = 0; i < a_size; ++i) {
            a[0] = random() ^ random() ^ random() ^ random() ^ 0xf982a35b;
        }
        // for (size_t reps = 0; reps < 64; reps++) {
        //     for (size_t i = 0; i < a_size; i++) {
        u64 b[a_size];
        memcpy(b, a, a_size * 8);
        // u64 c[a_size];
        // memcpy(c, a, a_size * 8);
        size_t begin = 0;
        size_t end = 0;
        while ((end <= begin) or (end + 7 > a_size * 8)) {
            begin = random() % range;
            end = random() % range;
        }

        std::cout << "begin: \t" << begin << std::endl;
        std::cout << "end:   \t" << end << std::endl;
        std::cout << std::string(80, '=') << std::endl;
        std::cout << str_bitsMani::str_array_as_memory_no_delim((u8 *) b, 8);
        std::cout << std::endl
                  << std::string(80, '~') << std::endl;
        for (size_t j = 0; j < 7; j++) {
            shift_arr_k_bits_right_att_wrapper((u8 *) b, begin + j, end + j, a_size * 8, 1);
            // size_t min_end = std::min(end + j, a_size * 8);
            // shift_arr_k_bits_right_att_wrapper((u8 *) b, begin + j, end + j, a_size * 8, 2);
            // auto s = str_bitsMani::str_array_as_memory((u8 *) b, a_size);
            auto s = str_bitsMani::str_array_as_memory_no_delim((u8 *) b, 8);
            std::cout << s << std::endl;
            // std::cout << std::string(80, '-') << std::endl;
        }


        // }
        // }
    }
    bool comp_test_shift_arr_k_bits_right_att_wrapper() {
        constexpr size_t a_size = 64;
        constexpr size_t range = a_size * 64;
        u64 a[64];
        for (size_t i = 0; i < a_size; ++i) {
            a[0] = random() ^ random() ^ random();
        }
        for (size_t reps = 0; reps < 64; reps++) {
            for (size_t i = 0; i < a_size; i++) {
                u64 b[64];
                u64 c[64];
                memcpy(b, a, a_size * 8);
                memcpy(c, a, a_size * 8);
                size_t begin = 0;
                size_t end = 0;
                while (end <= begin) {
                    begin = random() % range;
                    end = random() % range;
                }
                shift_arr_k_bits_right_att_wrapper((u8 *) b, begin, end, a_size * 8, 8);
                //CONTINUE-FROM-HERE!
                //FIXME!
                //TODO!
                /* shift_arr_4bits_right_att_wrapper(c, begin, end, )
                assert(memcmp(att, val, a_size * 8) == 0);

                size_t k = (random() % 30) + 2;
                size_t max_items = (a_size * sizeof(a[0]) * 8 + k - 1) / k;
                size_t items = (random() % max_items) + 1;
                bool res = Shift_op::check::test_pack_unpack_array_gen_k(a, items, k);
                if (!res) {
                    std::cout << "reps: \t" << reps << std::endl;
                    std::cout << "i:    \t" << i << std::endl;
                }
                assert(res); */
            }
            std::cout << "reps: " << reps << std::endl;
        }
        return true;
    }
}// namespace Shift_op::check

namespace bitsMani {

    bool zero0s_between_k_ones_word(size_t k, size_t range, u64 word) {
        const u64 mask = (1ULL << range) - 1;
        const uint64_t y = _pdep_u64(mask << k, word);
        if (!y) return false;

        auto tz = _tzcnt_u64(y);
        u64 z = y >> tz;
        bool att = (z == mask);
        assert(att == bitsMani::test::zero0s_between_k_ones_word_naive(k, range, word));
        return att;
    }

    bool f2(size_t k, size_t range, u64 word) {
        const u64 mask = (1ULL << range) - 1;
        const uint64_t y = _pdep_u64(mask << k, word);
        return y && ((y >> _tzcnt_u64(y)) == mask);
    }

    bool f3(size_t k, size_t range, u64 word) {
        const u64 mask = (1ULL << range) - 1;
        const uint64_t y = _pdep_u64(mask << k, word);
        const uint64_t y_or = y | _blsmsk_u64(y);
        bool nec_consec_cond = (_blsr_u64(y_or + 1) == 0);
        return (y && nec_consec_cond) && (pop64(y) == range);
    }

    bool f4(size_t k, size_t range, u64 word) {
        static int c = 0;
        c++;
        const u64 mask = (1ULL << range) - 1;
        const uint64_t y = _pdep_u64(mask << k, word);
        const uint64_t bit = _blsi_u64(y);
        bool res = y && ((y + bit) == (bit << range));
        bool val = f3(k, range, word);
        if (res == val)
            return res;

        std::cout << "c: " << c << std::endl;
        std::cout << "word:  \t" << str_bitsMani::format_word_to_string(word) << std::endl;
        std::cout << "y:     \t" << str_bitsMani::format_word_to_string(y) << std::endl;
        std::cout << "bit:   \t" << str_bitsMani::format_word_to_string(bit) << std::endl;
        std::cout << "range: \t" << range << std::endl;
        std::cout << "k:     \t" << k << std::endl;
        std::cout << "val:   \t" << val << std::endl;
        std::cout << "res:   \t" << res << std::endl;
        std::cout << std::string(80, '?') << std::endl;
        std::cout << std::string(80, '!') << std::endl;
        return res;
    }

    bool f5_weaker(size_t k, size_t range, u64 word) {
        const u64 mask = (1ULL << range) - 1;
        const uint64_t y = _pdep_u64(mask << k, word);
        const uint64_t bit = _blsi_u64(y);
        bool res = ((y + bit) == (bit << range));
        return res;
    }

    bool f6(size_t k, size_t range, u64 word) {
        const u64 mask = (1ULL << range) - 1;
        const uint64_t y = _pdep_u64(mask << k, word);
        const uint64_t bit = _blsi_u64(y);
        bool res = ((y + bit) == (bit << range));
        return (res) ? y : 0;
    }

    bool f7(size_t k, size_t range, u64 word) {
        const u64 mask = (1ULL << range) - 1;
        const uint64_t y = _pdep_u64(mask << k, word);
        return _bextr_u64(word, _tzcnt_u64(y), range) == mask;
    }

    /* bool f8(size_t k, size_t range, u64 word) {
        const u64 mask = (1ULL << range) - 1;
        const uint64_t y = _pdep_u64(mask << k, word);
        return _pext_u64(y, word) == mask;
    } */

    bool compare_bits_ranged_naive(const u64 *a, u8 rem, size_t rem_length, size_t start_index1, size_t end_index1) {
        for (size_t i = start_index1; i < end_index1; i += rem_length) {
            if (compare_bits(a, rem, rem_length, i)) return true;
        }
        return false;
    }



    u64 bitmask_compare_k_packed_items(u64 word, u32 rem, size_t rem_length, size_t items) {
        // assert(rem_length <= 8);
        assert(rem_length);
        assert(rem_length * items <= 64);
        assert(items <= 64);
        const u64 mask = (1ULL << rem_length) - 1;
        assert(rem <= mask);
        u64 res = 0;
        u64 b = 1ULL;
        for (size_t i = 0; i < items; i++) {

            if ((word & mask) == rem) res |= b;
            word >>= rem_length;
            b <<= 1ull;
            /* if (((word ^ rem) & rem) == 0)
                return true; */
        }
        return res;
    }



    u64 bitmask_cmp_bits_inside_un_aligned_single_word(const u8 *a, u32 rem, size_t rem_length, size_t start_index1, size_t end_index1) {
        assert(((end_index1 - start_index1) % rem_length) == 0);
        const size_t items = (end_index1 - start_index1) / rem_length;
        // auto mp = (const u8 *) a;
        const size_t total_bits_to_compare = end_index1 - start_index1;
        const size_t offset = start_index1 & 7;
        assert(total_bits_to_compare + offset <= 64);

        u64 word = 0;
        memcpy(&word, a + (start_index1 / 8), 8);
        word >>= offset;
        return bitmask_compare_k_packed_items(word, rem, rem_length, items);
    }

    bool compare_bits_ranged(const u64 *a, u8 rem, size_t rem_length, size_t start_index1, size_t end_index1) {
        assert(((end_index1 - start_index1) % rem_length) == 0);
        const size_t items = (end_index1 - start_index1) / rem_length;
        auto mp = (const u8 *) a;
        const size_t total_bits_to_compare = end_index1 - start_index1;
        const size_t offset = start_index1 & 7;
        if (total_bits_to_compare + offset <= 64) {
            u64 word = 0;
            memcpy(&word, mp + (start_index1 / 8), 8);
            word >>= offset;
            return compare_k_packed_items(word, rem, rem_length, items);
        }
        size_t first_part_bits = (64 - offset) / rem_length * rem_length;
        assert(first_part_bits);
        auto temp = cmp_bits_inside_un_aligned_single_word(a, rem, rem_length, start_index1, start_index1 + first_part_bits);
        auto rest = (start_index1 + first_part_bits < end_index1) && compare_bits_ranged(a, rem, rem_length, start_index1 + first_part_bits, end_index1);
        return temp or rest;
    }

    u64 get_compare_mask(const u8 *a, u32 rem, size_t rem_length, size_t start_index1, size_t end_index1) {
        assert(((end_index1 - start_index1) % rem_length) == 0);
        const size_t items = (end_index1 - start_index1) / rem_length;
        assert(items <= 64);
        const size_t total_bits_to_compare = end_index1 - start_index1;
        const size_t offset = start_index1 & 7;
        if (total_bits_to_compare + offset <= 64) {
            auto mp = (const u8 *) a;
            u64 word = 0;
            memcpy(&word, mp + (start_index1 / 8), 8);
            word >>= offset;
            return bitmask_compare_k_packed_items(word, rem, rem_length, items);
        }
        const size_t part_one_items = (64 - offset) / rem_length;
        size_t first_part_bits = part_one_items * rem_length;
        assert(first_part_bits);
        u64 temp = bitmask_cmp_bits_inside_un_aligned_single_word(a, rem, rem_length, start_index1, start_index1 + first_part_bits);
        u64 rest = (start_index1 + first_part_bits < end_index1) ? get_compare_mask(a, rem, rem_length, start_index1 + first_part_bits, end_index1) : 0;
        assert(!(temp & (rest << part_one_items)));
        return temp | (rest << part_one_items);
    }

    bool compare_bits(const u64 *a, u8 rem, size_t rem_length, size_t index1) {
        assert(rem_length <= 8);
        const u8 mask = (1 << rem_length) - 1;
        assert(rem <= mask);
        return ((a[index1 / 64] >> (index1 & 63)) & mask) == rem;
    }

    bool compare_bits2(const u64 *a, u8 rem, size_t rem_length, size_t index1) {
        assert(rem_length <= 8);
        assert(rem <= ((1 << rem_length) - 1));
        return _bextr_u64(a[index1 / 64], index1 & 63, rem_length);
    }


    u64 extract_bits(const u8 *a, size_t start1, size_t end1) {
//        static int b = 0;
        assert(end1 - start1 <= 64);
        const size_t length = end1 - start1;
        const u64 mask = _bzhi_u64(-1, length);
        const size_t offset = start1 & 7;
        u64 word;
//        std::cout << "b: \t" << b++ << std::endl;
        memcpy(&word, &a[start1 / 8], 8);
        word >>= offset;
        if (offset + length <= 64) {
            return word & mask;
        } else {
            u64 hi = (a[(start1 / 8) + 8]) << (64 - offset);
            return mask | hi;
        }
    }
    void update_bits_inside_8bytes_boundaries(u8 *a, size_t start1, size_t end1, u64 value) {
        //This might contain a problem, of writing to a place i'm not supposed to, although i do not change it.
        assert(end1 - start1 <= 64);
        const size_t length = end1 - start1;
        assert(value <= _bzhi_u64(-1, length));
        const u64 mask = _bzhi_u64(-1, length);
        const size_t offset = start1 & 7;
        u64 word;
//        assert()
        memcpy(&word, &a[start1 / 8], 8);
        u64 shifted_mask = mask << offset;
        u64 mid = value << offset;
        u64 new_word = (word & ~shifted_mask) | mid;
        memcpy(&a[start1 / 8], &new_word, 8);
    }

    void update_bits_inside_8bytes_boundaries_safer(u8 *a, size_t start1, size_t k, u64 value) {
        //This might contain a problem, of writing to a place I'm not supposed to, although I do not change it.
//        static int A[4] = {0};
//        A[0]++;
        assert(k <= 32);
        size_t first_byte = start1 / 8;
        size_t last_byte = (start1 + k - 1) / 8;
        if (first_byte == last_byte){
//            constexpr u64 ind = (1ULL << 20) - 1;
//            if (((A[1]++) & ind) == 0){
//                std::cout << "A[1]: \t" << A[1] << std::endl;
//            }
            u8 prev = a[first_byte];
            size_t shift = start1 & 7;
            u8 mid_mask = ((1 << k) - 1) << shift;
            u8 outer = prev & ~mid_mask;
            u8 inner = value << shift;
            assert(!(inner & outer));
            a[first_byte] = inner | outer;
            return;
        }
//        std::cout << "A[2]: \t" << A[1]++ << std::endl;
//        A[0]++;
//        return;
        size_t bytes_to_write = last_byte - first_byte + 1;
        assert(bytes_to_write <= 8);
        const size_t length = k;
        assert(value <= _bzhi_u64(-1, length));
        const u64 mask = _bzhi_u64(-1, length);
        assert(mask);
        const size_t offset = start1 & 7;
        u64 word = 0;
        // 8 instead of bytes to write would be better, but this might give some warnings.
//        std::cout << "a[0]: \t" << (u16)a[0];// << std::endl;
//        std::cout << "\t bytes: \t" << bytes_to_write << std::endl;
//        std::cout << "A: \t" << A++ << std::endl;
        memcpy(&word, a + first_byte, bytes_to_write);
        u64 shifted_mask = mask << offset;
        u64 mid = value << offset;
        u64 inner = mid;
        u64 outer = (word & ~shifted_mask);
        assert(!(inner & outer));
        u64 new_word = (word & ~shifted_mask) | mid;
        memcpy(a + first_byte, &new_word, bytes_to_write);
    }

    void update_bits(u8 *a, size_t start1, size_t end1, u64 value) {
        assert(end1 - start1 <= 64);
        assert(value <= _bzhi_u64(-1, end1 - start1));
        if (end1 - start1 + (start1 & 7) <= 64)
            return update_bits_inside_8bytes_boundaries(a, start1, end1, value);

        const u64 val1 = value & _bzhi_u64(-1, end1 - (start1 + 8));
        const u64 val2 = value >> (end1 - (start1 + 8));
        update_bits_inside_8bytes_boundaries(a, start1, end1 - 8, val1);
        update_bits_inside_8bytes_boundaries(a, end1 - 8, end1, val2);
    }
}// namespace bitsMani
namespace bitsMani::test {
    bool zero0s_between_k_ones_word_naive(size_t k, size_t range, u64 word) {
        size_t begin = select64(word, k);
        // size_t temp = begin + range;
        size_t wanted_end = begin + range - 1;
        return (wanted_end < 64) && (select64(word, k + range - 1) == wanted_end);
    }

    bool val_zero0s_single(size_t k, size_t range, u64 word) {
        static int c = 0;
        c++;
        bool att = zero0s_between_k_ones_word(k, range, word);
        bool att2 = f3(k, range, word);
        bool att3 = f4(k, range, word);
        bool val = zero0s_between_k_ones_word_naive(k, range, word);
        bool all = (att == val) and (att2 == att3) and (att2 == att);
        assert(all);
        /* bool att8 = f8(k, range, word);
        if (att8 != val) {
            std::cout << "att8: " << att8 << std::endl;
            const u64 mask = (1ULL << range) - 1;
            const uint64_t y = _pdep_u64(mask << k, word);
            u64 z = _pext_u64(y, word);
            u64 z2 = _pext_u64(word, y);
            std::cout << "c:     \t" << c << std::endl;
            std::cout << "range: \t" << range << std::endl;
            std::cout << "k:     \t" << k << std::endl;
            std::cout << std::string(80, '=') << std::endl;
            std::cout << "word:  \t" << str_bitsMani::format_word_to_string(word) << std::endl;
            std::cout << "mask:  \t" << str_bitsMani::format_word_to_string(mask) << std::endl;
            std::cout << "y:     \t" << str_bitsMani::format_word_to_string(y) << std::endl;
            std::cout << "z:     \t" << str_bitsMani::format_word_to_string(z) << std::endl;
            std::cout << "z2:    \t" << str_bitsMani::format_word_to_string(z2) << std::endl;
            assert(0);
        } */
        if (all)
            return true;

        std::cout << "att: " << att << std::endl;
        std::cout << "att2: " << att2 << std::endl;
        std::cout << "val: " << val << std::endl;
        prt_zeros_failed(k, range, word);
        return false;
    }

    void prt_zeros_failed(size_t k, size_t range, u64 word) {

        bool val = zero0s_between_k_ones_word_naive(k, range, word);

        auto pop0 = pop64(word);
        size_t begin = select64(word, k);
        size_t end = select64(word, k + range);

        std::cout << std::endl;
        std::cout << std::string(80, '=') << std::endl;
        std::cout << std::endl;
        std::cout << "k:     \t" << k << std::endl;
        std::cout << "range: \t" << range << std::endl;
        std::cout << "pop0:  \t" << pop0 << std::endl;
        std::cout << "begin: \t" << begin << std::endl;
        std::cout << "end:   \t" << end << std::endl;
        std::cout << "word:  \t" << str_bitsMani::format_word_to_string(word) << std::endl;
        std::cout << std::string(80, '~') << std::endl;
        std::cout << "att: " << !val << std::endl;
        std::cout << "val: " << val << std::endl;
        std::cout << std::string(80, '~') << std::endl;
        const u64 mask = (1ULL << range) - 1;
        const uint64_t y = _pdep_u64(mask << k, word);
        // const uint64_t z = _blsmsk_u64(y);
        const uint64_t z2 = _blsmsk_u64(y);
        // const uint64_t u = _blsr_u64(_blsmsk_u64(_pdep_u64(mask << k, word)) + 1);
        bool b1 = (!_blsr_u64(z2 + 1));
        auto pop_y = pop64(y);
        std::cout << "mask:  \t" << str_bitsMani::format_word_to_string(mask) << std::endl;
        std::cout << "y:     \t" << str_bitsMani::format_word_to_string(y) << std::endl;
        std::cout << "pop_y: \t" << pop_y << std::endl;
        std::cout << "b1:    \t" << b1 << std::endl;
        std::cout << std::endl;
        std::cout << std::string(80, '=') << std::endl;
        std::cout << std::endl;
    }

    bool val_zero0s_between_k_ones_word_rand(size_t reps) {
        u64 word = 0;
        for (size_t i = 0; i < reps; i++) {
            while (!word) {
                for (size_t j = 0; j < 4; j++) {
                    word ^= random();
                }
            }
            assert(word);
            size_t k = random() % 64;
            size_t range = (random() % 16) + 1;
            bool temp = val_zero0s_single(k, range, word);
            if (!temp) {
                std::cout << "i: " << i << std::endl;
                assert(0);
            }
        }
        return true;
    }

    bool update_plus_extract(u8 *a, size_t start, size_t end, u64 value) {
        update_bits(a, start, end, value);
        u64 ext_val = extract_bits(a, start, end);
        return ext_val == value;
    }

    bool wrap_extract_update_bits() {
        constexpr size_t a_size = 8;
        constexpr size_t range = a_size * 64;
        u64 a[a_size];

        for (size_t reps = 0; reps < 64; reps++) {
            for (size_t j = 0; j < a_size; ++j) {
                a[j] = random() ^ random();
            }
            u64 b[64];
            memcpy(b, a, a_size * 8);
            for (size_t i = 0; i < a_size; i++) {
                size_t begin = 0;
                size_t end = 0;
                while (1) {
                    begin = random() % range;
                    end = random() % range;
                    bool c1 = (end <= begin);
                    bool c2 = end - begin <= 64;
                    if (c1 and c2)
                        break;
                }
                u64 mask = _bzhi_u64(-1, end - begin);
                u64 ext0 = extract_bits((const u8 *) b, begin, end);
                u64 val0 = random() & mask;
                bool r0 = update_plus_extract((u8 *) b, begin, end, val0);
                assert(r0);
                bool r1 = update_plus_extract((u8 *) b, begin, end, ext0);
                assert(r1);

                bool cmp = !memcmp(a, b, a_size * 8);
                assert(cmp);
                /* shift_arr_4bits_right_att_wrapper(c, begin, end, )
                assert(memcmp(att, val, a_size * 8) == 0);

                size_t k = (random() % 30) + 2;
                size_t max_items = (a_size * sizeof(a[0]) * 8 + k - 1) / k;
                size_t items = (random() % max_items) + 1;
                bool res = Shift_op::check::test_pack_unpack_array_gen_k(a, items, k);
                if (!res) {
                    std::cout << "reps: \t" << reps << std::endl;
                    std::cout << "i:    \t" << i << std::endl;
                }
                assert(res); */
            }
            // std::cout << "reps: " << reps << std::endl;
        }
        return true;
    }
}// namespace bitsMani::test
