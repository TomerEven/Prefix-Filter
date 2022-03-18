#include "tc-sym.hpp"


namespace tc_sym::check {

    auto validate_number_of_quotient(const __m512i *pd) -> bool {
        auto pd64 = (const u64 *) pd;
        u64 h0 = pd64[0];
        u64 h1 = pd64[1];
        u64 h1_masked = pd64[1] & _bzhi_u64(-1, QUOTS + MAX_CAP - 64);
        auto pop0 = _mm_popcnt_u64(h0);
        auto pop1 = _mm_popcnt_u64(h1_masked);
        auto pop1_extended = _mm_popcnt_u64(h1);
        auto pop1_e2 = _mm_popcnt_u64(h1 & _bzhi_u64(-1, 40));
        if (pop0 + pop1 == QUOTS)
            return true;


        std::cout << "h0:   \t" << format_word_to_string(h0) << std::endl;
        std::cout << "h1:   \t" << format_word_to_string(h1) << std::endl;
        std::cout << "h1_m: \t" << format_word_to_string(h1_masked) << std::endl;

        assert(0);
        return false;
    }


    auto validate_number_of_quotient(const __m512i *pd, const __m512i *backup_pd) -> bool {
        auto pd64 = (const u64 *) pd;
        u64 h0 = pd64[0];
        u64 h1 = pd64[1];
        u64 h1_masked = pd64[1] & _bzhi_u64(-1, QUOTS + MAX_CAP - 64);
        auto pop0 = _mm_popcnt_u64(h0);
        auto pop1 = _mm_popcnt_u64(h1_masked);
        auto pop1_extended = _mm_popcnt_u64(h1);
        auto pop1_e2 = _mm_popcnt_u64(h1 & _bzhi_u64(-1, 40));
        if (pop0 + pop1 == QUOTS)
            return true;


        auto bpd64 = (const u64 *) backup_pd;
        std::cout << std::string(80, '=') << std::endl;
        std::cout << "h0:   \t" << format_word_to_string(h0) << std::endl;
        std::cout << "h0:   \t" << format_word_to_string(bpd64[0]) << std::endl;
        std::cout << std::endl;
        std::cout << "h1:   \t" << format_word_to_string(h1) << std::endl;
        std::cout << "h1:   \t" << format_word_to_string(bpd64[1]) << std::endl;
        std::cout << std::endl;
        std::cout << "h1_m: \t" << format_word_to_string(h1_masked) << std::endl;
        std::cout << "h1_m: \t" << format_word_to_string(bpd64[1] & _bzhi_u64(-1, 40)) << std::endl;
        std::cout << std::string(80, '=') << std::endl;
        // std::cout << std::string(80, '~') << std::endl;
        // std::cout << "h1:   \t" << format_word_to_string(bpd64[1]) << std::endl;

        assert(0);
        return false;
    }


}// namespace tc_sym::check

// for printing
namespace tc_sym::check {
    void p_format_word(uint64_t x) {
        std::string res = to_bin(x, 64);
        std::cout << space_string(res) << std::endl;
    }

    auto format_word_to_string(uint64_t x, size_t length) -> std::string {
        std::string res = to_bin(x, length);// + "\n";
        return space_string(res);
        // std::cout << space_string(res) << std::endl;
    }

    /**
     * @brief This prints the binary representation of x. Usually, a reversed representation is needed.
     *
     * @param x
     * @param length
     * @return std::string
     */
    auto to_bin(uint64_t x, size_t length) -> std::string {
        assert(length <= 64);
        uint64_t b = 1ULL;
        std::string res;
        for (size_t i = 0; i < length; i++) {
            res += (b & x) ? "1" : "0";
            b <<= 1ul;
        }
        return res;
    }

    auto space_string(std::string s) -> std::string {
        std::string new_s = "";
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

    void print_pd_first_two_words(const __m512i *pd, size_t tabs = 1) {
        uint64_t h0 = 0, h1 = 0;
        memcpy(&h0, pd, 8);
        memcpy(&h1, reinterpret_cast<const uint64_t *>(pd) + 1, 8);
        auto s0 = format_word_to_string(h0, 64);
        auto s1 = format_word_to_string(h1, 64);

        auto tabs_to_add = std::string(tabs, '\t');
        //        size_t pops[2] = {_mm_popcnt_u64(h0), _mm_popcnt_u64(h1)};
        std::cout << "h0: \t" + tabs_to_add << s0;
        std::cout << "\t|\t (" << _mm_popcnt_u64(h0) << ", " << (64 - _mm_popcnt_u64(h0)) << ")" << std::endl;
        std::cout << "h1: \t" + tabs_to_add << s1;
        std::cout << "\t|\t (" << _mm_popcnt_u64(h1) << ", " << (64 - _mm_popcnt_u64(h1)) << ")" << std::endl;
    }

    void print_pd(const __m512i *pd) {
        std::cout << std::string(80, '~') << std::endl;
        //        assert(pd512::get_capacity(pd) == pd512::get_capacity_naive(pd));
        std::cout << "pd capacity:" << get_cap(pd) << std::endl;
        //        v_pd512_plus::print_headers_extended(pd);
        print_pd_first_two_words(pd);
        auto mp = ((const u8 *) pd) + 16;
        auto body_str = str_bitsMani::str_array_with_line_numbers(mp, MAX_CAP);
        std::cout << body_str << std::endl;
        std::cout << std::string(80, '~') << std::endl;
    }

}// namespace tc_sym::check
