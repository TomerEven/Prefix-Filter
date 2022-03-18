
#include "min_pd256.hpp"

namespace min_pd::check {
    bool validate_decoding(const __m256i *pd) {
        constexpr size_t t = MAX_CAP0 - 1;
        const u64 dirty_h = ((const u64 *) pd)[0];
        size_t status = dirty_h & 31;
        size_t last_quot = select64(~(dirty_h >> 6), t) - t;
        if(status != last_quot) {
            std::cout << "status:    \t" << status << std::endl;
            std::cout << "last_quot: \t" << last_quot << std::endl;
            assert(0);
        }
        return true;
    }

    bool val_header(const __m256i *pd) {
        const uint64_t h0 = reinterpret_cast<const u64 *>(pd)[0];
        const u64 h = (h0 >> 6) & H_mask;
        auto pop0 = _mm_popcnt_u64(h);
        assert(pop0 == QUOTS);
        return true;
    }

    bool val_last_quot_is_sorted(const __m256i *pd){
        if (!did_pd_overflowed(pd))
            return true;
        size_t last_quot = get_last_occupied_quot_only_full_pd(pd);
        size_t lq_cap = get_spec_quot_cap(last_quot, pd);
        assert(lq_cap);
        if (lq_cap == 1)
            return true;

        auto mp = (u8*)pd + 32 - lq_cap;
        for (size_t i = 1; i < lq_cap; ++i) {
            assert(mp[i-1] <= mp[i]);
        }
        return true;
    }
}// namespace min_pd::check
