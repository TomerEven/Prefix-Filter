/* Taken from
 * https://github.com/FastFilter/fastfilter_cpp
 * */

//#pragma once
#ifndef FILTERS_WRAPPERS_HPP
#define FILTERS_WRAPPERS_HPP
// #define CONTAIN_ATTRIBUTES inline
#define CONTAIN_ATTRIBUTES __attribute__((always_inline))
// #define CONTAIN_ATTRIBUTES __attribute__((noinline))


#include <stdexcept>

#include "../Bloom_Filter/bloom.hpp"
#include "../Bloom_Filter/simd-block-fixed-fpp.h"
#include "../Bloom_Filter/Impala512.h"
#include "../Bloom_Filter/simd-block.h"
#include "../Prefix-Filter/min_pd256.hpp"
#include "../TC-Shortcut/TC-shortcut.hpp"
#include "../cuckoofilter/src/cuckoofilter.h"
#include "../cuckoofilter/src/cuckoofilter_stable.h"
// #include "linux-perf-events.h"
#include <map>

enum filter_id {
    Trivial_id,
    CF,
    SIMD,
    BBF,
    BBF_gen_id,
    SIMD_fixed,
    BBF_Flex,
    prefix_id,
    TC_shortcut_id,
    VQF_Wrapper_id,
    cf1ma_id,
    cf3ma_id,
    cf_stable_id,
    cf_flex_id,
    bloom_id,
    bf_ma_id,
    bloomSimple_id,
    bloomTrivial_id,
    bloomPower_id,
    bloomPowerDoubleHash_id,
    simple_bbf_id,
};

template<typename Table>
struct FilterAPI {
};

class TrivialFilter {
public:
    TrivialFilter(size_t max_items) {
    }

    __attribute__((always_inline)) inline static constexpr uint16_t fixed_reduce(uint16_t hash) {
        // http://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/
        return (uint16_t) (((uint32_t) hash * 6400) >> 16);
    }


    inline auto Find(const u64 &item) const -> bool {
        return true;
    }

    void Add(const u64 &item) {}

    auto get_capacity() const -> size_t {
        return -1;
    }

    auto get_name() const -> std::string {
        return "Trivial-Filter ";
    }

    auto get_byte_size() const -> size_t {
        return 0;
    }

    auto get_cap() const -> size_t {
        return -1;
    }
};

template<>
struct FilterAPI<TrivialFilter> {
    using Table = TrivialFilter;

    static Table ConstructFromAddCount(size_t add_count) {
        return Table(add_count);
    }

    static void Add(uint64_t key, Table *table) {
        table->Add(key);
    }
    CONTAIN_ATTRIBUTES static bool Contain(uint64_t key, const Table *table) {
        return table->Find(key);
    }

    static void Remove(uint64_t key, Table *table) {
        throw std::runtime_error("Unsupported");
    }

    static std::string get_name(const Table *table) {
        return table->get_name();
    }

    static auto get_functionality(const Table *table) -> uint32_t {
        return 0;
    }
    static auto get_ID(const Table *table) -> filter_id {
        return Trivial_id;
    }

    static size_t get_byte_size(const Table *table) {
        return table->get_byte_size();
    }

    static size_t get_cap(const Table *table) {
        return table->get_cap();
    }
};


template<typename ItemType, size_t bits_per_item, template<size_t> class TableType, typename HashFamily>
struct FilterAPI<cuckoofilter::CuckooFilter<ItemType, bits_per_item, TableType, HashFamily>> {
    using Table = cuckoofilter::CuckooFilter<ItemType, bits_per_item, TableType, HashFamily>;

    static Table ConstructFromAddCount(size_t add_count) {
        return Table(add_count);
    }

    static void Add(uint64_t key, Table *table) {
        if (table->Add(key) != cuckoofilter::Ok) {
            std::cerr << "Cuckoo filter is too full. Insertion of the element (" << key << ") failed.\n";
            std::cout << get_info(table).str() << std::endl;

            throw std::logic_error("The filter is too small to hold all of the elements");
        }
    }

    static bool Add_attempt(uint64_t key, Table *table) {
        if (table->Add(key) != cuckoofilter::Ok) {
            std::cout << get_info(table).str() << std::endl;
            return false;
            // throw std::logic_error("The filter is too small to hold all of the elements");
        }
        return true;
    }

    static void Remove(uint64_t key, Table *table) {
        table->Delete(key);
    }

    CONTAIN_ATTRIBUTES static bool Contain(uint64_t key, const Table *table) {
        return (0 == table->Contain(key));
    }

    static std::string get_name(const Table *table) {
        auto ss = table->Info();
        std::string temp = "PackedHashtable";
        if (ss.find(temp) != std::string::npos) {
            return "CF-ss";
        }
        if (bits_per_item == 8) {
            // return "Cuckoo-8-mod-m1";
            return "Cuckoo-8";
        } else if (bits_per_item == 12) {
            return "Cuckoo-12";
            // return "Cuckoo-12-mod-m1";
        } else if (bits_per_item == 16)
            return "Cuckoo-16";
        else if (bits_per_item == 32) {
            return "Cuckoo-32";
        }
        return "Cuckoo-?";
        // return "Cuckoo-" + std::to_string(bits_per_item);
    }

    static auto get_info(const Table *table) -> std::stringstream {
        std::string state = table->Info();
        std::stringstream ss;
        ss << state;
        return ss;
        // std::cout << state << std::endl;
    }

    /**
     * Returns int indicating which function can the filter do.
     * 1 is for lookups.
     * 2 is for adds.
     * 4 is for deletions.
     */
    static auto get_functionality(const Table *table) -> uint32_t {
        return 7;
    }

    static auto get_ID(const Table *table) -> filter_id {
        return CF;
    }

    static size_t get_byte_size(const Table *table) {
        return table->SizeInBytes();
    }

    static size_t get_cap(const Table *table) {
        return table->get_cap();
    }
};
template<typename ItemType, size_t bits_per_item, template<size_t> class TableType, typename HashFamily>
struct FilterAPI<cuckoofilter::CuckooFilterStable<ItemType, bits_per_item, TableType, HashFamily>> {
    using Table = cuckoofilter::CuckooFilterStable<ItemType, bits_per_item, TableType, HashFamily>;

    static Table ConstructFromAddCount(size_t add_count) {
        return Table(add_count);
    }

    static void Add(uint64_t key, Table *table) {
        if (table->Add(key) != cuckoofilter::Ok) {
            std::cerr << "Stable Cuckoo filter is too full. Insertion of the element (" << key << ") failed.\n";
            std::cout << get_info(table).str() << std::endl;

            throw std::logic_error("The filter is too small to hold all of the elements");
        }
    }

    static void Remove(uint64_t key, Table *table) {
        table->Delete(key);
    }

    CONTAIN_ATTRIBUTES static bool Contain(uint64_t key, const Table *table) {
        return (0 == table->Contain(key));
    }

    static std::string get_name(const Table *table) {
        auto ss = table->Info();
        std::string temp = "PackedHashtable";
        if (ss.find(temp) != std::string::npos) {
            return "CF-ss";
        }
        if (bits_per_item == 8) {
            // return "Cuckoo-8-mod-m1";
            return "CuckooStable-8";
        } else if (bits_per_item == 12) {
            return "CuckooStable-12";
        } else if (bits_per_item == 16)
            return "CuckooStable-16";
        else if (bits_per_item == 32) {
            return "CuckooStable-32";
        }
        return "Cuckoo-?";
    }

    static auto get_info(const Table *table) -> std::stringstream {
        std::string state = table->Info();
        std::stringstream ss;
        ss << state;
        return ss;
        // std::cout << state << std::endl;
    }

    /**
             * Returns int indicating which function can the filter do.
             * 1 is for lookups.
             * 2 is for adds.
             * 4 is for deletions.
             */
    static auto get_functionality(const Table *table) -> uint32_t {
        return 7;
    }

    static auto get_ID(const Table *table) -> filter_id {
        return cf_stable_id;
    }
    static size_t get_byte_size(const Table *table) {
        return table->SizeInBytes();
    }

    static size_t get_cap(const Table *table) {
        return table->get_cap();
    }

    static double get_eLoad(const Table *table) {
        return table->get_effective_load();
    }
};

template<>
struct FilterAPI<SimdBlockFilter<>> {
    using Table = SimdBlockFilter<>;

    static Table ConstructFromAddCount(size_t add_count) {
        Table ans(ceil(log2(add_count * 8.0 / CHAR_BIT)));
        return ans;
    }

    static void Add(uint64_t key, Table *table) {
        table->Add(key);
    }

    CONTAIN_ATTRIBUTES static bool Contain(uint64_t key, const Table *table) {
        return table->Find(key);
    }

    static void Remove(uint64_t key, Table *table) {
        throw std::runtime_error("Unsupported");
    }

    static std::string get_name(const Table *table) {
        return "SimdBlockFilter";
    }

    static auto get_info(const Table *table) -> std::stringstream {
        assert(false);
        std::stringstream ss;
        return ss;
    }

    /**
     * Returns int indicating which function can the filter do.
     * 1 is for lookups.
     * 2 is for adds.
     * 4 is for deletions.
     */
    static auto get_functionality(const Table *table) -> uint32_t {
        return 3;
    }
    static auto get_ID(const Table *table) -> filter_id {
        return BBF;
    }

    static size_t get_byte_size(const Table *table) {
        return table->SizeInBytes();
    }

    static size_t get_cap(const Table *table) {
        return table->get_cap();
    }
};


template<>
struct FilterAPI<SimdBlockFilterFixed<>> {
    using Table = SimdBlockFilterFixed<>;

    static Table ConstructFromAddCount(size_t add_count) {
        Table ans(ceil(add_count * 8.0 / CHAR_BIT));
        return ans;
    }

    static void Add(uint64_t key, Table *table) {
        table->Add(key);
    }

    CONTAIN_ATTRIBUTES static bool Contain(uint64_t key, const Table *table) {
        return table->Find(key);
    }

    static void Remove(uint64_t key, Table *table) {
        throw std::runtime_error("Unsupported");
    }

    static std::string get_name(const Table *table) {
        return "BBF-Fixed";
    }

    static auto get_info(const Table *table) -> std::stringstream {
        assert(false);
        std::stringstream ss;
        return ss;
    }

    /**
     * Returns int indicating which function can the filter do.
     * 1 is for lookups.
     * 2 is for adds.
     * 4 is for deletions.
     */
    static auto get_functionality(const Table *table) -> uint32_t {
        return 3;
    }
    static auto get_ID(const Table *table) -> filter_id {
        return SIMD_fixed;
    }
    static size_t get_byte_size(const Table *table) {
        return table->SizeInBytes();
    }

    static size_t get_cap(const Table *table) {
        return table->get_cap();
    }
};

template<>
struct FilterAPI<Impala512<>> {
    using Table = Impala512<>;

    static Table ConstructFromAddCount(size_t add_count) {
        return Table(add_count);
        // return ans;
    }

    static void Add(uint64_t key, Table *table) {
        table->Add(key);
    }

    CONTAIN_ATTRIBUTES static bool Contain(uint64_t key, const Table *table) {
        return table->Find(key);
    }

    static void Remove(uint64_t key, Table *table) {
        throw std::runtime_error("Unsupported");
    }

    static std::string get_name(const Table *table) {
        return "Impala512";
    }

    static auto get_info(const Table *table) -> std::stringstream {
        assert(false);
        std::stringstream ss;
        return ss;
    }

    /**
     * Returns int indicating which function can the filter do.
     * 1 is for lookups.
     * 2 is for adds.
     * 4 is for deletions.
     */
    static auto get_functionality(const Table *table) -> uint32_t {
        return 3;
    }
    static auto get_ID(const Table *table) -> filter_id {
        return SIMD_fixed;
    }
    static size_t get_byte_size(const Table *table) {
        return table->SizeInBytes();
    }

    static size_t get_cap(const Table *table) {
        return table->get_cap();
    }
};


template<>
struct FilterAPI<TC_shortcut> {
    using Table = TC_shortcut;
    //    using Table = dict512<TableType, spareItemType, itemType>;

    static Table ConstructFromAddCount(size_t add_count) {
        constexpr float load = .935;
        return Table(add_count, load);
    }

    static void Add(uint64_t key, Table *table) {
        if (!table->insert(key)) {
            std::cout << table->info() << std::endl;
            //            std::cout << "max_load: \t" << 0.945 << std::endl;
            throw std::logic_error(table->get_name() + " is too small to hold all of the elements");
        }
    }

    static bool Add_attempt(uint64_t key, Table *table) {
        if (!table->insert(key)) {
            std::cout << "load when failed: \t" << table->get_effective_load() << std::endl;
            std::cout << table->info() << std::endl;
            return false;
        }
        return true;
    }

    static void Remove(uint64_t key, Table *table) {
        // throw std::runtime_error("Unsupported");
        table->remove(key);
    }

    CONTAIN_ATTRIBUTES static bool Contain(uint64_t key, const Table *table) {
        // std::cout << "here!!!" << std::endl;
        return table->lookup(key);
        // return table->lookup_consecutive_only_body(key);
        // return table->lookup_consecutive(key);
    }

    static std::string get_name(const Table *table) {
        return table->get_name();
    }

    static auto get_info(const Table *table) -> std::stringstream {
        std::stringstream ss;
        ss << "";
        return ss;
        // return table->get_extended_info();
    }

    static auto get_functionality(const Table *table) -> uint32_t {
        return 7;
    }

    static auto get_ID(const Table *table) -> filter_id {
        return TC_shortcut_id;
    }

    static size_t get_byte_size(const Table *table) {
        return table->get_byte_size();
    }

    static size_t get_cap(const Table *table) {
        return table->get_cap();
    }

    static double get_eLoad(const Table *table) {
        return table->get_effective_load();
    }
};


template<typename Table>
inline size_t get_l2_slots(size_t l1_items, const double overflowing_items_ratio, const float loads[2]) {
    const double expected_items_reaching_next_level = l1_items * overflowing_items_ratio;
    size_t slots_in_l2 = (expected_items_reaching_next_level / loads[1]);
    return slots_in_l2;
}

template<>
inline size_t get_l2_slots<cuckoofilter::CuckooFilterStable<u64, 12>>(size_t l1_items, const double overflowing_items_ratio, const float loads[2]) {
    // const double expected_items_reaching_next_level = l1_items * 0.0752;
    // const double spare_workload = 0.0752 / 0.0586;
    // size_t slots_in_l2 = std::ceil(expected_items_reaching_next_level);
    // return slots_in_l2;
    constexpr auto expected_items95 = 0.0586;
    constexpr auto expected_items100 = 0.07952;
    constexpr auto expected_items105 = 0.1031;
    constexpr auto spare_workload = 0.94;
    constexpr auto safety = 1.08;
    constexpr auto factor95 = safety * expected_items95 / spare_workload;
    constexpr auto factor100 = safety * expected_items100 / spare_workload;
    constexpr auto factor105 = safety * expected_items105 / spare_workload;
    // const double expected_items_reaching_next_level = l1_items * (0.06 / 0.9);
    const double expected_items_reaching_next_level = l1_items * factor95;
    return expected_items_reaching_next_level;
}

template<>
inline size_t get_l2_slots<TC_shortcut>(size_t l1_items, const double overflowing_items_ratio, const float loads[2]) {
    constexpr auto expected_items95 = 0.0586;
    constexpr auto expected_items100 = 0.07952;
    constexpr auto expected_items105 = 0.1031;
    constexpr auto spare_workload = 0.935;
    constexpr auto safety = 1.08;
    constexpr auto factor95 = safety * expected_items95 / spare_workload;
    constexpr auto factor100 = safety * expected_items100 / spare_workload;
    constexpr auto factor105 = safety * expected_items105 / spare_workload;
    // const double expected_items_reaching_next_level = l1_items * (0.06 / 0.9);
    const double expected_items_reaching_next_level = l1_items * factor95;
    size_t slots_in_l2 = std::ceil(expected_items_reaching_next_level);
    return slots_in_l2;
}


template<>
inline size_t get_l2_slots<SimdBlockFilter<>>(size_t l1_items, const double overflowing_items_ratio, const float loads[2]) {
    const double expected_items_reaching_next_level = l1_items * overflowing_items_ratio;
    size_t slots_in_l2 = (expected_items_reaching_next_level / loads[1]);
    return slots_in_l2 * 4;
}

template<>
inline size_t get_l2_slots<SimdBlockFilterFixed<>>(size_t l1_items, const double overflowing_items_ratio, const float loads[2]) {
    const double expected_items_reaching_next_level = l1_items * overflowing_items_ratio;
    size_t slots_in_l2 = (expected_items_reaching_next_level / loads[1]);
    return slots_in_l2 * 2;
}

template<>
inline size_t get_l2_slots<Impala512<>>(size_t l1_items, const double overflowing_items_ratio, const float loads[2]) {
    constexpr auto expected_items95 = 0.0586;
    constexpr auto expected_items100 = 0.07952;
    constexpr auto expected_items105 = 0.1031;
    constexpr auto spare_workload = 1;
    constexpr auto safety = 1.08;
    constexpr auto factor95 = safety * expected_items95 / spare_workload;
    constexpr auto factor100 = safety * expected_items100 / spare_workload;
    constexpr auto factor105 = safety * expected_items105 / spare_workload;
    // const double expected_items_reaching_next_level = l1_items * (0.06 / 0.9);
    const double expected_items_reaching_next_level = l1_items * factor95;
    size_t slots_in_l2 = std::ceil(expected_items_reaching_next_level);
    return slots_in_l2;
}


template<typename Table>
class Prefix_Filter {
    const size_t filter_max_capacity;
    const size_t number_of_pd;
    size_t cap[2] = {0};

    hashing::TwoIndependentMultiplyShift Hasher, H0;
    __m256i *pd_array;
    Table GenSpare;

    static double constexpr overflowing_items_ratio = 0.0586;//  = expected_items95

public:
    Prefix_Filter(size_t max_items, const float loads[2])
        : filter_max_capacity(max_items),
          number_of_pd(std::ceil(1.0 * max_items / (min_pd::MAX_CAP0 * loads[0]))),
          GenSpare(FilterAPI<Table>::ConstructFromAddCount(get_l2_slots<Table>(max_items, overflowing_items_ratio, loads))),
          Hasher(), H0() {

        int ok = posix_memalign((void **) &pd_array, 32, 32 * number_of_pd);
        if (ok != 0) {
            std::cout << "Space allocation failed!" << std::endl;
            assert(false);
            exit(-3);
        }

        constexpr uint64_t pd256_plus_init_header = (((INT64_C(1) << min_pd::QUOTS) - 1) << 6) | 32;
        std::fill(pd_array, pd_array + number_of_pd, __m256i{pd256_plus_init_header, 0, 0, 0});

        // size_t l1 = sizeof(__m256i) * number_of_pd;
        // size_t l2 = FilterAPI<Table>::get_byte_size(&GenSpare);
        // double ratio = 1.0 * l2 / l1;
        // std::cout << get_name() << ".\t";
        // std::cout << "spare-size / First level:\t\t " << ratio << std::endl;
    }

    ~Prefix_Filter() {
        free(pd_array);
    }

    __attribute__((always_inline)) inline static constexpr uint32_t reduce32(uint32_t hash, uint32_t n) {
        // http://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/
        return (uint32_t) (((uint64_t) hash * n) >> 32);
    }


    __attribute__((always_inline)) inline static constexpr uint16_t fixed_reduce(uint16_t hash) {
        // http://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/
        return (uint16_t) (((uint32_t) hash * 6400) >> 16);
    }


    inline auto Find(const u64 &item) const -> bool {
        const u64 s = H0(item);
        uint32_t out1 = s >> 32u, out2 = s;
        const uint32_t pd_index = reduce32(out1, (uint32_t) number_of_pd);
        const uint16_t qr = fixed_reduce(out2);
        const int64_t quot = qr >> 8;
        const uint8_t rem = qr;
        // return min_pd::pd_find_25(quot, rem, &pd_array[pd_index]);
        // return (!min_pd::cmp_qr1(qr, &pd_array[pd_index])) ? min_pd::pd_find_25(quot, rem, &pd_array[pd_index])
        return (!min_pd::cmp_qr1(qr, &pd_array[pd_index])) ? min_pd::find_core(quot, rem, &pd_array[pd_index])
                                                           : incSpare_lookup(pd_index, qr);
    }

    inline auto incSpare_lookup(size_t pd_index, u16 qr) const -> bool {
        const u64 data = (pd_index << 13u) | qr;
        //        u64 hashed_res = Hasher(data);
        return FilterAPI<Table>::Contain(data, &GenSpare);
    }

    inline void incSpare_add(size_t pd_index, const min_pd::add_res &a_info) {
        cap[1]++;
        u16 qr = (((u16) a_info.quot) << 8u) | a_info.rem;
        const u64 data = (pd_index << 13u) | qr;
        //        u64 hashed_res = Hasher(data);
        return FilterAPI<Table>::Add(data, &GenSpare);
    }

    void Add(const u64 &item) {
        const u64 s = H0(item);
        constexpr u64 full_mask = (1ULL << 55);
        uint32_t out1 = s >> 32u, out2 = s;

        const uint32_t pd_index = reduce32(out1, (uint32_t) number_of_pd);

        auto pd = pd_array + pd_index;
        const uint64_t header = reinterpret_cast<const u64 *>(pd)[0];
        const bool not_full = !(header & full_mask);

        const uint16_t qr = fixed_reduce(out2);
        const int64_t quot = qr >> 8;
        const uint8_t rem = qr;

        if (not_full) {
            cap[0]++;
            assert(!min_pd::is_pd_full(pd));
            size_t end = min_pd::select64(header >> 6, quot);
            assert(min_pd::check::val_header(pd));
            const size_t h_index = end + 6;
            const u64 mask = _bzhi_u64(-1, h_index);
            const u64 lo = header & mask;
            const u64 hi = ((header & ~mask) << 1u);// & h_mask;
            assert(!(lo & hi));
            const u64 h7 = lo | hi;
            memcpy(pd, &h7, 7);

            assert(min_pd::check::val_header(pd));

            const size_t body_index = end - quot;
            min_pd::body_add_case0_avx(body_index, rem, pd);
            // auto mp = (u8 *) pd + 7 + body_index;
            // const size_t b2m = (32 - 7) - (body_index + 1);
            // memmove(mp + 1, mp, b2m);
            // mp[0] = rem;
            assert(min_pd::find_core(quot, rem, pd));
            assert(Find(item));
            return;
        } else {
            auto add_res = min_pd::new_pd_swap_short(quot, rem, pd);
            assert(min_pd::check::val_last_quot_is_sorted(pd));
            incSpare_add(pd_index, add_res);
            assert(Find(item));
        }
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////// Validation functions.////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////

    auto get_capacity() const -> size_t {
        size_t res = 0;
        for (size_t i = 0; i < number_of_pd; ++i) {
            res += min_pd::get_capacity(&pd_array[i]);
        }
        assert(res == cap[0]);
        return res;
    }

    auto get_name() const -> std::string {
        std::string s0 = "Prefix-Filter ";
        std::string s1 = FilterAPI<Table>::get_name(&GenSpare);
        return s0 + "[ " + s1 + " ]";
    }

    auto count_overflowing_PDs() -> size_t {
        size_t count_overflowing_PD = 0;
        for (int i = 0; i < number_of_pd; ++i) {
            bool add_cond = min_pd::pd_full(&pd_array[i]);
            count_overflowing_PD += add_cond;
            bool is_full = min_pd::pd_full(&pd_array[i]);
            //            bool is_full2 = pd_vec[i]->is_full();
            //            assert(is_full == is_full2);
            bool final = (!add_cond or is_full);
            // assert(final);
        }
        return count_overflowing_PD;
    }

    auto count_empty_PDs() -> size_t {
        size_t count_empty_PD = 0;
        for (int i = 0; i < number_of_pd; ++i) {
            bool add_cond = (min_pd::get_capacity(&pd_array[i]) <= 0);
            count_empty_PD += add_cond;
        }
        return count_empty_PD;
    }

    auto get_byte_size() const -> size_t {
        size_t l1 = sizeof(__m256i) * number_of_pd;
        //        size_t l2 = FilterAPI<Table>::get_byte_size(GenSpare);
        size_t l2 = FilterAPI<Table>::get_byte_size(&GenSpare);
        auto res = l1 + l2;
        return res;
    }

    auto get_cap() const -> size_t {
        return cap[0] + cap[1];
    }
};


template<typename filterTable>
struct FilterAPI<Prefix_Filter<filterTable>> {
    using Table = Prefix_Filter<filterTable>;

    static Table ConstructFromAddCount(size_t add_count) {
        constexpr float loads[2] = {.95, .95};
        // std::cout << "Lower workload" << std::endl;
        // std::cout << "Workload 1!" << std::endl;
        return Table(add_count, loads);
    }

    static void Add(u64 key, Table *table) {
        table->Add(key);
    }
    static void Remove(u64 key, Table *table) {
        throw std::runtime_error("Unsupported");
    }

    CONTAIN_ATTRIBUTES static bool Contain(u64 key, const Table *table) {
        return table->Find(key);
    }

    static std::string get_name(const Table *table) {
        return table->get_name();
    }

    static auto get_functionality(const Table *table) -> uint32_t {
        return 3;
    }

    static auto get_ID(const Table *table) -> filter_id {
        return prefix_id;
    }

    static size_t get_byte_size(const Table *table) {
        return table->get_byte_size();
    }

    static size_t get_cap(const Table *table) {
        return table->get_cap();
    }
};


template<typename ItemType,
         size_t bits_per_item,
         bool branchless,
         typename HashFamily>
struct FilterAPI<bloomfilter::BloomFilter<ItemType, bits_per_item, branchless, HashFamily>> {
    using Table = bloomfilter::BloomFilter<ItemType, bits_per_item, branchless, HashFamily>;

    static Table ConstructFromAddCount(size_t add_count) {
        return Table(add_count);
    }

    static void Add(uint64_t key, Table *table) {
        table->Add(key);
    }

    static void Remove(uint64_t key, Table *table) {
        throw std::runtime_error("Unsupported");
    }

    inline static bool Contain(uint64_t key, const Table *table) {
        return table->Contain(key) == bloomfilter::Ok;
    }

    static std::string get_name(const Table *table) {
        return table->get_name();
    }

    static auto get_info(const Table *table) -> std::stringstream {
        assert(0);
        std::stringstream ss;
        return ss;
    }

    static auto get_functionality(const Table *table) -> uint32_t {
        return 3;
    }
    static auto get_ID(const Table *table) -> filter_id {
        return bloom_id;
    }

    static size_t get_byte_size(const Table *table) {
        return table->SizeInBytes();
    }

    static size_t get_cap(const Table *table) {
        return -1;
        // return table->get_cap();
    }
};

/* 
// The statistics gathered for each table type:
struct Statistics {
    size_t add_count;
    double nanos_per_add;
    double nanos_per_remove;
    // key: percent of queries that were expected to be positive
    map<int, double> nanos_per_finds;
    double false_positive_probabilty;
    double bits_per_item;
};


// Output for the first row of the table of results. type_width is the maximum number of
// characters of the description of any table type, and find_percent_count is the number
// of different lookup statistics gathered for each table. This function assumes the
// lookup expected positive probabiilties are evenly distributed, with the first being 0%
// and the last 100%.
inline string StatisticsTableHeader(int type_width, const std::vector<double> &found_probabilities) {
    ostringstream os;

    os << string(type_width, ' ');
    os << setw(8) << right << "";
    os << setw(8) << right << "";
    for (size_t i = 0; i < found_probabilities.size(); ++i) {
        os << setw(8) << "find";
    }
    os << setw(8) << "1*add+";
    os << setw(8) << "" << setw(11) << "" << setw(11)
       << "optimal" << setw(8) << "wasted" << setw(8) << "million" << endl;

    os << string(type_width, ' ');
    os << setw(8) << right << "add";
    os << setw(8) << right << "remove";
    for (double prob : found_probabilities) {
        os << setw(8 - 1) << static_cast<int>(prob * 100.0) << '%';
    }
    os << setw(8) << "3*find";
    os << setw(9) << "ε%" << setw(11) << "bits/item" << setw(11)
       << "bits/item" << setw(8) << "space%" << setw(8) << "keys";
    return os.str();
}

// Overloading the usual operator<< as used in "std::cout << foo", but for Statistics
template<class CharT, class Traits>
basic_ostream<CharT, Traits> &operator<<(
        basic_ostream<CharT, Traits> &os, const Statistics &stats) {
    os << fixed << setprecision(2) << setw(8) << right
       << stats.nanos_per_add;
    double add_and_find = 0;
    os << fixed << setprecision(2) << setw(8) << right
       << stats.nanos_per_remove;
    for (const auto &fps : stats.nanos_per_finds) {
        os << setw(8) << fps.second;
        add_and_find += fps.second;
    }
    add_and_find = add_and_find * 3 / stats.nanos_per_finds.size();
    add_and_find += stats.nanos_per_add;
    os << setw(8) << add_and_find;

    // we get some nonsensical result for very small fpps
    if (stats.false_positive_probabilty > 0.0000001) {
        const auto minbits = log2(1 / stats.false_positive_probabilty);
        os << setw(8) << setprecision(4) << stats.false_positive_probabilty * 100
           << setw(11) << setprecision(2) << stats.bits_per_item << setw(11) << minbits
           << setw(8) << setprecision(1) << 100 * (stats.bits_per_item / minbits - 1)
           << " " << setw(7) << setprecision(3) << (stats.add_count / 1000000.);
    } else {
        os << setw(8) << setprecision(4) << stats.false_positive_probabilty * 100
           << setw(11) << setprecision(2) << stats.bits_per_item << setw(11) << 64
           << setw(8) << setprecision(1) << 0
           << " " << setw(7) << setprecision(3) << (stats.add_count / 1000000.);
    }
    return os;
}

struct samples {
    double found_probability;
    std::vector<uint64_t> to_lookup_mixed;
    size_t true_match;
    size_t actual_sample_size;
};

typedef struct samples samples_t;
 */
#endif
