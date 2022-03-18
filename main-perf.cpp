#include "Tests/smart_tests.hpp"

void bench_cmp_np();
void temp_main();

int main() {
    for (size_t i = 0; i < testSmart::ROUNDS; ++i) {
        bench_cmp_np();
    }
    return 0;
}


void bench_cmp_np() {
    //    using spare_item = uint64_t;
    using CF8 = cuckoofilter::CuckooFilter<uint64_t, 8>;
    using CF12 = cuckoofilter::CuckooFilter<uint64_t, 12>;
    using CF16 = cuckoofilter::CuckooFilter<uint64_t, 16>;
    using CF8_N2 = cuckoofilter::CuckooFilterStable<uint64_t, 8>;
    using CF12_Flex = cuckoofilter::CuckooFilterStable<uint64_t, 12>;
    using CF16_Flex = cuckoofilter::CuckooFilterStable<uint64_t, 16>;

    
    using inc4 = Prefix_Filter<SimdBlockFilterFixed<>>;
    using inc6 = Prefix_Filter<CF12_Flex>;
    using inc8 = Prefix_Filter<TC_shortcut>;
    // using inc9 = Prefix_Filter<Impala512<>>;

    using L_BF8 = bloomfilter::BloomFilter<uint64_t, 8, 0>;
    using L_BF12 = bloomfilter::BloomFilter<uint64_t, 12, 0>;
    using L_BF16 = bloomfilter::BloomFilter<uint64_t, 16, 0>;

    // constexpr size_t db_speeder = 1;
    constexpr size_t max_filter_capacity = ((1ULL << 28) * 94 / 100) / testSmart::db_speeder;// load is .94
    constexpr size_t lookup_reps = max_filter_capacity;
    constexpr size_t bench_precision = 20;
    std::vector<u64> v_add, v_find;

    testSmart::fill_vec_smart(&v_add, max_filter_capacity);
    testSmart::fill_vec_smart(&v_find, lookup_reps);


    const std::string path = "../scripts/Inputs/";

    constexpr bool to_print = false;
    
    testSmart::Bench_res_to_file_incremental_22<SimdBlockFilter<>>(max_filter_capacity, bench_precision, &v_add, &v_find, path, to_print);
    testSmart::Bench_res_to_file_incremental_22<SimdBlockFilterFixed<>>(max_filter_capacity, bench_precision, &v_add, &v_find, path, to_print);
    // testSmart::Bench_res_to_file_incremental_22<Impala512<>>(max_filter_capacity, bench_precision, &v_add, &v_find, path, to_print);
    
    testSmart::Bench_res_to_file_incremental_22<CF8>(max_filter_capacity, bench_precision, &v_add, &v_find, path, to_print);
    testSmart::Bench_res_to_file_incremental_22<CF12>(max_filter_capacity, bench_precision, &v_add, &v_find, path, to_print);
    testSmart::Bench_res_to_file_incremental_22<CF12_Flex>(max_filter_capacity, bench_precision, &v_add, &v_find, path, to_print);
    testSmart::Bench_res_to_file_incremental_22<TC_shortcut>(max_filter_capacity, bench_precision, &v_add, &v_find, path, to_print);
    
    // testSmart::Bench_res_to_file_incremental_22<inc4>(max_filter_capacity, bench_precision, &v_add, &v_find, path, to_print);
    testSmart::Bench_res_to_file_incremental_22<inc4>(max_filter_capacity, bench_precision, &v_add, &v_find, path, to_print);
    testSmart::Bench_res_to_file_incremental_22<inc6>(max_filter_capacity, bench_precision, &v_add, &v_find, path, to_print);
    testSmart::Bench_res_to_file_incremental_22<inc8>(max_filter_capacity, bench_precision, &v_add, &v_find, path, to_print);
    // testSmart::Bench_res_to_file_incremental_22<inc9>(max_filter_capacity, bench_precision, &v_add, &v_find, path, to_print);

    // testSmart::Bench_res_to_file_incremental_22<L_BF8>(max_filter_capacity, bench_precision, &v_add, &v_find, path, to_print);
    // testSmart::Bench_res_to_file_incremental_22<L_BF12>(max_filter_capacity, bench_precision, &v_add, &v_find, path, to_print);
    // testSmart::Bench_res_to_file_incremental_22<L_BF16>(max_filter_capacity, bench_precision, &v_add, &v_find, path, to_print);
    return;
}

