
#include "Tests/smart_tests.hpp"

void bench_to_csv();

int main() {
    // constexpr int k1 = (int) ((double) 8 * 0.693147180559945 + 0.5);
    // constexpr int k2 = (int) ((double) 12 * 0.693147180559945 + 0.5);
    // constexpr int k3 = (int) ((double) 16 * 0.693147180559945 + 0.5);
    bench_to_csv();
    return 0;
}


void bench_to_csv() {
    using CF8 = cuckoofilter::CuckooFilter<uint64_t, 8>;
    using CF12 = cuckoofilter::CuckooFilter<uint64_t, 12>;
    using CF16 = cuckoofilter::CuckooFilter<uint64_t, 16>;
    // using CF32 = cuckoofilter::CuckooFilter<uint64_t, 32>;

    using CF8_Flex = cuckoofilter::CuckooFilterStable<uint64_t, 8>;
    using CF12_Flex = cuckoofilter::CuckooFilterStable<uint64_t, 12>;
    // using CF16_Flex = cuckoofilter::CuckooFilterStable<uint64_t, 16>;

    using inc4 = Prefix_Filter<SimdBlockFilterFixed<>>;
    using inc6 = Prefix_Filter<CF12_Flex>;
    using inc8 = Prefix_Filter<TC_shortcut>;
    // using inc9 = Prefix_Filter<Impala512<>>;

    using BBF = SimdBlockFilter<>;
    using BBF_Fixed = SimdBlockFilterFixed<>;

    using L_BF8 = bloomfilter::BloomFilter<uint64_t, 8, 0>;
    using L_BF12 = bloomfilter::BloomFilter<uint64_t, 12, 0>;
    using L_BF16 = bloomfilter::BloomFilter<uint64_t, 16, 0>;

    using CF12_Flex = cuckoofilter::CuckooFilterStable<uint64_t, 12>;
    using CF16_Flex = cuckoofilter::CuckooFilterStable<uint64_t, 16>;

    constexpr size_t number_of_filters = 13;
    // constexpr size_t db_speeder = 1;
    constexpr size_t max_filter_capacity = ((1ULL << 28) * 94 / 100) / testSmart::db_speeder;

    std::stringstream ss_array[number_of_filters];
    std::string names[14] = {"Bloom-8[k=6]", "Bloom-12[k=8]", "Bloom-16[k=11]",
                             "CF-8", "CF-12", "CF-8-Flex", "CF-12-Flex","TC",
                             "PF[BBF_Fixed]", "PF[CF12-Flex]", "PF[TC]", 
                             "BBF", "BBF_Fixed" //"Impala512"
                             };
    for (size_t j = 0; j < number_of_filters; j++) {
        ss_array[j] << names[j] << ", ";
    }

    for (size_t i = 0; i < testSmart::ROUNDS; i++) {
        vector<u64> v_add;
        size_t j = 0;
        testSmart::fill_vec_smart(&v_add, max_filter_capacity);
        ss_array[j++] << testSmart::bench_build_to_file22<L_BF8>(max_filter_capacity, &v_add) << ", ";
        ss_array[j++] << testSmart::bench_build_to_file22<L_BF12>(max_filter_capacity, &v_add) << ", ";
        ss_array[j++] << testSmart::bench_build_to_file22<L_BF16>(max_filter_capacity, &v_add) << ", ";
        ss_array[j++] << testSmart::bench_build_to_file22<CF8>(max_filter_capacity, &v_add) << ", ";
        ss_array[j++] << testSmart::bench_build_to_file22<CF12>(max_filter_capacity, &v_add) << ", ";
        ss_array[j++] << testSmart::bench_build_to_file22<CF8_Flex>(max_filter_capacity, &v_add) << ", ";
        ss_array[j++] << testSmart::bench_build_to_file22<CF12_Flex>(max_filter_capacity, &v_add) << ", ";
        ss_array[j++] << testSmart::bench_build_to_file22<TC_shortcut>(max_filter_capacity, &v_add) << ", ";
        ss_array[j++] << testSmart::bench_build_to_file22<inc4>(max_filter_capacity, &v_add) << ", ";
        ss_array[j++] << testSmart::bench_build_to_file22<inc6>(max_filter_capacity, &v_add) << ", ";
        ss_array[j++] << testSmart::bench_build_to_file22<inc8>(max_filter_capacity, &v_add) << ", ";
        // ss_array[j++] << testSmart::bench_build_to_file22<inc9>(max_filter_capacity, &v_add) << ", ";
        ss_array[j++] << testSmart::bench_build_to_file22<BBF>(max_filter_capacity, &v_add) << ", ";
        ss_array[j++] << testSmart::bench_build_to_file22<BBF_Fixed>(max_filter_capacity, &v_add) << ", ";
        // ss_array[j++] << testSmart::bench_build_to_file22<Impala512<>>(max_filter_capacity, &v_add) << ", ";
    }

    const std::string file_name = "../scripts/build-all.csv";
    std::fstream file(file_name, std::fstream::in | std::fstream::out | std::fstream::app);
    file << endl;
    file << "n = " << max_filter_capacity << std::endl;

    for (size_t j = 0; j < number_of_filters; j++) {
        file << ss_array[j].str() << std::endl;
    }

    file.close();
}
