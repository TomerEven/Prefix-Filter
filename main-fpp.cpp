

#include "Tests/smart_tests.hpp"

void write_fpp_to_file();


int main() {
    write_fpp_to_file();
    return 0;
}

void write_fpp_to_file() {
    using CF8 = cuckoofilter::CuckooFilter<uint64_t, 8>;
    using CF12 = cuckoofilter::CuckooFilter<uint64_t, 12>;
    using CF16 = cuckoofilter::CuckooFilter<uint64_t, 16>;
    using CF32 = cuckoofilter::CuckooFilter<uint64_t, 32>;
    using CF8_N2 = cuckoofilter::CuckooFilterStable<uint64_t, 8>;
    using CF12_Flex = cuckoofilter::CuckooFilterStable<uint64_t, 12>;
    using CF16_N2 = cuckoofilter::CuckooFilterStable<uint64_t, 16>;
    using CF32_N2 = cuckoofilter::CuckooFilterStable<uint64_t, 32>;

    using inc4 = Prefix_Filter<SimdBlockFilterFixed<>>;
    using inc6 = Prefix_Filter<CF12_Flex>;
    using inc8 = Prefix_Filter<TC_shortcut>;
    using inc9 = Prefix_Filter<Impala512<>>;
    
    using L_BF8 = bloomfilter::BloomFilter<uint64_t, 8, 0>;
    using L_BF12 = bloomfilter::BloomFilter<uint64_t, 12, 0>;
    using L_BF16 = bloomfilter::BloomFilter<uint64_t, 16, 0>;

    constexpr size_t fp_capacity = ((1ULL << 28) * 94 / 100) / testSmart::db_speeder;    
    constexpr size_t fp_lookups = fp_capacity;

    std::vector<u64> fp_v_add, fp_v_find;
    testSmart::fill_vec_smart(&fp_v_add, fp_capacity);
    testSmart::fill_vec_smart(&fp_v_find, fp_lookups);


    std::string path = "../scripts/fpp_table.csv";

    std::fstream file(path, std::fstream::in | std::fstream::out | std::fstream::app);
    file << "n =, " << fp_capacity << ", Lookups =, " << fp_lookups << std::endl;
    std::string header = "Filter, Size in bytes, Ratio of yes-queries bits per item (average), optimal bits per item (w.r.t. yes-queries), difference of BPI to optimal BPI, ratio of BPI to optimal BPI";
    // file << "name, byte size, FPR, BPI, opt-BPI, bpi-additive-difference, bpi-ratio" << std::endl;
    file << header << std::endl;
    file.close();
    
    testSmart::FPR_test<CF8>(&fp_v_add, &fp_v_find, path);
    testSmart::FPR_test<CF12>(&fp_v_add, &fp_v_find, path);
    testSmart::FPR_test<CF16>(&fp_v_add, &fp_v_find, path);

    testSmart::FPR_test<CF8_N2>(&fp_v_add, &fp_v_find, path);
    testSmart::FPR_test<CF12_Flex>(&fp_v_add, &fp_v_find, path);
    testSmart::FPR_test<CF16_N2>(&fp_v_add, &fp_v_find, path);

    testSmart::FPR_test<inc4>(&fp_v_add, &fp_v_find, path);
    testSmart::FPR_test<inc6>(&fp_v_add, &fp_v_find, path);
    testSmart::FPR_test<inc8>(&fp_v_add, &fp_v_find, path);
    testSmart::FPR_test<inc9>(&fp_v_add, &fp_v_find, path);

    testSmart::FPR_test<SimdBlockFilterFixed<>>(&fp_v_add, &fp_v_find, path);
    testSmart::FPR_test<SimdBlockFilter<>>(&fp_v_add, &fp_v_find, path);
    testSmart::FPR_test<Impala512<>>(&fp_v_add, &fp_v_find, path);
    
    testSmart::FPR_test<L_BF8>(&fp_v_add, &fp_v_find, path);
    testSmart::FPR_test<L_BF12>(&fp_v_add, &fp_v_find, path);
    testSmart::FPR_test<L_BF16>(&fp_v_add, &fp_v_find, path);
    
    testSmart::FPR_test<TC_shortcut>(&fp_v_add, &fp_v_find, path);
}
