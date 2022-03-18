#ifndef FILTERS_SMART_TESTS_HPP
#define FILTERS_SMART_TESTS_HPP

#define PREVENT_PIPELINING (1)
#include "../Tests/wrappers.hpp"
// #include "PerfEvent.hpp"
#include <chrono>
#include <fstream>
#include <random>
#include <unistd.h>
#include <vector>

// #include "timing.hpp"
// #include <execution>

typedef std::chrono::nanoseconds ns;

namespace testSmart {
    constexpr size_t db_speeder = 1;
    constexpr size_t ROUNDS = 9;

    size_t count_uniques(std::vector<u64> *v);
    void vector_concatenate_and_shuffle(const std::vector<u64> *v1, const std::vector<u64> *v2, std::vector<u64> *res, std::mt19937_64 rng);

    void vector_concatenate_and_shuffle(const std::vector<u64> *v1, const std::vector<u64> *v2, std::vector<u64> *res);

    void weighted_vector_concatenate_and_shuffle(const std::vector<u64> *v_yes, const std::vector<u64> *v_uni, std::vector<u64> *res, double yes_div, std::mt19937_64 rng);

    size_t test_shuffle_function(const std::vector<u64> *v1, const std::vector<u64> *v2, std::vector<u64> *res);

    void print_data(const u64 *data, size_t size, size_t bench_precision, size_t find_step);


    //Building elements vector

    std::mt19937_64 fill_vec_smart(std::vector<u64> *vec, size_t number_of_elements);

    void my_naive_sample(size_t start, size_t end, size_t length, const std::vector<u64> *input_vec, std::vector<u64> *temp_vec, std::mt19937_64 &rng);

    void fill_vec_by_samples(size_t start, size_t end, size_t length, const std::vector<u64> *input_vec, std::vector<u64> *temp_vec, std::mt19937_64 rng);

    void fill_vec_by_samples(size_t start, size_t end, size_t length, const std::vector<u64> *input_vec, std::vector<u64> *temp_vec);
}// namespace testSmart

namespace testSmart {
    template<typename Table>
    void write_res_to_file_core(const Table *wrap_filter, size_t init_time, size_t filter_max_capacity, size_t lookup_reps, size_t bench_precision, const u64 *data, std::string file_prefix);

    template<typename Table>
    void write_perf_res_to_file(const Table *wrap_filter, size_t init_time, size_t filter_max_capacity, size_t lookup_reps, size_t bench_precision, const u64 *data, std::string file_prefix, std::stringstream &add_ss, std::stringstream &uni_find_ss, std::stringstream &yes_find_ss);

    template<class Table>
    void FPR_test0_after_build(const Table *wrap_filter, const std::vector<u64> *v_add, const std::vector<u64> *v_find, size_t bench_precision);

    template<class Table>
    std::string FPR_parse_data_str_22(const Table *wrap_filter, const std::vector<u64> *v_add, const std::vector<u64> *v_find, size_t yes_res);

    inline size_t sysrandom(void *dst, size_t dstlen) {
        char *buffer = reinterpret_cast<char *>(dst);
        std::ifstream stream("/dev/urandom", std::ios_base::binary | std::ios_base::in);
        stream.read(buffer, dstlen);

        return dstlen;
    }

    template<class Table>
    __attribute__((noinline)) auto time_lookups(const Table *wrap_filter, const std::vector<u64> *element_set, size_t start, size_t end) -> ulong {
        static volatile bool dummy;
        bool x = 0;

        auto t0 = std::chrono::high_resolution_clock::now();
        for (size_t i = start; i < end; ++i) {
            x ^= FilterAPI<Table>::Contain(element_set->at(i), wrap_filter);
            // #if PREVENT_PIPELINING
            // asm volatile(
            //         "rdtscp\n\t"
            //         "lfence"
            //         :
            //         :
            //         : "rax", "rcx", "rdx", "memory");
            // #endif
        }
        auto t1 = std::chrono::high_resolution_clock::now();

        dummy = x;
        return std::chrono::duration_cast<ns>(t1 - t0).count();
    }

    template<class Table>
    __attribute__((noinline)) auto time_insertions(Table *wrap_filter, const std::vector<u64> *element_set, size_t start, size_t end) -> ulong {
        auto t0 = std::chrono::steady_clock::now();
        for (size_t i = start; i < end; ++i) {
            FilterAPI<Table>::Add(element_set->at(i), wrap_filter);
            // #if PREVENT_PIPELINING
            // asm volatile(
            //         "rdtscp\n\t"
            //         "lfence"
            //         :
            //         :
            //         : "rax", "rcx", "rdx", "memory");
            // #endif
        }
        // FilterAPI<Table>::AddAll(*element_set, start, end, wrap_filter);
        auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<ns>(t1 - t0).count();
    }

    /*template<class Table>
    __attribute__((noinline)) auto time_lookups_perf(const Table *wrap_filter, const std::vector<u64> *element_set, size_t start, size_t end, std::stringstream &ss, bool with_header) -> ulong {
        static volatile bool dummy;
        bool x = 0;

        PerfEvent e;
        asm volatile(
                "rdtscp\n\t"
                "lfence"
                :
                :
                : "rax", "rcx", "rdx", "memory");
        e.startCounters();
        // auto t0 = std::chrono::high_resolution_clock::now();
        for (size_t i = start; i < end; ++i) {
            x ^= FilterAPI<Table>::Contain(element_set->at(i), wrap_filter);
            // #if PREVENT_PIPELINING
            // asm volatile(
            //         "rdtscp\n\t"
            //         "lfence"
            //         :
            //         :
            //         : "rax", "rcx", "rdx", "memory");
            // #endif
        }
        // auto t1 = std::chrono::high_resolution_clock::now();
        dummy = x;
        e.stopCounters();
        asm volatile(
                "rdtscp\n\t"
                "lfence"
                :
                :
                : "rax", "rcx", "rdx", "memory");
        if (with_header) {
            e.printReport(ss, end - start);
        } else {
            e.printReport_NoHeader(ss, end - start);
        }
        // return std::chrono::duration_cast<ns>(t1 - t0).count();
        return e.get_time();
    }

     template<class Table>
    __attribute__((noinline)) auto time_insertions_perf(Table *wrap_filter, const std::vector<u64> *element_set, size_t start, size_t end, std::stringstream &ss) -> ulong {
        PerfEvent e;
        e.startCounters();
        // auto t0 = std::chrono::high_resolution_clock::now();
        for (size_t i = start; i < end; ++i) {
            FilterAPI<Table>::Add(element_set->at(i), wrap_filter);
            // #if PREVENT_PIPELINING
            // asm volatile(
            //         "rdtscp\n\t"
            //         "lfence"
            //         :
            //         :
            //         : "rax", "rcx", "rdx", "memory");
            // #endif
        }
        // FilterAPI<Table>::AddAll(*element_set, start, end, wrap_filter);
        // auto t1 = std::chrono::high_resolution_clock::now();
        e.stopCounters();
        asm volatile(
                "rdtscp\n\t"
                "lfence"
                :
                :
                : "rax", "rcx", "rdx", "memory");
        if (start == 0) {
            e.printReport(ss, end - start);
        } else {
            e.printReport_NoHeader(ss, end - start);
        }
        // return std::chrono::duration_cast<ns>(t1 - t0).count();
        return e.get_time();
        // return std::chrono::duration_cast<ns>(t1 - t0).count();
    }
 */
    template<class Table>
    __attribute__((noinline)) auto time_deletions(Table *wrap_filter, const std::vector<u64> *element_set, size_t start, size_t end) -> ulong {
        if (!(FilterAPI<Table>::get_functionality(wrap_filter) & 4)) {
            //FIXME: UNCOMMENT!
            std::cout << FilterAPI<Table>::get_name(wrap_filter) << " does not support deletions." << std::endl;
            return 0;
        }

        auto t0 = std::chrono::steady_clock::now();
        for (size_t i = start; i < end; ++i) {
            FilterAPI<Table>::Remove(element_set->at(i), wrap_filter);
        }
        auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<ns>(t1 - t0).count();
    }

    template<class Table>
    void benchmark_single_round_np_incremental(Table *wrap_filter, const std::vector<u64> *add_vec, const std::vector<u64> *find_vec, size_t round_counter, size_t benchmark_precision, u64 *data, bool to_print, std::stringstream &add_ss, std::stringstream &uni_find_ss, std::stringstream &yes_find_ss) {
        /* switch between the commented lines with "time_time_insertions" and "time_insertions_perf" to also get a csv file with data on the performances counters.*/
        const size_t find_step = find_vec->size() / benchmark_precision;
        size_t add_step = add_vec->size() / benchmark_precision;
        const size_t true_find_step = add_step;
        size_t add_start = round_counter * add_step;
        

        auto insertion_time = time_insertions(wrap_filter, add_vec, add_start, add_start + add_step);
        // auto insertion_time = time_insertions_perf(wrap_filter, add_vec, add_start, add_start + add_step, add_ss);
        asm volatile(
                "rdtscp\n\t"
                "lfence"
                :
                :
                : "rax", "rcx", "rdx", "memory");
        auto find_start = find_step * round_counter;
        auto uniform_lookup_time = time_lookups(wrap_filter, find_vec, find_start, find_start + find_step);
        // auto uniform_lookup_time = time_lookups_perf(wrap_filter, find_vec, find_start, find_start + find_step, uni_find_ss, round_counter == 0);
        asm volatile(
                "rdtscp\n\t"
                "lfence"
                :
                :
                : "rax", "rcx", "rdx", "memory");
        // auto true_lookup_time = time_lookups(wrap_filter, add_vec, del_start, del_start + add_step);
        std::vector<u64> temp_vec;
        fill_vec_by_samples(0, add_start + add_step, true_find_step, add_vec, &temp_vec);
        asm volatile(
                "rdtscp\n\t"
                "lfence"
                :
                :
                : "rax", "rcx", "rdx", "memory");
        auto true_lookup_time = time_lookups(wrap_filter, &temp_vec, 0, true_find_step);
        // auto true_lookup_time = time_lookups_perf(wrap_filter, &temp_vec, 0, true_find_step, yes_find_ss, round_counter == 0);
        asm volatile(
                "rdtscp\n\t"
                "lfence"
                :
                :
                : "rax", "rcx", "rdx", "memory");

        size_t index = 4 * (round_counter);
        data[index + 0] = insertion_time;
        data[index + 1] = uniform_lookup_time;
        data[index + 2] = true_lookup_time;
        data[index + 3] = 0;

        if (to_print) {
            constexpr size_t width = 12;
            std::cout << round_counter << ": \t";
            std::cout << std::setw(width) << std::left << ((1.0 * add_step) / (1.0 * data[index + 0] / 1e9)) << ", ";
            std::cout << std::setw(width) << std::left << ((1.0 * find_step) / (1.0 * data[index + 1] / 1e9)) << ", ";
            std::cout << std::setw(width) << std::left << ((1.0 * add_step) / (1.0 * data[index + 2] / 1e9)) << ", ";
            std::cout << std::setw(width) << std::left << ((1.0 * add_step) / (1.0 * data[index + 3] / 1e9)) << std::endl;
        }
    }

    template<typename Table>
    void Bench_res_to_file_incremental_22(size_t filter_max_capacity, size_t bench_precision, const std::vector<u64> *add_vec, const std::vector<u64> *lookup_vec, std::string file_prefix, bool to_print = false) {
        const size_t data_size = (bench_precision + 2) * 4;
        assert(data_size < 1024);
        u64 data[data_size];
        std::fill(data, data + data_size, 0);
        // std::cout << "H0!"  << std::endl;
        auto t0 = std::chrono::high_resolution_clock::now();
        Table filter = FilterAPI<Table>::ConstructFromAddCount(filter_max_capacity);
        auto t1 = std::chrono::high_resolution_clock::now();
        // std::cout << "Here!"  << std::endl;
        auto init_time = std::chrono::duration_cast<ns>(t1 - t0).count();
        Table *wrap_filter = &filter;

        std::string filter_name = FilterAPI<Table>::get_name(wrap_filter);
        if (to_print)
            std::cout << "filter_name: " << filter_name << std::endl;
        // benchmark_round0_np(wrap_filter, add_vec, lookup_vec, bench_precision, rng, data, to_print);

        if (to_print) std::cout << std::string(80, '=') << std::endl;
        std::stringstream add_ss, uni_find_ss, yes_find_ss;
        for (size_t round = 0; round < bench_precision; ++round) {
            benchmark_single_round_np_incremental(wrap_filter, add_vec, lookup_vec, round, bench_precision, data, to_print, add_ss, uni_find_ss, yes_find_ss);
            // if (to_print) std::cout << std::string(80, '=') << std::endl;
            asm volatile(
                    "rdtscp\n\t"
                    "lfence"
                    :
                    :
                    : "rax", "rcx", "rdx", "memory");
        }
        if (to_print) std::cout << std::string(80, '=') << std::endl;

        //UNCOMMENT me to get CSV files.
        // write_perf_res_to_file(wrap_filter, init_time, filter_max_capacity, lookup_vec->size(), bench_precision, data, file_prefix, add_ss, uni_find_ss, yes_find_ss);
        write_res_to_file_core(wrap_filter, init_time, filter_max_capacity, lookup_vec->size(), bench_precision, data, file_prefix);

        if (to_print)
            FPR_test0_after_build(wrap_filter, add_vec, lookup_vec, bench_precision);
    }

    template<typename Table>
    void write_perf_res_to_file(const Table *wrap_filter, size_t init_time, size_t filter_max_capacity, size_t lookup_reps, size_t bench_precision, const u64 *data, std::string file_prefix, std::stringstream &add_ss, std::stringstream &uni_find_ss, std::stringstream &yes_find_ss) {
        std::string filter_name = FilterAPI<Table>::get_name(wrap_filter);
        std::string file_name = file_prefix + filter_name + ".csv";
        std::cout << "file_name: " << file_name << std::endl;
        std::fstream file(file_name, std::fstream::in | std::fstream::out | std::fstream::app);
        // file << std::endl;
        // std::fstream file(file_name, std::fstream::in | std::fstream::out | std::fstream::trunc);
        // std::string str_add = add_ss.str();
        // std::cout << std::string(80, '@') << std::endl;
        // std::cout << str_add << std::endl;
        // std::cout << std::string(80, '@') << std::endl;
        file << "add, uni_find, yes_find" << std::endl;
        for (size_t i = 0; i < bench_precision + 1; i++) {
            std::string temp[3];
            std::getline(add_ss, temp[0]);
            std::getline(uni_find_ss, temp[1]);
            std::getline(yes_find_ss, temp[2]);
            file << temp[0] << ",";
            file << temp[1] << ",";
            file << temp[2] << std::endl;
        }
        file.close();
    }

    template<typename Table>
    void write_res_to_file_core(const Table *wrap_filter, size_t init_time, size_t filter_max_capacity, size_t lookup_reps, size_t bench_precision, const u64 *data, std::string file_prefix) {
        std::string filter_name = FilterAPI<Table>::get_name(wrap_filter);
        std::string file_name = file_prefix + filter_name;
        std::cout << "file_name: " << file_name << std::endl;
        std::fstream file(file_name, std::fstream::in | std::fstream::out | std::fstream::app);
        file << endl;
        // std::fstream file(file_name, std::fstream::in | std::fstream::out | std::fstream::trunc);
        file << "# This is a comment." << std::endl;
        file << "NAME\t" << filter_name << std::endl;
        file << "INIT_TIME(NANO_SECOND)\t" << init_time << std::endl;
        file << "FILTER_MAX_CAPACITY\t" << filter_max_capacity << std::endl;
        file << "BYTE_SIZE\t" << FilterAPI<Table>::get_byte_size(wrap_filter) << std::endl;
        file << "NUMBER_OF_LOOKUP\t" << lookup_reps << std::endl;
        file << std::endl;
        file << "# add, uniform lookup, true_lookup, deletions. Each columns unit is in nano second." << std::endl;
        file << std::endl;
        file << "BENCH_START" << std::endl;
        for (size_t i = 0; i < bench_precision; i++) {
            size_t index = i * 4;
            file << data[index];
            for (size_t j = 1; j < 4; j++) {
                file << ", " << data[index + j];
            }
            file << std::endl;
        }
        file << std::endl;
        file << "BENCH_END" << std::endl;
        file << "END_OF_FILE!" << std::endl;
        file.close();
    }

    template<typename Table>
    void write_build_res_to_file(const Table *wrap_filter, size_t init_time, size_t built_time, size_t filter_max_capacity, std::string file_prefix) {
        std::string filter_name = FilterAPI<Table>::get_name(wrap_filter);
        std::string file_name = file_prefix + filter_name;
        std::cout << "file_name: " << file_name << std::endl;
        std::fstream file(file_name, std::fstream::in | std::fstream::out | std::fstream::app);
        file << endl;
        // std::fstream file(file_name, std::fstream::in | std::fstream::out | std::fstream::trunc);
        file << "# This is a comment." << std::endl;
        file << "# Results for build." << std::endl;
        file << "NAME\t" << filter_name << std::endl;
        file << "INIT_TIME(NANO_SECOND)\t" << init_time << std::endl;
        file << "BUILT_TIME(NANO_SECOND)\t" << built_time << std::endl;
        file << "FILTER_MAX_CAPACITY(Actually-number-of-items-in-the-filter)\t" << filter_max_capacity << std::endl;
        file << "BYTE_SIZE\t" << FilterAPI<Table>::get_byte_size(wrap_filter) << std::endl;
        file << std::endl;
        file << "END_OF_FILE!" << std::endl;
        file.close();
    }


    template<typename Table>
    void bench_build_to_file(size_t filter_max_capacity, const std::vector<u64> *v_add, std::string file_prefix) {
        auto t0 = std::chrono::high_resolution_clock::now();
        Table filter = FilterAPI<Table>::ConstructFromAddCount(filter_max_capacity);
        auto t1 = std::chrono::high_resolution_clock::now();
        auto init_time = std::chrono::duration_cast<ns>(t1 - t0).count();
        Table *wrap_filter = &filter;
        std::string filter_name = FilterAPI<Table>::get_name(wrap_filter);
        auto is_static = (FilterAPI<Table>::get_functionality(wrap_filter) == 1);
        auto built_time = time_insertions(wrap_filter, v_add, 0, v_add->size());
        write_build_res_to_file(wrap_filter, init_time, built_time, filter_max_capacity, file_prefix);
    }

    template<typename Table>
    size_t bench_build_to_file22(size_t filter_max_capacity, const std::vector<u64> *v_add) {
        auto t0 = std::chrono::high_resolution_clock::now();
        Table filter = FilterAPI<Table>::ConstructFromAddCount(filter_max_capacity);
        auto t1 = std::chrono::high_resolution_clock::now();
        auto init_time = std::chrono::duration_cast<ns>(t1 - t0).count();
        Table *wrap_filter = &filter;
        // std::string filter_name = FilterAPI<Table>::get_name(wrap_filter);
        // auto is_static = (FilterAPI<Table>::get_functionality(wrap_filter) == 1);
        auto built_time = time_insertions(wrap_filter, v_add, 0, v_add->size());
        return built_time;
    }
    //////////////////////////////////////////////////////////////// ////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////// ////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////// ////////////////////////////////////////////////////////////////

    /* template<class Table, typename ItemType>
size_t count_finds(Table *wrap_filter, const std::vector<ItemType> *vec) {
    size_t counter = 0;
    for (size_t i = 0; i < vec->size(); i++) { counter += FilterAPI<Table>::Contain(vec->at(i), wrap_filter);
    }
    return counter;
} */

    template<class Table>
    size_t count_finds(const Table *wrap_filter, const std::vector<u64> *vec) {
        size_t counter = 0;
        for (size_t i = 0; i < vec->size(); i++) { counter += FilterAPI<Table>::Contain(vec->at(i), wrap_filter); }
        return counter;
    }

    /**
 * @brief Get the FPR (false positive probability) test0 object.
 * 
 * @tparam Table 
 * @param wrap_filter 
 * @param yes_vec 
 * @return std::tuple<size_t, size_t> 
 */
    template<class Table>
    size_t get_FPR_test0(Table *wrap_filter, const std::vector<u64> *v_add, const std::vector<u64> *v_find) {
        /**Insertion*/
        for (auto el : *v_add) {
            FilterAPI<Table>::Add(el, wrap_filter);
        }
        size_t counter = 0;
        for (auto el : *v_add) {
            counter++;
            if (!FilterAPI<Table>::Contain(el, wrap_filter)) {
                auto name = FilterAPI<Table>::get_name(wrap_filter);
                std::cout << "Filter (" << name << ") has2 a false negative. exiting." << std::endl;
                std::cout << "counter: \t" << counter << std::endl;
                FilterAPI<Table>::Contain(el, wrap_filter);
                assert(0);
                exit(-42);
            }
        }

        size_t yes_res = count_finds(wrap_filter, v_find);
        return yes_res;
    }

    template<class Table>
    void FPR_test0_after_build(const Table *wrap_filter, const std::vector<u64> *v_add, const std::vector<u64> *v_find, size_t bench_precision) {
        // Table filter = FilterAPI<Table>::ConstructFromAddCount(v_add->size());
        // Table *wrap_filter = &filter;
        size_t counter = 0;
        const size_t lim = v_add->size() / bench_precision * bench_precision;
        for (auto el : *v_add) {
            counter++;
            if (counter >= lim)
                break;
            if (!FilterAPI<Table>::Contain(el, wrap_filter)) {
                auto name = FilterAPI<Table>::get_name(wrap_filter);
                std::cout << "Filter (" << name << ") has3 a false negative. exiting." << std::endl;
                std::cout << "counter: \t" << counter << std::endl;
                FilterAPI<Table>::Contain(el, wrap_filter);
                assert(0);
                exit(-42);
            }
        }

        size_t yes_res = count_finds(wrap_filter, v_find);
        // auto yes_res = get_FPR_test0(wrap_filter, v_add, v_find);

        auto s = FPR_parse_data_str_22(wrap_filter, v_add, v_find, yes_res);
        std::cout << "s:\n"
                  << s << std::endl;
        // FPR_printer(wrap_filter, v_add, v_find, yes_res);
    }

    template<class Table>
    std::string FPR_parse_data_str_22(const Table *wrap_filter, const std::vector<u64> *v_add, const std::vector<u64> *v_find, size_t yes_res) {
        size_t true_counter = yes_res;
        // size_t true_counter = yes_res;
        assert(true_counter <= v_find->size());
        size_t false_counter = v_find->size() - yes_res;
        const size_t filter_max_capacity = v_add->size();
        const size_t filter_true_cap = FilterAPI<Table>::get_cap(wrap_filter);
        auto filter_id = FilterAPI<Table>::get_ID(wrap_filter);
        bool skip = (filter_id == BBF) or (filter_id == SIMD_fixed);
        if ((!skip) and (filter_max_capacity != filter_true_cap)) {
            std::cout << "filter_name:     \t" << FilterAPI<Table>::get_name(wrap_filter) << std::endl;
            std::cout << "filter_max_cap:  \t" << filter_max_capacity << std::endl;
            std::cout << "filter_true_cap: \t" << filter_true_cap << std::endl;
        }
        std::string filter_name = FilterAPI<Table>::get_name(wrap_filter);
        const size_t filter_byte_size = FilterAPI<Table>::get_byte_size(wrap_filter);
        double positive_ratio = 1.0 * true_counter / (false_counter + true_counter);
        double bpi = filter_byte_size * 8.0 / filter_max_capacity;
        double optimal_bits_for_err = -log2(positive_ratio);
        double bpi_diff = (bpi - optimal_bits_for_err);
        double bpi_ratio = (bpi / optimal_bits_for_err);
        double fpp_mult_factor = positive_ratio / (1.0 / 256);
        double data[] = {positive_ratio, bpi, optimal_bits_for_err, bpi_diff, bpi_ratio, fpp_mult_factor};
        std::stringstream ss;
        ss << std::setw(34) << std::left << filter_name << "\t, " << std::setw(12) << std::left << filter_byte_size;
        for (auto &&x : data) { ss << ", " << std::setw(12) << std::left << x; }
        ss << std::endl;
        std::string line = ss.str();
        return line;
    }

    template<class Table>
    std::string FPR_test_as_str(Table *wrap_filter, const std::vector<u64> *v_add, const std::vector<u64> *v_find) {
        auto yes_res = get_FPR_test0(wrap_filter, v_add, v_find);
        size_t true_counter = yes_res;
        assert(true_counter <= v_find->size());
        size_t false_counter = v_find->size() - yes_res;
        const size_t filter_max_capacity = v_add->size();
        const size_t filter_true_cap = FilterAPI<Table>::get_cap(wrap_filter);
        auto filter_id = FilterAPI<Table>::get_ID(wrap_filter);
        bool skip = (filter_id == BBF) or (filter_id == SIMD_fixed);
        if ((!skip) and (filter_max_capacity != filter_true_cap)) {
            // UNCOMMENT ME!!! FIXME!!!
            std::cout << "filter_name:     \t" << FilterAPI<Table>::get_name(wrap_filter) << std::endl;
            std::cout << "filter_max_cap:  \t" << filter_max_capacity << std::endl;
            std::cout << "filter_true_cap: \t" << filter_true_cap << std::endl;
        }
        std::string filter_name = FilterAPI<Table>::get_name(wrap_filter);
        const size_t filter_byte_size = FilterAPI<Table>::get_byte_size(wrap_filter);
        double positive_ratio = 1.0 * true_counter / (false_counter + true_counter);
        double bpi = filter_byte_size * 8.0 / filter_max_capacity;
        double optimal_bits_for_err = -log2(positive_ratio);
        double bpi_diff = (bpi - optimal_bits_for_err);
        double bpi_ratio = (bpi / optimal_bits_for_err);
        // double fpp_mult_factor = positive_ratio / (1.0 / 256);
        double data[] = {positive_ratio, bpi, optimal_bits_for_err, bpi_diff, bpi_ratio};
        std::stringstream ss;
        ss << std::setw(34) << std::left << filter_name << "\t, " << std::setw(12) << std::left << filter_byte_size;
        for (auto &&x : data) { ss << ", " << std::setw(12) << std::left << x; }
        ss << std::endl;
        std::string line = ss.str();
        return line;
        // std::cout << line;// << std::endl;
    }


    template<class Table>
    void FPR_test(const std::vector<u64> *v_add, const std::vector<u64> *v_find, std::string path, bool create_file = false) {


        const size_t filter_max_capacity = v_add->size();
        Table filter = FilterAPI<Table>::ConstructFromAddCount(v_add->size());
        Table *wrap_filter = &filter;
        auto line = FPR_test_as_str(wrap_filter, v_add, v_find);

        std::fstream file(path, std::fstream::in | std::fstream::out | std::fstream::app);
        file << line;
        file.close();

        // std::cout << line << std::endl;
    }

    template<class Table>
    void profile_benchmark(Table *wrap_filter, const std::vector<const std::vector<u64> *> *elements) {
        auto add_vec = elements->at(0);
        auto find_vec = elements->at(1);
        auto delete_vec = elements->at(2);

        auto insertion_time = time_insertions(wrap_filter, add_vec, 0, add_vec->size());

        printf("insertions done\n");
        fflush(stdout);
        ulong uniform_lookup_time = 0;
        ulong true_lookup_time = 0;
        // size_t true_lookup_time = 0;
        char buf[1024];
        // sprintf(buf, "perf record -p %d &", getpid());
        // sprintf(buf, "perf stat -p %d -e cycles -e instructions -e cache-misses -e cache-references -e L1-dcache-load-misses -e L1-dcache-loads -e LLC-load-misses -e LLC-loads -e dTLB-load-misses -e dTLB-loads -e node-load-misses -e node-loads -e branches -e branch-misses &", getpid());
        sprintf(buf, "perf stat -p %d \
    -e cycles                   \
    -e instructions             \
    -e cache-misses             \
    -e cache-references         \
    -e L1-dcache-load-misses    \
    -e L1-dcache-loads          \
    -e LLC-load-misses          \
    -e LLC-loads                \
    -e dTLB-load-misses         \
    -e dTLB-loads               \
    -e node-load-misses         \
    -e node-loads               \
    -e alignment-faults         \
    -e branches                 \
    -e branch-misses            \
    -e branch-loads             \
    -e branch-loads-misses      \
    &",
                getpid());
        // sprintf(buf, "perf stat -p %d -e cycles -e instructions -e cache-misses -e cache-references -e L1-dcache-load-misses -e L1-dcache-loads -e LLC-load-misses -e LLC-loads -e dTLB-load-misses -e dTLB-loads -e node-load-misses -e node-loads -e branches -e branch-misses -e uops_executed.stall_cycles &", getpid());
        auto junk = system(buf);
        for (int i = 0; i < 16; i++) {
            // true_lookup_time = time_lookups(wrap_filter, add_vec, 0, add_step);
            uniform_lookup_time += time_lookups(wrap_filter, find_vec, 0, find_vec->size());
            // true_lookup_time += time_lookups(wrap_filter, add_vec, 0, add_vec->size());
            // uniform_lookup_time += time_lookups(wrap_filter, find_vec, 0, find_vec->size());
        }
        // printf("%zd\n", 500 * add_step);
        printf("%zd\n", 16 * find_vec->size());
        // printf("%zd\n", 16 * add_vec->size());
        // printf("%zd\n", 8 * find_vec->size() + 8 * add_vec->size() );
        // printf("%zd\n", 500 * true_find_step);
        exit(0);
    }

    inline std::string get_int_with_commas(uint64_t x) {
        auto s = std::to_string(x);

        if (s.size() <= 3)
            return s;


        std::string res;
        std::string prefix = s;
        // const size_t s_size = s.size();
        while (prefix.size() > 3) {
            size_t index = prefix.size() - 3;
            std::string temp = prefix.substr(index, 3);
            res = "," + temp + res;

            prefix = prefix.substr(0, index);
        }
        res = prefix + res;
        return res;
    }

    template<class Table>
    void profile_benchmark_cache(Table *wrap_filter, const std::vector<const std::vector<u64> *> *elements) {
        auto add_vec = elements->at(0);
        auto find_vec = elements->at(1);
        // auto delete_vec = elements->at(2);

        auto insertion_time = time_insertions(wrap_filter, add_vec, 0, add_vec->size());

        printf("insertions done\n");
        fflush(stdout);
        ulong uniform_lookup_time = 0;
        // ulong true_lookup_time = 0;
        // size_t true_lookup_time = 0;
        char buf[1024];


        // sprintf(buf, "perf stat -p %d
        sprintf(buf, "perf stat -p %d \
        -e cpu/event=0x2e,umask=0x41,name=LONGEST_LAT_CACHE.MISS/ \
                             & ",
                getpid());

        /* -e cache-misses           \
        -e cache-references       \
        -e L1-dcache-load-misses  \
        -e L1-dcache-loads        \
        -e LLC-load-misses        \
        -e LLC-loads              \
        -e dTLB-load-misses       \
        -e dTLB-loads             \
        -e node-load-misses       \
        -e node-loads             \
        -e branches               \
        -e branch-misses          \
         */
        auto junk = system(buf);
        constexpr size_t reps = 16;
        for (size_t i = 0; i < reps; i++) {
            uniform_lookup_time += time_lookups(wrap_filter, find_vec, 0, find_vec->size());
        }
        // printf("%zd\n", 16 * find_vec->size());
        std::cout << "Number of lookups: \t" << get_int_with_commas(reps * find_vec->size()) << std::endl;
        printf("Number of lookups: %zd\n", reps * find_vec->size());
        exit(0);
    }
}// namespace testSmart

#endif// FILTERS_CON_TESTS_HPP
