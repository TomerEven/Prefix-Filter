#include "smart_tests.hpp"
namespace testSmart {

    size_t count_uniques(std::vector<u64> *v) {
        std::sort(v->begin(), v->end());
        auto uniqueCount = std::unique(v->begin(), v->end()) - v->begin();
        return uniqueCount;
    }

    void vector_concatenate_and_shuffle(const std::vector<u64> *v1, const std::vector<u64> *v2, std::vector<u64> *res, std::mt19937_64 rng) {
        size_t all_size = v1->size() + v2->size();
        res->resize(all_size);
        res->insert(res->end(), v1->begin(), v1->end());
        res->insert(res->end(), v2->begin(), v2->end());
        std::shuffle(res->begin(), res->end(), rng);
    }

    void weighted_vector_concatenate_and_shuffle(const std::vector<u64> *v_yes, const std::vector<u64> *v_uni, std::vector<u64> *res, double yes_div, std::mt19937_64 rng) {
        assert(yes_div <= 1);
        auto v_yes_items = std::ceil(v_yes->size() * yes_div);
        size_t all_size = v_yes_items + v_uni->size();
        res->resize(all_size);
        res->insert(res->end(), v_yes->begin(), v_yes->begin() + v_yes_items);
        res->insert(res->end(), v_uni->begin(), v_uni->end());
        std::shuffle(res->begin(), res->end(), rng);
    }

    void vector_concatenate_and_shuffle(const std::vector<u64> *v1, const std::vector<u64> *v2, std::vector<u64> *res) {
        std::uint_least64_t seed;
        sysrandom(&seed, sizeof(seed));
        std::mt19937_64 new_rng(seed);
        vector_concatenate_and_shuffle(v1, v2, res, new_rng);
    }

    size_t test_shuffle_function(const std::vector<u64> *v1, const std::vector<u64> *v2, std::vector<u64> *res) {
        size_t all_size = v1->size() + v2->size();
        res->resize(all_size);
        res->insert(res->end(), v1->begin(), v1->end());
        res->insert(res->end(), v2->begin(), v2->end());
        res->shrink_to_fit();
        std::vector<u64> temp_vec(*res);
        for (size_t i = 0; i < 100; i++) { assert(temp_vec.at(i) == res->at(i)); }

        std::uint_least64_t seed;
        sysrandom(&seed, sizeof(seed));
        std::mt19937_64 new_rng(seed);
        std::shuffle(res->begin(), res->end(), new_rng);
        new_rng();
        // std::shuffle(res->begin(), res->end(), new_rng);new_rng();
        // std::shuffle(res->begin(), res->end(), new_rng);new_rng();
        // std::shuffle(res->begin(), res->end(), new_rng);new_rng();

        size_t counter = 0;
        for (size_t i = 0; i < all_size; i++) { counter += (temp_vec.at(i) == res->at(i)); }

        auto u_count = count_uniques(&temp_vec);
        std::cout << "u_count: \t" << u_count << std::endl;
        auto ratio = (1.0 * u_count) / temp_vec.size();
        std::cout << "ratio:   \t" << ratio << std::endl;

        return counter;
        // assert
    }

    void print_vector(const std::vector<u64> *vec) {
        std::cout << vec->at(0);// << std::endl;
        for (size_t i = 1; i < vec->size(); i++) {
            std::cout << ", " << vec->at(i);// << std::endl;
        }
        std::cout << std::endl;
    }

    void print_vector(const std::vector<u64> *vec, size_t start, size_t end) {
        assert(start < end);
        assert(end < vec->size());
        std::cout << vec->at(start);// << std::endl;
        for (size_t i = start + 1; i < end; i++) {
            std::cout << ", " << vec->at(i);// << std::endl;
        }
        std::cout << std::endl;
    }


    std::mt19937_64 fill_vec_smart(std::vector<u64> *vec, size_t number_of_elements) {

        std::uint_least64_t seed;
        sysrandom(&seed, sizeof(seed));
        //        if (1) {
        //            std::cout << "seed is constant: " << 0xf1234'5678'9abc << std::endl;
        //            seed = 0xf1234'5678'9abc;
        //        }
        std::mt19937_64 rng(seed);
        std::uniform_int_distribution<uint64_t> dist(0, UINT64_MAX);

        vec->resize(number_of_elements);
        for (size_t i = 0; i < number_of_elements; ++i) {
            vec->at(i) = dist(rng);
        }
        return rng;
    }

    void fill_vec_from_range_and_shuffle(size_t start, size_t end, size_t length, const std::vector<u64> *input_vec, std::vector<u64> *temp_vec, std::mt19937_64 rng) {
        assert(start < end);
        assert(start + length <= end);
        temp_vec->resize(end - start);

        for (size_t i = start; i < end; i++) {
            temp_vec->at(i - start) = input_vec->at(i);
        }
        std::shuffle(temp_vec->begin(), temp_vec->end(), rng);
    }

    void my_naive_sample(size_t start, size_t end, size_t length, const std::vector<u64> *input_vec, std::vector<u64> *temp_vec, std::mt19937_64 &rng) {
        std::uniform_int_distribution<> dis(start, end);
        for (size_t i = 0; i < length; i++) {
            u64 temp_item = input_vec->at(dis(rng));
            temp_vec->at(i) = temp_item;
        }
    }

    void fill_vec_by_samples(size_t start, size_t end, size_t length, const std::vector<u64> *input_vec, std::vector<u64> *temp_vec, std::mt19937_64 rng) {
        std::uint_least64_t seed;
        sysrandom(&seed, sizeof(seed));
        std::mt19937_64 new_rng(seed);
        assert(start < end);
        assert(start + length <= end);
        temp_vec->resize(length);
        // auto it_start = input_vec->begin() + start;
        // auto it_end = input_vec->begin() + end;
        // std::sample(it_start, it_end, temp_vec->begin(), length, new_rng);
        my_naive_sample(start, end, length, input_vec, temp_vec, new_rng);
        temp_vec->shrink_to_fit();
        assert(temp_vec->size() == length);
        std::shuffle(temp_vec->begin(), temp_vec->end(), new_rng);
    }

    void fill_vec_by_samples(size_t start, size_t end, size_t length, const std::vector<u64> *input_vec, std::vector<u64> *temp_vec) {
        std::uint_least64_t seed;
        sysrandom(&seed, sizeof(seed));
        std::mt19937_64 new_rng(seed);
        assert(start < end);
        assert(start + length <= end);
        temp_vec->resize(length);
        // auto it_start = input_vec->begin() + start;
        // auto it_end = input_vec->begin() + end;
        // std::sample(it_start, it_end, temp_vec->begin(), length, new_rng);
        my_naive_sample(start, end, length, input_vec, temp_vec, new_rng);

        temp_vec->shrink_to_fit();
        assert(temp_vec->size() == length);
        std::shuffle(temp_vec->begin(), temp_vec->end(), new_rng);
    }

    void print_data(const u64 *data, size_t size, size_t bench_precision, size_t find_step) {
        const unsigned width = 12;
        const unsigned items_in_line = 4;
        const unsigned lines = bench_precision / items_in_line;
        for (size_t i = 0; i < lines; i++) {
            auto temp_sum = 0;
            size_t index = i * items_in_line;
            std::cout << i << ":\t" << std::setw(width) << ((1.0 * find_step) / (1.0 * data[index] / 1e9));
            temp_sum += data[index];
            for (size_t j = 1; j < items_in_line; j++) {
                std::cout << ", " << std::setw(width) << ((1.0 * find_step) / (1.0 * data[index + j] / 1e9));
                temp_sum += data[index + j];
            }
            std::cout << "|Average:\t" << ((1.0 * items_in_line * find_step) / (1.0 * temp_sum / 1e9)) << std::endl;
            // std::cout << std::endl;
        }
    }
}// namespace testSmart
