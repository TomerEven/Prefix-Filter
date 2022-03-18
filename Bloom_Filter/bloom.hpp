/* Taken from https://github.com/FastFilter/fastfilter_cpp */
#ifndef BLOOM_FILTER_BLOOM_FILTER_H_
#define BLOOM_FILTER_BLOOM_FILTER_H_

#include <algorithm>
#include <assert.h>
#include <sstream>

#include "../hashutil.h"

using namespace std;
using namespace hashing;

namespace bloomfilter {
    // status returned by a Bloom filter operation
    enum Status {
        Ok = 0,
        NotFound = 1,
        NotEnoughSpace = 2,
        NotSupported = 3,
    };

    inline uint32_t reduce(uint32_t hash, uint32_t n) {
        // http://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/
        return (uint32_t) (((uint64_t) hash * n) >> 32);
    }

    /**
* Given a value "word", produces an integer in [0,p) without division.
* The function is as fair as possible in the sense that if you iterate
* through all possible values of "word", then you will generate all
* possible outputs as uniformly as possible.
*/
    static inline uint32_t fastrange32(uint32_t word, uint32_t p) {
        // http://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/
        return (uint32_t) (((uint64_t) word * (uint64_t) p) >> 32);
    }


    /**
* Given a value "word", produces an integer in [0,p) without division.
* The function is as fair as possible in the sense that if you iterate
* through all possible values of "word", then you will generate all
* possible outputs as uniformly as possible.
*/
    static inline uint64_t fastrange64(uint64_t word, uint64_t p) {
        // http://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/
        // #ifdef __SIZEOF_INT128__// then we know we have a 128-bit int
        return (uint64_t) (((__uint128_t) word * (__uint128_t) p) >> 64);
    }


#ifndef UINT32_MAX
#define UINT32_MAX (0xffffffff)
#endif// UINT32_MAX

    /**
* Given a value "word", produces an integer in [0,p) without division.
* The function is as fair as possible in the sense that if you iterate
* through all possible values of "word", then you will generate all
* possible outputs as uniformly as possible.
*/
    static inline size_t fastrangesize(uint64_t word, size_t p) {
#if (SIZE_MAX == UINT32_MAX)
        return (size_t) fastrange32(word, p);
#else // assume 64-bit
        return (size_t) fastrange64(word, p);
#endif// SIZE_MAX == UINT32_MAX
    }

    static inline size_t getBestK(size_t bitsPerItem) {
        return max(1, (int) round((double) bitsPerItem * log(2)));
    }

    inline uint64_t getBit(uint32_t index) { return 1L << (index & 63); }

    template<typename ItemType, size_t bits_per_item, bool branchless,
             typename HashFamily = TwoIndependentMultiplyShift,
             int k = (int) ((double) bits_per_item * 0.693147180559945 + 0.5)>
    class BloomFilter {
    public:
        uint64_t *data;
        size_t size;
        size_t arrayLength;
        size_t bitCount;
        int kk;
        HashFamily hasher;

        double BitsPerItem() const { return k; }

        explicit BloomFilter(const size_t n) : hasher() {
            this->size = 0;
            this->kk = getBestK(bits_per_item);
            this->bitCount = n * bits_per_item;
            this->arrayLength = (bitCount + 63) / 64;
            data = new uint64_t[arrayLength];
            std::fill_n(data, arrayLength, 0);
        }

        ~BloomFilter() { delete[] data; }

        // Add an item to the filter.
        Status Add(const ItemType &item);

        // Add multiple items to the filter.
        Status AddAll(const vector<ItemType> &data, const size_t start,
                      const size_t end) {
            return AddAll(data.data(), start, end);
        }
        Status AddAll(const ItemType *data, const size_t start,
                      const size_t end);
        // Report if the item is inserted, with false positive rate.
        Status Contain(const ItemType &item) const;

        /* methods for providing stats  */
        // summary infomation
        std::string Info() const;

        // number of current inserted items;
        size_t Size() const { return size; }

        // size of the filter in bytes.
        size_t SizeInBytes() const { return arrayLength * 8; }

        std::string get_name() const {
            size_t BPI = bits_per_item;
            size_t hash_k = k;
            std::string name = "Lem-BF-" + std::to_string(BPI) + "[k=" + std::to_string(hash_k) + "]";
            return name;
            // if (branchless) {
            //     return name + "-{branchless}";
            // } else {
            //     return name + "-{with-branch}";
            // }
        }
    };

    template<typename ItemType, size_t bits_per_item, bool branchless,
             typename HashFamily, int k>
    Status BloomFilter<ItemType, bits_per_item, branchless, HashFamily, k>::Add(
            const ItemType &key) {
        uint64_t hash = hasher(key);
        uint64_t a = (hash >> 32) | (hash << 32);
        uint64_t b = hash;
        for (int i = 0; i < k; i++) {
            // int index = reduce(a, this->bitCount);
            // data[index >> 6] |= getBit(index);
            // reworked to avoid overflows
            // use the fact that reduce is not very sensitive to lower bits of a
            data[fastrangesize(a, this->arrayLength)] |= getBit(a);
            a += b;
        }
        return Ok;
    }

    const int blockShift = 15;
    const int blockLen = 1 << blockShift;

    inline void applyBlock(uint32_t *tmp, int block, int len, uint64_t *data) {
        for (int i = 0; i < len; i++) {
            uint32_t index = tmp[(block << blockShift) + i];
            data[index >> 6] |= getBit(index);
        }
    }

    template<typename ItemType, size_t bits_per_item, bool branchless,
             typename HashFamily, int k>
    Status BloomFilter<ItemType, bits_per_item, branchless, HashFamily, k>::AddAll(
            const ItemType *keys, const size_t start, const size_t end) {
        // we have that AddAll assumes that arrayLength << 6 is a
        // 32-bit integer
        if (arrayLength > 0x3ffffff) {
            for (size_t i = start; i < end; i++) {
                Add(keys[i]);
            }
            return Ok;
        }
        int blocks = 1 + arrayLength / blockLen;
        uint32_t *tmp = new uint32_t[blocks * blockLen];
        int *tmpLen = new int[blocks]();
        for (size_t i = start; i < end; i++) {
            uint64_t key = keys[i];
            uint64_t hash = hasher(key);
            uint64_t a = (hash >> 32) | (hash << 32);
            uint64_t b = hash;
            for (int j = 0; j < k; j++) {
                int index = fastrangesize(a, this->arrayLength);
                int block = index >> blockShift;
                int len = tmpLen[block];
                tmp[(block << blockShift) + len] = (index << 6) + (a & 63);
                tmpLen[block] = len + 1;
                if (len + 1 == blockLen) {
                    applyBlock(tmp, block, len + 1, data);
                    tmpLen[block] = 0;
                }
                a += b;
            }
        }
        for (int block = 0; block < blocks; block++) {
            applyBlock(tmp, block, tmpLen[block], data);
        }
        delete[] tmp;
        delete[] tmpLen;
        return Ok;
    }

    inline char bittest64(const uint64_t *t, uint64_t bit) {
        return (*t & (1L << (bit & 63))) != 0;
    }
    template<typename ItemType, size_t bits_per_item, bool branchless,
             typename HashFamily, int k>
    Status BloomFilter<ItemType, bits_per_item, branchless, HashFamily, k>::Contain(
            const ItemType &key) const {
        uint64_t hash = hasher(key);
        uint64_t a = (hash >> 32) | (hash << 32);
        uint64_t b = hash;
        if (branchless && k >= 3) {
            int b0 = data[fastrangesize(a, this->arrayLength)] >> (a & 63);
            a += b;
            int b1 = data[fastrangesize(a, this->arrayLength)] >> (a & 63);
            a += b;
            int b2 = data[fastrangesize(a, this->arrayLength)] >> (a & 63);
            if ((b0 & b1 & b2 & 1) == 0) {
                return NotFound;
            }
            for (int i = 3; i < k; i++) {
                a += b;
                if (((data[fastrangesize(a, this->arrayLength)] >> (a & 63)) & 1) == 0) {
                    return NotFound;
                }
            }
            return Ok;
        }
        for (int i = 0; i < k; i++) {
            if ((data[fastrangesize(a, this->arrayLength)] & getBit(a)) == 0) {
                return NotFound;
            }
            a += b;
        }
        return Ok;
    }

    template<typename ItemType, size_t bits_per_item, bool branchless,
             typename HashFamily, int k>
    std::string
    BloomFilter<ItemType, bits_per_item, branchless, HashFamily, k>::Info() const {
        std::stringstream ss;
        ss << "BloomFilter Status:\n"
           << "\t\tKeys stored: " << Size() << "\n";
        if (Size() > 0) {
            ss << "\t\tk:   " << BitsPerItem() << "\n";
        } else {
            ss << "\t\tk:   N/A\n";
        }
        return ss.str();
    }

    template<size_t m_up, size_t m_down, size_t k, bool branchless,
             typename HashFamily = TwoIndependentMultiplyShift>
    class BF_MA {
    public:
        uint64_t *data;
        // const size_t m;
        size_t cap = 0;
        size_t arrayLength;
        size_t bitCount;
        int kk = k;
        HashFamily hasher;

        double BitsPerItem() const { return k; }

        size_t get_m(size_t n) {
            size_t res = (n * m_up + m_down - 1) / m_down;
            assert(res > 0);
            assert(res < (1ULL << 60u));
            return res;
        }

        explicit BF_MA(const size_t n) : bitCount(get_m(n)),
                                         arrayLength((get_m(n) + 63) / 64), hasher() {
            //
            // arrayLength((bitCount + 63) / 64),bitCount(n * bits_per_item),kk(k),hasher() {
            // this->cap = 0;
            // this->kk = getBestK(bits_per_item);
            // this->bitCount = n * bits_per_item;
            // bitCount(n * bits_per_item) this->arrayLength = (bitCount + 63) / 64;
            data = new uint64_t[arrayLength];
            std::fill_n(data, arrayLength, 0);
        }

        ~BF_MA() { delete[] data; }

        // Add an item to the filter.
        Status Add(const uint64_t &item);


        // Report if the item is inserted, with false positive rate.
        Status Contain(const uint64_t &item) const;

        Status contain_branchless(const uint64_t &item) const;

        /* methods for providing stats  */
        // summary infomation
        std::string Info() const;

        // number of current inserted items;
        size_t get_cap() const { return cap; }

        // size of the filter in bytes.
        size_t SizeInBytes() const { return arrayLength * 8; }

        std::string get_name() const {
            float BPI = (100 * m_up / m_down) / 100.0;
            // double BPI = BPI_full * 100

            size_t hash_k = k;
            std::string name = "BF-MA-[bpi=10,k=" + std::to_string(k) + "]";
            // std::string name = "BF-MA-" + std::to_string(BPI) + "[k=" + std::to_string(k) + "]";

            if (branchless) {
                return name + "-{branchless}";
            } else {
                return name + "-{with-branch}";
            }
        }
    };

    template<size_t m_up, size_t m_down, size_t k, bool branchless, typename HashFamily>
    Status BF_MA<m_up, m_down, k, branchless, HashFamily>::Add(const uint64_t &key) {
        cap++;
        uint64_t hash = hasher(key);
        uint64_t a = (hash >> 32) | (hash << 32);
        uint64_t b = hash;
        for (int i = 0; i < k; i++) {
            // int index = reduce(a, this->bitCount);
            // data[index >> 6] |= getBit(index);
            // reworked to avoid overflows
            // use the fact that reduce is not very sensitive to lower bits of a
            data[fastrangesize(a, this->arrayLength)] |= getBit(a);
            a += b;
        }
        return Ok;
    }
    template<size_t m_up, size_t m_down, size_t k, bool branchless, typename HashFamily>
    Status BF_MA<m_up, m_down, k, branchless, HashFamily>::Contain(
            const uint64_t &key) const {
        uint64_t hash = hasher(key);
        uint64_t a = (hash >> 32) | (hash << 32);
        uint64_t b = hash;

        for (int i = 0; i < k; i++) {
            if ((data[fastrangesize(a, this->arrayLength)] & getBit(a)) == 0) {
                return NotFound;
            }
            a += b;
        }
        return Ok;











        if ((k == 2) and branchless) {
            int b0 = data[fastrangesize(a, this->arrayLength)] >> (a & 63);
            a += b;
            int b1 = data[fastrangesize(a, this->arrayLength)] >> (a & 63);
            if ((b0 & b1 & 1) == 0)
                return NotFound;
            return Ok;
        } else if (k == 2) {
            if (!(data[fastrangesize(a, this->arrayLength)] >> (a & 63)))
                return NotFound;

            a += b;
            if (data[fastrangesize(a, this->arrayLength)] >> (a & 63)) {
                return Ok;
            }
            return NotFound;
        }

        if (branchless && k >= 3) {
            int b0 = data[fastrangesize(a, this->arrayLength)] >> (a & 63);
            a += b;
            int b1 = data[fastrangesize(a, this->arrayLength)] >> (a & 63);
            a += b;
            int b2 = data[fastrangesize(a, this->arrayLength)] >> (a & 63);
            if ((b0 & b1 & b2 & 1) == 0) {
                return NotFound;
            }
            for (int i = 3; i < k; i++) {
                a += b;
                if (((data[fastrangesize(a, this->arrayLength)] >> (a & 63)) & 1) == 0) {
                    return NotFound;
                }
            }
            return Ok;
        }
        for (int i = 0; i < k; i++) {
            if ((data[fastrangesize(a, this->arrayLength)] & getBit(a)) == 0) {
                return NotFound;
            }
            a += b;
        }
        return Ok;
    }

    /* template<size_t m_up, size_t m_down, size_t k, bool branchless, typename HashFamily>
    Status BF_MA<m_up, m_down, k, branchless, HashFamily>::contain_branchless(
            const uint64_t &key) const {
        uint64_t hash = hasher(key);
        uint64_t a = (hash >> 32) | (hash << 32);
        uint64_t b = hash;
        size_t locations[k] = {0};
        for (size_t i = 0; i < k; i++)
        {
            locations[i] = fastrangesize(a, this->arrayLength);
            a += b;
        }

        int res = 1u;
        for (size_t i = 0; i < k; i++)
        {
            res &= data[locations[i]] >> ()
        }
        

    }

 */
    template<size_t m_up, size_t m_down, size_t k, bool branchless, typename HashFamily>
    std::string
    BF_MA<m_up, m_down, k, branchless, HashFamily>::Info() const {
        std::stringstream ss;
        ss << "BloomFilter Status:\n"
           << "\t\tKeys stored: " << get_cap() << "\n";
        if (get_cap() > 0) {
            ss << "\t\tk:   " << BitsPerItem() << "\n";
        } else {
            ss << "\t\tk:   N/A\n";
        }
        return ss.str();
    }


}// namespace bloomfilter

#endif// BLOOM_FILTER_BLOOM_FILTER_H_
