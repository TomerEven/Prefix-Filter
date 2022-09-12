# The Prefix Filter

Implementation of **Prefix-Filter**, which is an incremental filter (approximate set-membership queries).
If you plan on using the Prefix-Filter, please cite our paper:
**[Prefix Filter: Practically and Theoretically Better Than Bloom](https://arxiv.org/abs/2203.17139).** Tomer Even, Guy Even, Adam Morrison.
To appear in PVLDB, 15(7).

Short talk abouth the Prefix Filter: https://www.youtube.com/watch?v=KMVtvACSGo0

## Prerequisites

- **Compiler:** A C++17 compiler such as GNU G++ or LLVM Clang++.
- CMake (Version 3.10 or higher).
- **System:** Linux.
- **Hardware:** Intel. Support of AVX512 is a must.

###### Python Packages To Produce The Graphs

- matplotlib
- brokenaxes (From [here](https://github.com/bendichter/brokenaxes)).
- pandas


<!-- ###### Optional - Perf Event Wrapper.
- **Root Access:** Using the linux perf event wrapper (Link: [viktorleis/perfevent](https://github.com/viktorleis/perfevent)) requires root access.
# How To Use -->

There are three main targets for three different benchmarks.

1) `measure_perf` for benchmarking performance of insertions and lookups under various loads.
2) `measure_build` for build-time.
3) `measure_fpp` for evaluating the false positive probability, and the "effective space consumption".

There is also an example of how to use the Prefix-Filter in the file `example.cpp`:
```cpp
#include "Tests/wrappers.hpp"

int main(){
    using spare = TC_shortcut; // Any incremental filter can replace TC_shortcut.
    using prefixFilter = Prefix_Filter<spare>; 

    size_t filter_max_capacity = 1'000'000; // Choose any size.
    prefixFilter example_filter = FilterAPI<prefixFilter>::ConstructFromAddCount(filter_max_capacity);

    uint64_t x1 = 0x0123'4567'89ab'cdef;
    FilterAPI<prefixFilter>::Add(x1, &example_filter); // Insertions of an item x1. Insertion can be performed only one step at a time.

    bool res = FilterAPI<prefixFilter>::Contain(x1, &example_filter); // Lookup of x1.
    assert(res); //No false negative.
    
    uint64_t y1 = ~0x0123'4567'89ab'cdef;
    bool res2 = FilterAPI<prefixFilter>::Contain(y1, &example_filter); // Lookup of y1.
    std::cout << res2 << std::endl; // Possible false positive. (Although with one item in the filter, this is highly unlikely.)
    return 0;
}
```

### To build

```
git clone -b master https://github.com/TheHolyJoker/Prefix-Filter.git
cd Prefix-Filter
mkdir build
cd build
cmake .. -DCMAKE_C_COMPILER=gcc-10 -DCMAKE_CXX_COMPILER=g++-10
t="measure_perf measure_built measure_fpp"; time cmake --build ./ --target $t -j 20
```
**On GCC release:** Older releases like 9.3 will work. We recommend using newer releases, which seems to perform better.
### To Run

To run all benchmarks (from `Prefix-Filter/build` directory):
```
cp ../RunAll.sh RunAll.sh 
./RunAll.sh
```

<!-- If you are planning on using the perf event wrapper, the last line should run with root privilege: `sudo ./RunAll.sh` -->
###### Specific Target

Any specific target (for example `measure_perf`) can be built and executed with as follows:
```
t="measure_perf"; time cmake --build ./ --target $t -j 20 && time taskset -c 2 ./$t 
```

<!-- If you are planning on using the perf event wrapper, then use `sudo` after the `&&`.
For running on different core than `2`, line 78 in `Tests/PerfEvent.hpp` should be changed. -->

# Credits

- **Xor Filter**:
["Xor Filters: Faster and Smaller Than Bloom and Cuckoo
Filters."](https://arxiv.org/pdf/1912.08258.pdf)
Graf, Thomas Mueller, and Daniel Lemire. \
[Repository](https://github.com/FastFilter/fastfilter_cpp).\
We build upon Xor filter's benchmarks.
We also used Its BBF variant, its fast Bloom filter, and its [fast modulo alternative](https://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/).
<!-- ---
We Integrated a perf event wrapper taken from [viktorleis/perfevent](https://github.com/viktorleis/perfevent). This requires running the code as root.

--- -->
- **Cuckoo Filter** \
["Cuckoo Filter: Practically Better Than Bloom." ](https://www.cs.cmu.edu/~dga/papers/cuckoo-conext2014.pdf) Fan B, Andersen DG, Kaminsky M, Mitzenmacher MD.
[Repository](https://github.com/efficient/cuckoofilter)
- **Blocked Bloom Filter**\
We used two variants taken from the Xor filter's repository.
<!-- One taken from Impala repository, one from Xor filter's repository, and we also implemented another one, using AVX512 instructions, which is build upon the Xor filter's variant. -->
<!-- --- -->
- **Vector Quotient Filter**\
["Vector Quotient Filters: Overcoming The Time/Space Trade-Off In Filter Design."](https://research.vmware.com/files/attachments/0/0/0/0/1/4/7/sigmod21.pdf). Pandey P, Conway A, Durie J, Bender MA, Farach-Colton M, Johnson R. Vector quotient filters: Overcoming the time/space trade-off in filter design.\
[Repository](https://github.com/splatlab/vqf).\
However, we used our own implementation, called *twoChoicer* (In file `TC-Shortcut/TC-shortcut.hpp`).
