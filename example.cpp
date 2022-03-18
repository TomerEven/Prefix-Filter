#include "Tests/wrappers.hpp"

int main(){
    using spare = TC_shortcut; // Any incremental filter can replace TC_shortcut.
    using prefixFilter = Prefix_Filter<spare>; // 

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