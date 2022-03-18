cmake_minimum_required(VERSION 3.10)

# set(CMAKE_C_COMPILER gcc-10)
# set(CMAKE_CXX_COMPILER g++-10)
# set(CMAKE_C_COMPILER clang)
# set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_CXX_STANDARD 17)
project(APD_ONLY)

# set(COMPILE_FLAGS "-fprofile-instr-generate -fcoverage-mapping")
# set(CMAKE_EXE_LINKER_FLAGS "-fprofile-instr-generate")

set(CMAKE_CXX_FLAGS "-march=native")

# Seems like those are necessary.
# set(CMAKE_CXX_FLAGS "-mavx2 -mbmi -mbmi2 -mlzcnt -mpopcnt -mavx512vl -mavx512vbmi -mavx512dq -mavx512f -mavx512cd")

#-fsanitize=undefined
#-fsanitize=memory
#set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=undefined -fno-omit-frame-pointer")
#set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fsanitize=address -fno-omit-frame-pointer")

set(PF_FILES
        Prefix-Filter/Shift_op.hpp Prefix-Filter/Shift_op.cpp
        Prefix-Filter/min_pd256.hpp Prefix-Filter/min_pd256.cpp
)

set(CF_FILES
        cuckoofilter/src/bitsutil.h cuckoofilter/src/permencoding.h
        cuckoofilter/src/printutil.cc cuckoofilter/src/debug.h
        cuckoofilter/src/singletable.h cuckoofilter/src/packedtable.h
        cuckoofilter/src/cuckoofilter.h
        cuckoofilter/src/cuckoofilter_stable.h
        )

set(TESTS_FILES 
        hashutil.h
        Tests/wrappers.hpp
        Tests/smart_tests.hpp
        Tests/smart_tests.cpp
        # Tests/PerfEvent.hpp
        )

set(TC_SHORTCUT_FILES
#        hashutil.h
        TC-Shortcut/tc-sym.hpp
        TC-Shortcut/tc-sym.cpp
        TC-Shortcut/TC-shortcut.hpp)


set(Bloom_Files
        Bloom_Filter/simd-block.h
        Bloom_Filter/simd-block-fixed-fpp.h
        Bloom_Filter/Impala512.h
        Bloom_Filter/bloom.hpp
        # Bloom_Filter/bloom-simple.hpp
)

set(ALL_FILES_NM
        ${Bloom_Files}
        ${TESTS_FILES}
        ${PF_FILES}
        
        ${CF_FILES}
        ${TC_SHORTCUT_FILES}
        
        )
set(ALL_FILES
        ${ALL_FILES_NM}
        # ${XOR_FILES}
        all_main.cpp
        )


################################################################################
################################################################################
################################################################################
################################################################################
# Measure built
add_executable(measure_built0 ${ALL_FILES_NM} main-built.cpp)
add_executable(measure_built3 ${ALL_FILES_NM} main-built.cpp)
add_executable(measure_built ${ALL_FILES_NM} main-built.cpp)
target_compile_options(measure_built0 PRIVATE -O0 -g3 -W)
target_compile_options(measure_built3 PRIVATE -O3 -g3)
target_compile_options(measure_built PRIVATE -Ofast -DNDEBUG)

################################################################################

# Measure fpp
add_executable(measure_fpp0 ${ALL_FILES_NM} main-fpp.cpp)
add_executable(measure_fpp3 ${ALL_FILES_NM} main-fpp.cpp)
add_executable(measure_fpp ${ALL_FILES_NM} main-fpp.cpp)
target_compile_options(measure_fpp0 PRIVATE -O0 -g3 -W)
target_compile_options(measure_fpp3 PRIVATE -O3 -g3)
target_compile_options(measure_fpp PRIVATE -Ofast -DNDEBUG )

################################################################################

# Measure perf
add_executable(measure_perf0 ${ALL_FILES_NM} main-perf.cpp)
add_executable(measure_perf3 ${ALL_FILES_NM} main-perf.cpp)
add_executable(measure_perf ${ALL_FILES_NM} main-perf.cpp)
target_compile_options(measure_perf0 PRIVATE -O0 -g3 -W)
target_compile_options(measure_perf3 PRIVATE -O3 -g3)
target_compile_options(measure_perf PRIVATE -Ofast -DNDEBUG)

################################################################################

# Measure example
add_executable(example0 ${ALL_FILES_NM} example.cpp)
add_executable(example ${ALL_FILES_NM} example.cpp)
target_compile_options(example0 PRIVATE -O0 -g3 -W)
target_compile_options(example PRIVATE -Ofast -DNDEBUG)


################################################################################
