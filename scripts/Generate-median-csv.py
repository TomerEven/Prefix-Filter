#!/usr/bin/env python3

from Naming import name_dict, get_time
from typing import *
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
# import matplotlib
# import pandas as pd
import os
import sys

# https://github.com/bendichter/brokenaxes
from brokenaxes import brokenaxes

from statistics import median
USE_AVERAGE = 0
USE_MEDIAN = 1

assert (USE_AVERAGE ^ USE_MEDIAN)


def get_lines(dir: str) -> list:
    # file_name = os.path.basename(dir)
    f = open(dir, "r")
    lines = f.readlines()
    f.close()
    return lines


def find_line_index(lines: list, phrase: str) -> int:
    for i in range(len(lines)):
        line = lines[i].strip()
        if (len(line) == 0) or (line.startswith("#")):
            continue
        if line.startswith(phrase):
            return i
    return -1


def get_filter_name_from_lines(lines):
    i = find_line_index(lines, "NAME")
    if (i == -1):
        print(lines)
        assert 0
    assert i != -1
    name = lines[i][5:].strip()
    return name

def get_all_values_from_perf_list(lines: List[str]):
    def find_all_partial_matches_in_lines(lines: List[str], pattern: str):
        indexes = []
        for i in range(len(lines)):
            if lines[i].startswith(pattern):
                indexes.append(i)
        return indexes

    beg_list = find_all_partial_matches_in_lines(lines, "NAME")
    end_list = find_all_partial_matches_in_lines(lines, "BENCH_END")

    def get_performance_list(lines: List[str]):
        start_line = find_line_index(lines, "BENCH_START") + 1
        end_line = find_line_index(lines, "BENCH_END") - 1  # inclusive.

        performance_list = [[] for _ in range(4)]
        for line in lines[start_line:end_line]:
            if not (len(line.strip())):
                continue

            temp_list0 = line.split(",")
            temp_list1 = [int(i.strip()) for i in temp_list0]
            assert(len(temp_list1) == 4)
            for i in range(4):
                performance_list[i].append(temp_list1[i])

        return performance_list

    assert(len(beg_list) == len(end_list))
    perfs_lists = []
    for i in range(len(beg_list)):
        temp_lines = lines[beg_list[i]: end_list[i] + 1]
        assert temp_lines[0].startswith("NAME")
        assert temp_lines[-1].startswith("BENCH")
        temp_perf = get_performance_list(temp_lines)
        perfs_lists.append(temp_perf)

    def built_temp_list(single_perf_list: List, op: int, k: int):
        """[summary]

        Args:
            single_perf_list (List): [description]
            op (int): [description]
            k (int): [description]

        Returns:
            [Lists of lists]: [Each cell in the list, is ]
        """
        # single_perf_list[Run][Operation][Round]
        # temp_l == [Operation][Round][Run(0);Run(last)]
        runs = len(single_perf_list)
        assert len(single_perf_list) == runs
        temp_l = [single_perf_list[i][op][k] for i in range(runs)]
        return temp_l

    assert len(perfs_lists[0]) == 4
    operations = len(perfs_lists[0])
    rounds = len(perfs_lists[0][0])

    def built_raw_single_list():
        fl = [[0] * rounds for _ in range(operations)]
        assert operations == 4
        for op in range(operations - 1):
            for k in range(rounds):
                temp_med = built_temp_list(perfs_lists, op, k)
                fl[op][k] = temp_med
        return fl

    med_fl_list = built_raw_single_list()
    return med_fl_list

def get_median_performance_list(lines: List[str]) -> List[List[float]]:
    """Returns a List of Lists: [operation][round(load)]

    Args:
        lines (List[str]): [description]

    Returns:
        List[List[float]]: [description]
    """
    def find_all_partial_matches_in_lines(lines: List[str], pattern: str):
        indexes = []
        for i in range(len(lines)):
            if lines[i].startswith(pattern):
                indexes.append(i)
        return indexes

    beg_list = find_all_partial_matches_in_lines(lines, "NAME")
    end_list = find_all_partial_matches_in_lines(lines, "BENCH_END")

    def get_performance_list(lines: List[str]):
        start_line = find_line_index(lines, "BENCH_START") + 1
        end_line = find_line_index(lines, "BENCH_END") - 1  # inclusive.

        performance_list = [[] for _ in range(4)]
        for line in lines[start_line:end_line]:
            if not (len(line.strip())):
                continue

            temp_list0 = line.split(",")
            temp_list1 = [int(i.strip()) for i in temp_list0]
            assert(len(temp_list1) == 4)
            for i in range(4):
                performance_list[i].append(temp_list1[i])

        return performance_list

    assert(len(beg_list) == len(end_list))
    perfs_lists = []
    for i in range(len(beg_list)):
        temp_lines = lines[beg_list[i]: end_list[i] + 1]
        assert temp_lines[0].startswith("NAME")
        assert temp_lines[-1].startswith("BENCH")
        temp_perf = get_performance_list(temp_lines)
        perfs_lists.append(temp_perf)

    def built_temp_list(single_perf_list: List, op: int, k: int) -> List[float]:
        # single_perf_list[Run][Operation][Round]
        runs = len(single_perf_list)
        assert len(single_perf_list) == runs
        temp_l = [single_perf_list[i][op][k] for i in range(runs)]
        return temp_l

    assert len(perfs_lists[0]) == 4
    operations = len(perfs_lists[0])
    rounds = len(perfs_lists[0][0])

    def built_median_single_list() -> List[List[float]]:
        fl = [[0] * rounds for _ in range(operations)]
        assert operations == 4
        for op in range(operations - 1):
            for k in range(rounds):
                temp_med = median(built_temp_list(perfs_lists, op, k))
                fl[op][k] = temp_med
        return fl

    med_fl_list = built_median_single_list()
    return med_fl_list

def get_op_divisors(lines: List[str]):
    # a = find_line_index(lines, "NAME")
    b = find_line_index(lines, "FILTER_MAX_CAPACITY")
    c = find_line_index(lines, "NUMBER_OF_LOOKUP")

    # name = lines[a].split()[1]
    filter_max_cap = int(lines[b].split()[1])
    lookup_reps = int(lines[c].split()[1])
    return filter_max_cap, lookup_reps


def get_raw_all_data(f_list: list):
    lines_list = []
    names_list = []
    for temp_file in f_list:
        temp_lines = get_lines(temp_file)
        lines_list.append(temp_lines)
        names_list.append(get_filter_name_from_lines(temp_lines))

    perf_lists = []

    for temp_lines in lines_list:
        # temp_perf == [Operation][Round][Run(0);Run(last)]
        temp_perf = get_all_values_from_perf_list(temp_lines)
        perf_lists.append(temp_perf)

    def get_all_diviate_list(l):
        def ratio(x, denom):
            assert(denom != 0)
            return (x-denom)/denom
        t_med = median(l)
        fl = [ratio(i, t_med) for i in l]
        return fl

    flat_div_list = []
    filters_num = len(perf_lists)
    operation_num = len(perf_lists[0])
    rounds = len(perf_lists[0][0])
    for op in range(3):
        for t_filter in range(filters_num):
            for t_round in range(rounds):
                temp_l = perf_lists[t_filter][op][t_round]
                # print(op, t_filter, t_round, end=":\t")
                # print(temp_l)
                # if temp_l == 0:
                temp_res = get_all_diviate_list(temp_l)
                flat_div_list += temp_res
            # print()
    return flat_div_list


def final_diver(f_list: list) -> None:
    flat_diviate = get_raw_all_data(f_list)
    s_fd = sorted(flat_diviate)
    above_05 = [i for i in s_fd if abs(i) > 0.005]
    above_1 = [i for i in s_fd if abs(i) > 0.01]
    r1 = len(above_1)/len(s_fd)
    r05 = len(above_05)/len(s_fd)
    print(
        "fraction of elements that are at most 1% away from median (in thier category)  \t{:<.5f}".format(1-r1))
    print(
        "fraction of elements that are at most 0.5% from median (in thier category)\t{:<.5f}".format(1-r05))
    print("min & max diviations:", s_fd[0], s_fd[-1])


def generate_csvs(f_list: list):
    lines_list = []
    names_list = []
    for temp_file in f_list:
        temp_lines = get_lines(temp_file)
        lines_list.append(temp_lines)
        names_list.append(get_filter_name_from_lines(temp_lines))

    perf_lists = []

    for temp_lines in lines_list:
        temp_perf = []
        if (USE_MEDIAN):
            assert(not USE_AVERAGE)
            temp_perf = get_median_performance_list(temp_lines)
        elif (USE_AVERAGE):
            temp_perf = get_average_performance_list(temp_lines)
        else:
            assert(0)
        perf_lists.append(temp_perf)

    filter_max_cap, lookup_reps = get_op_divisors(lines_list[0])
    rounds_num = len(perf_lists[0][0])
    add_step = round(filter_max_cap / rounds_num)
    find_step = round(lookup_reps / rounds_num)

    ###########################################
    ins_list = [temp_perf[0] for temp_perf in perf_lists]
    uni_list = [temp_perf[1] for temp_perf in perf_lists]
    yes_list = [temp_perf[2] for temp_perf in perf_lists]
    ###########################################
    curr_time = get_time()
    names = ["add-med-({:}).csv", "uni-med-({:}).csv", "yes-med-({:}).csv"]
    names = [nm.format(curr_time) for nm in names]
    f_add = open(names[0], "a")
    f_uni = open(names[1], "a")
    f_yes = open(names[2], "a")
    files = [f_add, f_uni, f_yes]
    lists = [ins_list, uni_list, yes_list]
    filters_names = [name_dict(t) for t in names_list]

    size = len(filters_names)
    assert size == len(ins_list)
    for op in range(len(files)):
        t_file = files[op]
        t_list = lists[op]
        for i in range(size):
            t_file.write(filters_names[i] + ",")
            temp_line = str(t_list[i])[1:-1]
            t_file.write(temp_line + "\n")

    f_add.close()
    f_uni.close()
    f_yes.close()


def main_helper(path):
        chosen_files = os.listdir(path)
        files_list = [os.path.join(path, i)
                      for i in chosen_files if not i.endswith(".csv")]
        files_list.sort()
        final_diver(files_list)
        generate_csvs(files_list)


def main():
    """
    sys.argv[1] = path
    """
    
    argc: int = len(sys.argv)
    if argc == 1:
        path = os.path.abspath(os.getcwd())
        path = os.path.join(path, "Inputs")
        assert os.path.isdir(path)
        # name = "bench{:}".get_time()
        main_helper(path)
    elif argc == 2:
        path = sys.argv[1]
        # name = "bench{:}".format(get_time())
        main_helper(path)
    else:
        print("Too many arguments where given ({:})".format(argc))


main_helper("./Inputs/")
