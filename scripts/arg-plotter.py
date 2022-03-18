#!/usr/bin/env python3

from Naming import name_dict, get_marker_type, get_time
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


def get_average_performance_list(lines: List[str]):
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

        name = get_filter_name_from_lines(lines)
        # print(name)
        # print(performance_list)
        return performance_list

    assert(len(beg_list) == len(end_list))
    perfs_lists = []
    for i in range(len(beg_list)):
        temp_lines = lines[beg_list[i]: end_list[i] + 1]
        assert temp_lines[0].startswith("NAME")
        assert temp_lines[-1].startswith("BENCH")
        temp_perf = get_performance_list(temp_lines)
        perfs_lists.append(temp_perf)

    def get_lists_average(ll: List[List]):
        k = len(ll)
        cols = len(ll[0])
        assert (cols == 4)
        rows = len(ll[0][0])
        z_list = [0] * rows
        avg_fl_list = [z_list[:] for _ in range(cols)]
        # print(len(avg_fl_list))
        # print(len(avg_fl_list[0]))
        for col in range(cols):
            for row in range(rows):
                for i in range(k):
                    # print(col, row)
                    avg_fl_list[col][row] += ll[i][col][row]

        for col in range(cols):
            for row in range(rows):
                avg_fl_list[col][row] /= k
        return avg_fl_list

    avg_fl_list = get_lists_average(perfs_lists)
    return avg_fl_list
    # end_list = []


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

    # print("="*40)
    # print(len(perfs_lists))
    # print(len(perfs_lists[0]))
    # print(len(perfs_lists[0][0]))
    # print("="*40)
    runs = len(perfs_lists)
    assert len(perfs_lists[0]) == 4
    operations = len(perfs_lists[0])
    rounds = len(perfs_lists[0][0])

    def built_median_single_list() -> List[List[float]]:
        fl = [[0] * rounds for _ in range(operations)]
        assert operations == 4
        for op in range(operations - 1):
            for k in range(rounds):
                # print(op, k)
                temp_med = median(built_temp_list(perfs_lists, op, k))
                fl[op][k] = temp_med
        return fl

    med_fl_list = built_median_single_list()
    return med_fl_list


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


def get_op_divisors(lines: List[str]):
    # a = find_line_index(lines, "NAME")
    b = find_line_index(lines, "FILTER_MAX_CAPACITY")
    c = find_line_index(lines, "NUMBER_OF_LOOKUP")

    # name = lines[a].split()[1]
    filter_max_cap = int(lines[b].split()[1])
    lookup_reps = int(lines[c].split()[1])
    return filter_max_cap, lookup_reps


def get_y_values(filters_names: List[str], lists_of_lists: List[List], divisor, op_name: str, units=1e9):
    assert (len(filters_names) == len(lists_of_lists))
    fl_y = []
    for temp_y_vals in lists_of_lists:
        y_range = temp_y_vals
        y_range = [divisor / (i / units) if i != 0 else 0 for i in y_range]
        fl_y.append(y_range)
    return fl_y


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
        "fraction of elements that are <=1% from median (in thier category)  \t{:}".format(r1))
    print(
        "fraction of elements that are <=0.5% from median (in thier category)\t{:}".format(r05))
    print("min & max diviations:", s_fd[0], s_fd[-1])


def get_data(f_list: list):
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
    ins_arg = (names_list, ins_list, add_step, "Insertions")
    uni_arg = (names_list, uni_list, find_step, "Uni-Lookups")
    yes_arg = (names_list, yes_list, add_step, "Yes-Lookups")
    ###########################################

    args = [ins_arg, uni_arg, yes_arg]
    fl_y_list = [get_y_values(*temp_arg) for temp_arg in args]
    rounds = len(ins_list[0])
    x_range = [round((i + 1) / rounds, 4) for i in range(rounds)]
    filters_names = [name_dict(t) for t in names_list]

    def sorter():
        size = len(filters_names)
        temp_for_sort = [(filters_names[i], fl_y_list[0][i],fl_y_list[1][i],fl_y_list[2][i]) for i in range(size)]
        temp_for_sort.sort()
        flatted_data = []
        for op in range(4):
            temp = [temp_for_sort[i][op] for i in range(size)]
            flatted_data.append(temp[:])
        
        new_fl_y_list = flatted_data[1:]
        new_filters_names = flatted_data[0]
        return new_fl_y_list, x_range, new_filters_names

    return sorter()
    # return fl_y_list, x_range, filters_names


def get_paths_from_base_names(base_names: List[str]):
    return [os.path.join("Inputs", i) for i in base_names]


def get_ba_limits(op_ll: list):
    filter_num = len(filters_names)
    if filter_num != len(op_ll):
        print(filter_num, len(op_ll))
    assert filter_num == len(op_ll)
    min_max_bbf = []
    min_others = 1e10
    max_others = 0
    for i in range(filter_num):
        temp_filter_name = filters_names[i]
        # y_range = [i]
        y_range = op_ll[i]
        if (temp_filter_name == "BBF"):
            min_max_bbf = min(y_range), max(y_range)
        else:
            max_others = max(max_others, max(y_range))
            min_others = min(min_others, min(y_range))
    return *min_max_bbf, min_others, max_others


def get_ba_limits_all(op_ll: list):
    filter_num = len(filters_names)
    if filter_num != len(op_ll):
        print(filter_num, len(op_ll))
    assert filter_num == len(op_ll)
    min_max_bbf = [1e10, 0]
    min_others = 1e10
    max_others = 0
    for i in range(filter_num):
        temp_filter_name = filters_names[i]
        # y_range = [i]
        y_range = op_ll[i]
        if (temp_filter_name.startswith("BBF")):
            if (temp_filter_name == "BBF-Flex"):
                min_max_bbf[0] = min(y_range)
            elif (temp_filter_name == "BBF"):
                min_max_bbf[1] = max(y_range)
            else:
                print(temp_filter_name)
                assert 0

            # min_max_bbf = min(y_range), max(y_range)
        else:
            max_others = max(max_others, max(y_range))
            min_others = min(min_others, min(y_range))
    return *min_max_bbf, min_others, max_others


def fig3_ba_gridspec_all(data, name="", set_fontsize: int = 14):
    y_lll, x_range, filters_names = data
    # fig = plt.subplots(figsize=(15,5))
    fig = plt.figure(figsize=(15, 5))
    sps1, sps2, sps3 = GridSpec(1, 3, figure=fig)
    spec_list = [sps1, sps2, sps3]
    baxs = []
    op_names_list = ["(a) Insertions", "(b) Uniform lookups",
                     "(c) Yes lookups"]

    def get_ba_limits_all(op_ll: list):
        bbfs_names = ["BBF", "BBF-Flex", "Impala"]
        assert len(filters_names) == len(op_ll)
        indexes = []
        for i in range(len(filters_names)):
            t_name = filters_names[i]
            for prefix in bbfs_names:
                if t_name.startswith(prefix):
                    indexes.append(i)
                    break

        min_max_bbf = [1e10, 0]
        min_others = 1e10
        max_others = 0

        for i in range(len(filters_names)):
            if i not in indexes:
                y_range = op_ll[i]
                max_others = max(max_others, max(y_range))
                min_others = min(min_others, min(y_range))

        # min_max_bbf = [1e10, 0]
        if len(indexes):
            min_max_bbf[0] = min([min(op_ll[i]) for i in indexes])
            min_max_bbf[1] = max([max(op_ll[i]) for i in indexes])
        return *min_max_bbf, min_others, max_others

    for fig_index in range(len(y_lll)):
        yba_lim = get_ba_limits_all(y_lll[fig_index])
        y_delim = ((yba_lim[2]*0.9, yba_lim[3] * 1.04),
                   (yba_lim[0]*0.98, yba_lim[1]*1.02))
        if fig_index == 1:
            bax = brokenaxes(subplot_spec=spec_list[fig_index])
        else:
            bax = brokenaxes(ylims=y_delim, d=0.005,
                             subplot_spec=spec_list[fig_index])
        bax.set_title(op_names_list[fig_index], fontsize=18)
        bax.set_xlabel("Load", fontsize=14)
        bax.set_ylabel("ops/sec", fontsize=14)
        for i in range(len(y_lll[fig_index])):
            yv = y_lll[fig_index][i]
            marker_shape = get_marker_type(filters_names[i])
            bax.plot(
                x_range, yv, label=filters_names[i], marker=marker_shape, markersize=2)
            # bax.grid(axis='both', which='major', ls='-',linewidth=1)
            # bax.grid(axis='both', which='minor', ls='--',linewidth=0.4)
            bax.grid(axis='y', which='major', ls='-', linewidth=1)
            bax.grid(axis='y', which='minor', ls='--', linewidth=0.4)
            baxs.append(bax)
            # axes[fig_index].plot(x_range, yv, label=filters_names[i], marker=marker_shape, markersize=3)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    for bax in baxs:
        for handle in bax.diag_handles:
            handle.remove()
        bax.draw_diags()
        bax.minorticks_on()
    # plt.show()
    # return
    # plt.show()

    handles, labels = baxs[0].axs[0].get_legend_handles_labels()
    # labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t))
    plt.legend(handles, labels, loc='upper center',
               bbox_to_anchor=(-0.6, -0.1), ncol=(len(filters_names) + 1)//2)
    # plt.legend(handles, labels, loc='upper center', bbox_to_anchor=(-0.6, -0.05), ncol=5)
    # plt.tight_layout()
    file_name = name
    if not len(name):
        file_name = "bench-median-lables-" + str(set_fontsize) + ".pdf"
    else:
        file_name += ".pdf"
    plt.savefig(file_name, dpi=400, bbox_inches="tight")
    # plt.show()


def fig3_no_brokenaxis_local_arg(y_lll, x_range, filters_names, name: str = "default"):
    fig = plt.figure(figsize=(15, 5))
    sps1, sps2, sps3 = GridSpec(1, 3, figure=fig)
    spec_list = [sps1, sps2, sps3]
    baxs = []
    op_names_list = ["(a) Insertions", "(b) Uniform lookups",
                     "(c) Yes lookups"]

    for fig_index in range(len(y_lll)):
        bax = brokenaxes(subplot_spec=spec_list[fig_index])
        bax.set_title(op_names_list[fig_index])
        for i in range(len(y_lll[fig_index])):
            yv = y_lll[fig_index][i]
            marker_shape = get_marker_type(filters_names[i])
            bax.plot(
                x_range, yv, label=filters_names[i], marker=marker_shape, markersize=2)
            bax.grid(axis='y', which='major', ls='-', linewidth=1)
            bax.grid(axis='y', which='minor', ls='--', linewidth=0.4)
            baxs.append(bax)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    for bax in baxs:
        for handle in bax.diag_handles:
            handle.remove()
        bax.draw_diags()
        bax.minorticks_on()

    handles, labels = baxs[0].axs[0].get_legend_handles_labels()
    plt.legend(handles, labels, loc='upper center',
               bbox_to_anchor=(-0.6, -0.05), ncol=len(filters_names)//3)

    plt.savefig(name + ".pdf", dpi=400, bbox_inches="tight")
    plt.show()


def fig3_no_brokenAxis(name: str = "default"):
    fig = plt.figure(figsize=(15, 5))
    sps1, sps2, sps3 = GridSpec(1, 3, figure=fig)
    spec_list = [sps1, sps2, sps3]
    baxs = []
    op_names_list = ["(a) Insertions", "(b) Uniform lookups",
                     "(c) Yes lookups"]

    for fig_index in range(len(y_lll)):
        bax = brokenaxes(subplot_spec=spec_list[fig_index])
        bax.set_title(op_names_list[fig_index])
        for i in range(len(y_lll[fig_index])):
            yv = y_lll[fig_index][i]
            marker_shape = get_marker_type(filters_names[i])
            bax.plot(
                x_range, yv, label=filters_names[i], marker=marker_shape, markersize=2)
            # bax.grid(axis='both', which='major', ls='-',linewidth=1)
            # bax.grid(axis='both', which='minor', ls='--',linewidth=0.4)
            bax.grid(axis='y', which='major', ls='-', linewidth=1)
            bax.grid(axis='y', which='minor', ls='--', linewidth=0.4)
            baxs.append(bax)
            # axes[fig_index].plot(x_range, yv, label=filters_names[i], marker=marker_shape, markersize=3)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    for bax in baxs:
        for handle in bax.diag_handles:
            handle.remove()
        bax.draw_diags()
        bax.minorticks_on()
    # plt.show()
    # return
    # plt.show()

    handles, labels = baxs[0].axs[0].get_legend_handles_labels()
    # labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t))
    plt.legend(handles, labels, loc='upper center',
               bbox_to_anchor=(-0.6, -0.05), ncol=len(filters_names)//3)
    # plt.tight_layout()
    plt.savefig(name + ".pdf", dpi=400, bbox_inches="tight")
    plt.show()


def old_main():
    """
    sys.argv[1] = path
    """
    
    def main_helper(path, name):
        chosen_files = os.listdir(path)
        files_list = [os.path.join(path, i)
                      for i in chosen_files if not i.endswith(".csv")]
        files_list.sort()
        data = get_data(files_list)
        final_diver(files_list)
        fig3_ba_gridspec_all(data, name)

    argc: int = len(sys.argv)
    if argc == 1:
        path = os.path.abspath(os.getcwd())
        path = os.path.join(path, "Inputs")
        assert os.path.isdir(path)
        name = "bench{:}".format(get_time())
        main_helper(path, name)
    elif argc == 2:
        path = sys.argv[1]
        name = "bench{:}".format(get_time())
        main_helper(path, name)
    else:
        print("Too many arguments where given ({:})".format(argc))


old_main()
