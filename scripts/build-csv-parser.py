#!/usr/bin/env python3

from typing import *
import matplotlib.pyplot as plt
import pandas as pd

import os
from matplotlib import cm
import numpy as np
from statistics import median
from Naming import name_dict, get_time

Def_COLOR = cm.inferno_r(np.linspace(.4, .8, 30))

dummy = []
dummy_names = []
dummy_tuple_list = []


def built_plot_bar(filters_names: List[str], built_time_list_ns: List[int], divisor, units=1e9):
    # print("Here!")
    built_time = [i / units for i in built_time_list_ns]
    filters_names = [name_dict(t.strip()) for t in filters_names]

    tuples_list = [(built_time[_], filters_names[_])
                   for _ in range(len(built_time))]
    tuples_list.sort()
    dummy_tuple_list.append(tuples_list)
    s_BT = [i[0] for i in tuples_list]
    s_names = [i[1] for i in tuples_list]

    line = "{:30}:\t{:.7f}"
    for i in tuples_list:
        print(line.format(i[1], i[0]))

    s = pd.Series(
        s_BT,
        index=s_names,
        # Title = "tomer"
    )
    factor = 100
    # temp_color = cm.inferno_r(np.linspace(0.1, .7, len(filters_names) + 2))
    # temp_color = cm.RdYlGn(np.linspace(0, 1, len(filters_names) + 2))[::-1]
    # color_plate = cm.RdYlGn(np.linspace(0, 1, factor + 1))[::-1]
    color_plate = cm.RdYlGn(np.linspace(0.2, 0.8, factor + 1))[::-1]

    def get_index_in_range(x):
        r_min, r_max = min(built_time), max(built_time)
        assert r_min <= x <= r_max
        shifted = x - r_min
        normalized = shifted / (r_max - r_min)
        index = round(normalized*factor)
        assert index <= factor
        return index

    color_indexes = [get_index_in_range(i[0]) for i in tuples_list]
    final_colors = [color_plate[i] for i in color_indexes]
    s.plot(
        kind='bar',
        stacked=True,
        color=final_colors,
        # ylabel = "Seconds"
        # title="Built Time",

        # colormap='Paired',
    )
    plt.grid(which="major", linestyle='--', linewidth=.8, axis='y')
    plt.grid(which="minor", linestyle='-', linewidth=.4, axis='y')
    # plt.set_title("Built Time", fontsize = 18)
    plt.minorticks_on()
    plt.ylabel("Seconds", fontsize=14)
    # plt.yticks()
    # s.set_axis()
    pic_dir = "Built-time-median-({:}).pdf".format(get_time())
    plt.savefig(pic_dir, format="pdf", bbox_inches='tight')
    # plt.show()
    plt.clf()


def get_lines(file_name: str) -> list:
    # file_name = os.path.basename(dir)
    # RunAll_path = os.getcwd()
    # main_path = os.path.dirname(RunAll_path)
    # path = os.path.join(main_path, "scripts")
    # path = os.path.join(path, file_name)
    f = open(file_name, "r")
    lines = f.readlines()
    f.close()
    return lines


def process_list_of_files(csv_file: list):
    lines = get_lines(csv_file)
    first_line = 0
    for i in range(-1, -len(lines) - 1, -1):
        if lines[i].startswith("n ="):
            first_line = i
            break
    assert first_line != 0
    lines = lines[first_line:]
    div = int(lines[0].split("=")[1])
    values_list = []
    names_list = []
    for line in lines[1:]:
        split_l = line.split(",")
        names_list.append(split_l[0])
        temp = [int(i) for i in split_l[1:-1]]
        values_list.append(temp)

    built_time_ns_list = []
    for i in values_list:
        temp_built = median(i)
        built_time_ns_list.append(temp_built)

    filter_max_cap = div

    built_plot_bar(names_list, built_time_ns_list, filter_max_cap)


def get_paths_from_base_names(base_names: List[str]):
    return [os.path.join("Built-Inputs", i) for i in base_names]


def main():
    filename_list = [i for i in os.listdir("../scripts/") if i.startswith("build-all") and i.endswith(".csv")]
    #print("filename_list:",filename_list)
    assert len(filename_list) == 1
    filename = filename_list[0]
    path = os.path.join("../scripts/", filename)
    process_list_of_files(path)


main()
