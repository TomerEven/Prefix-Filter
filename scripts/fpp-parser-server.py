#!/usr/bin/env python3

# from typing import *
# import matplotlib.pyplot as plt
# import numpy as np
import os
from typing import *
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import time
import os

import numpy as np


headers = ["Filter", "Size in bytes", "Ratio of yes-queries",
           "bits per item (average)", "optimal bits per item (w.r.t. yes-queries)",
           "difference of BPI to optimal BPI", "ratio of BPI to optimal BPI"]

headers2 = ["Filter", "Size in bytes", "yes-queries ratio",
            "BPI (average)", "opt BPI",
            "Difference", "Ratio"]

headers3 = ["Filter", "bytes size", "PR ratio",
            " BPI", "opt BPI", "Difference", " Ratio"]


def get_lines(path: str) -> list:
    # file_name = os.path.basename(dir)
    f = open(path, "r")
    lines = f.readlines()
    f.close()
    return lines


def parse_line(line: str) -> list:
    assert len(line)
    assert not line.startswith("#")
    # print(line)

    temp_data = line.split(",")
    temp_data1 = [i.strip() for i in temp_data]

    fl = []
    fl += [temp_data1[0], int(temp_data1[1])]
    fl += [float(i) for i in temp_data1[2:]]
    assert (len(fl) == len(headers))
    return fl


def beauty_table(my_table):
    names = ['CF-8',
             'CF-8-Flex',
             'CF-12',
             'CF-12-Flex',
             'CF-16',
             'CF-16-Flex',
             '\\midrule',
             'PF[CF-12-Flex]',
             'PF[TC]',
             'PF[Impala512]',
             '\\midrule',
             'BBF',
             'BBF-Flex',
             'Impala512',
             'Bloom-8[k=6]',
             'Bloom-12[k=8]',
             'Bloom-16[k=10]',
             '\\midrule',
             'TC']

    size = len(my_table)
    assert size >= len(names) - 3
    offset = 0

    for i in range(size):
        if names[i + offset] == '\\midrule':
            print("\\midrule")
            offset += 1

        if names[i + offset] != my_table[i][0].strip():
            print(names[i + offset], my_table[i][0].strip())
            assert 0

        line = "{:}\t & {:<.4f} & {:<.2f} & {:<.2f}& {:<.2f} & {:<.3f} \\\\"
        arg = names[i + offset], my_table[i][2]*100, *my_table[i][3:]
        print(line.format(*arg))


def main(path):
    lines = get_lines(path)
    table = []
    for line in lines:
        line = line.strip()
        to_skip = (len(line) == 0) or (line.startswith("#"))
        if to_skip:
            continue
        temp_fl = parse_line(line)
        table.append(temp_fl)

    beauty_table(table)
    return table


path = os.path.join(os.getcwd(), "fpp-pseudo.csv")
# path = "scripts/fpp-pseudo.csv"
res = main(path)
