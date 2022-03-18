from typing import *
from datetime import datetime

def get_marker_type(filter_name: str) -> str:
    filter_name = filter_name.strip()
    prefix_names_list = ["PF", "CF", "TC", "Bloom", "BBF", "Lem", "Impala512", "BF"]
    symbols = ["o", "s", "D", "D", "^", "X", '$\star$', "D"]
    default_symbol = "o"
    for i in range(len(prefix_names_list)):
        temp_prefix = prefix_names_list[i]
        if filter_name.startswith(temp_prefix):
            return symbols[i]

    return default_symbol


def name_dict(filter_name: str) -> str:
    filter_name = filter_name.strip()
    name_dict = {
        'BBF-Fixed': "BBF-Flex",
        'BBF_Fixed': "BBF-Flex",
        'Cuckoo-8':  'CF-8',
        'Cuckoo-12': 'CF-12',
        'Cuckoo-16': 'CF-16',
        'CuckooStable-8':  'CF-8-Flex',
        'CuckooStable-12': 'CF-12-Flex',
        'CuckooStable-16': 'CF-16-Flex',
        'SimdBlockFilter': "BBF",
        'TCD256': "TwoChoicer-256",
        'TC_V2': "TwoChoicer",
        'TC-shortcut': "TC",
        'TwoChoicer-dynamic': "TC-dyn",
        'VQF_Wrapper': "VQF",
        'inc2Choicer': "TwoChoicer",
        'Prefix-Filter [ evenFilter ]': "PF[AF-2]",
        'PF[BBF_Fixed]':  "PF[BBF-Flex]",
        'PF[BBF-Fixed]':  "PF[BBF-Flex]",
        'Prefix-Filter [ SimdBlockFilter ]':  "PF[BBF]",
        'Prefix-Filter [ Cuckoo-12 ]': 'PF[CF-12]',
        'Prefix-Filter [ CuckooStable-12 ]': 'PF[CF-12-Flex]',
        'PF[CF12-Flex]': 'PF[CF-12-Flex]',
        'Prefix-Filter [ TC-shortcut ]': 'PF[TC]',
        'Prefix-Filter [ BBF-Fixed ]': 'PF[BBF-Flex]',
        'Prefix-Filter [ Impala512 ]': 'PF[Impala]',
        'Bloom-8[k=6]': 'BF-8[k=6]',
        'Bloom-12[k=8]': 'BF-12[k=8]',
        'Bloom-16[k=11]': 'BF-16[k=11]',
        'Bloom-16[k=10]': 'BF-16[k=11]'
    }

    if filter_name in name_dict:
        return name_dict[filter_name]
    else:
        print(filter_name, "not in dict")
        return filter_name


def get_time():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
