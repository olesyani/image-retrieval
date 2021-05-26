import sys

from typing import List


def load_list(fname: str):
    return [e.strip() for e in open(fname, 'r').readlines()]


def compute_ap(positive, negative, ranked_list):
    intersect_size, old_recall, ap = 0.0, 0.0, 0.0
    old_precision, j = 1.0, 1.0
    
    for img in ranked_list:
        if img in negative:
            continue
            
        if img in positive:
            intersect_size += 1.0

        recall = intersect_size / len(positive)
        precision = intersect_size / j
        ap += (recall - old_recall) * ((old_precision + precision) / 2.0)

        old_recall = recall
        old_precision = precision
        j += 1.0

    return ap


def compute(arg):
    ranked_list_file = 'ranked_list.txt'
    
    try:
        ranked_list = load_list(ranked_list_file)
        pos_set = list(set(load_list("%s_good.txt" % arg) + load_list("%s_ok.txt" % arg)))
        junk_set = load_list("%s_junk.txt" % arg)

        return str(compute_ap(pos_set, junk_set, ranked_list))
    
    except IOError as e:
        print("IO error while opening files. %s" % e)
        sys.exit(1)
