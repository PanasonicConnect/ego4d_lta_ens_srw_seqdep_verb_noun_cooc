import argparse
import json
import math
import numpy as np
import os


def pxy(matrix, x, y):
    return matrix[x, y] / np.sum(matrix)
        
def px(matrix, x):
    return np.sum(matrix[x, :]) / np.sum(matrix)

def py(matrix, y):
    return np.sum(matrix[:, y]) / np.sum(matrix)

def calc_npmi(matrix, class_num):
    npmi_matrix = np.zeros((class_num + 1, class_num))
    for x in range(class_num + 1):
        for y in range(class_num):
            if pxy(matrix, x, y) == 0 or px(matrix, x) == 0 or py(matrix, y) == 0:
                npmi_matrix[x, y] = -1
            else:
                npmi_matrix[x, y] = math.log(pxy(matrix, x, y) / (px(matrix, x) * py(matrix, y))) / (-1 * math.log(pxy(matrix, x, y)))

    return npmi_matrix

def main():
    parser = argparse.ArgumentParser(description='Generate ')
    parser.add_argument(
        '--root_dir',
        help='root dirctory path which contains annotation files (fho_lta_taxonomy.json, fho_lta_train.json, fho_lta_val.json)',
        type=str)
    parser.add_argument(
        '--output_dir',
        help='output dirctory path which stastical data is saved (npmi_verb.csv, npmi_noun.csv, action_freq)',
        type=str)
    args = parser.parse_args()

    with open(os.path.join(args.root_dir, 'fho_lta_taxonomy.json'), 'r') as f:
        labels = json.load(f)

    verb_list = labels['verbs']
    noun_list = labels['nouns']

    verb_num = len(verb_list)
    noun_num = len(noun_list)

    verb_matrix = np.zeros((verb_num+1, verb_num))
    noun_matrix = np.zeros((noun_num+1, noun_num))
    action_matrix = np.zeros((verb_num, noun_num))

    with open(os.path.join(args.root_dir, 'fho_lta_train.json'), 'r') as f:
        data_train = json.load(f)
    with open(os.path.join(args.root_dir, 'fho_lta_val.json'), 'r') as f:
        data_val = json.load(f)

    known_clip_list = []

    for i in data_train['clips'] + data_val['clips']:
        if i["clip_uid"] not in known_clip_list:
            known_clip_list.append(i["clip_uid"])
            prev_verb = i['verb']
            prev_noun = i['noun']
            verb_matrix[-1, verb_list.index(i['verb'])] += 1
            noun_matrix[-1, noun_list.index(i['noun'])] += 1
            continue
        verb_matrix[verb_list.index(prev_verb), verb_list.index(i['verb'])] += 1
        noun_matrix[noun_list.index(prev_noun), noun_list.index(i['noun'])] += 1
        action_matrix[verb_list.index(i['verb']), noun_list.index(i['noun'])] += 1

    verb_npmi = calc_npmi(verb_matrix, verb_num)
    noun_npmi = calc_npmi(noun_matrix, noun_num)

    np.savetxt(os.path.join(args.output_dir, 'npmi_verb.csv'), verb_npmi, delimiter=",", fmt='%f')
    np.savetxt(os.path.join(args.output_dir, 'npmi_noun.csv'), noun_npmi, delimiter=",", fmt='%f')

    column_sums = np.sum(action_matrix, axis=0)
    normalized_action_matrix = np.nan_to_num(action_matrix / np.tile(column_sums, (verb_num, 1)))

    np.savetxt(os.path.join(args.output_dir, 'action_freq.csv.json'), normalized_action_matrix, delimiter=",", fmt='%f')


if __name__ == '__main__':
    main()
