import argparse
import json
from pathlib import Path
import shutil
from tqdm import tqdm

import torch
from torch.distributions.categorical import Categorical
torch.manual_seed(0)

import numpy as np
import os
import sys

K = 5
Z = 20

def generate(x, k=1):
    results = []
    for head_x in x:
        if k>1:
            preds_dist = Categorical(logits=head_x)
            preds = [preds_dist.sample() for _ in range(k)]
        elif k==1:
            preds = [head_x.argmax(2)]
        head_x = torch.stack(preds, dim=1)
        results.append(head_x)

    return results

def generate_npmi(x, k=1):
    results = []
    # Noun
    head_x = x[1] # [batch=1, Z, 521]
    preds_list = []
    for batch, head in enumerate(head_x):
        preds = []
        for step in range(Z):
            preds_pattern = []
            for pattern in range(k):
                probs = torch.softmax(head[step], dim=-1)
                if step != 0:
                    probs *= npmi_noun[prev[pattern]]
                if k == 0:
                    preds_pattern.append(head[step].argmax())
                elif k == 1:
                    preds_pattern.append(probs.argmax())
                else:
                    preds_dist = Categorical(probs)
                    preds_pattern.append(preds_dist.sample().detach().cpu())
            prev = preds_pattern # [5]
            preds.append(preds_pattern)
        preds_list.append(preds)
    head_x = torch.tensor(preds_list).permute((0, 2, 1)) # [batch, Z, K] -> [batch, K, Z]
    results.append(head_x)
    pred_nouns = preds_list

    # Verb
    head_x = x[0] # [batch=1, Z, 117]
    preds_list = []
    for batch, head in enumerate(head_x):
        preds = []
        for step in range(Z):
            preds_pattern = []
            for pattern in range(k):
                probs = torch.softmax(head[step], dim=-1)
                probs *= action_freq[:, pred_nouns[batch][step][pattern]]
                if step != 0:
                    probs *= npmi_verb[prev[pattern]]
                if k == 0:
                    preds_pattern.append(head[step].argmax())
                elif k == 1:
                    preds_pattern.append(probs.argmax())
                else:
                    preds_dist = Categorical(probs)
                    preds_pattern.append(preds_dist.sample().detach().cpu())
            prev = preds_pattern # [5]
            preds.append(preds_pattern)
        preds_list.append(preds)
    head_x = torch.tensor(preds_list).permute((0, 2, 1)) # [batch, Z, K] -> [batch, K, Z]
    results.append(head_x)

    return [results[1], results[0]]

def calc(methods, args):
    tmp_dir = Path('./tmp')
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for method in methods:
        for i, file in enumerate(Path(os.path.join(args.root_dir, method)).rglob('outputs_logits.json')):
            tmp_save_dir = tmp_dir / f'{method}'
            tmp_save_dir.mkdir(parents=True, exist_ok=True)

            with open(file, 'r') as f:
                logits = json.load(f)
            for k, v in tqdm(logits.items()):
                tmp_file = tmp_save_dir / f'{k}.json'
                with open(tmp_file, 'w') as f:
                    json.dump(v, f)

    all_preds = {}
    keys = set([p.stem for p in tmp_dir.glob('*/*.json')])

    for key in keys:
        file_path = []
        for method in methods:
            file_path += tmp_dir.glob(f'{method}/{key}.json')

        verb_logits_list = []
        noun_logits_list = []

        for file in file_path:
            with open(file, 'r') as f:
                logits = json.load(f)
            
            if file.parts[-2] == 'SlowFast_Concat_input8':
                verb_logits_list.append(torch.Tensor(logits['verb']) * args.alpha)
                noun_logits_list.append(torch.Tensor(logits['noun']) * args.alpha)
            elif file.parts[-2] == 'SlowFast-CLIP_Transformer':
                verb_logits_list.append(torch.Tensor(logits['verb']) * args.beta)
                verb_logits_list.append(torch.Tensor(logits['verb']) * args.beta)

        verb_logits = torch.sum(torch.stack(verb_logits_list), dim=0)
        noun_logits = torch.sum(torch.stack(noun_logits_list), dim=0)

        if args.npmi:
            preds = generate_npmi([torch.unsqueeze(verb_logits, 0), torch.unsqueeze(noun_logits, 0)], k=K)
        else:
            preds = generate([torch.unsqueeze(verb_logits, 0), torch.unsqueeze(noun_logits, 0)], k=K)
        all_preds[key] = {'verb': torch.squeeze(preds[0], 0).tolist(), 'noun': torch.squeeze(preds[1], 0).tolist()}

    # Add dummy outputs (zero padding) for missing test data for submission.
    if args.dataset == 'test':
        for last_visible_action_idx in range(8, 33+1):
            clip_uid = '814baefc-d043-40f8-bb52-3573266f613f'
            all_preds[f'{clip_uid}_{last_visible_action_idx}'] = {'verb': [[0]*Z]*K, 'noun': [[0]*Z]*K}

    json.dump(all_preds, open('outputs.json', 'w'))

    shutil.rmtree(tmp_dir)


def main():
    parser = argparse.ArgumentParser(description='Ensemble model logits.')
    parser.add_argument(
        '--root_dir',
        help='root dirctory path which contains logits files (outputs_logits.json)',
        type=str)
    parser.add_argument(
        '--data_dir',
        help='data dirctory path which contains stastical data (npmi_verb.csv, npmi_noun.csv, action_freq)',
        type=str)
    parser.add_argument(
        '--npmi',
        help='If true, reflect statistical data',
        action='store_true')
    parser.add_argument(
        '--dataset',
        help='data dirctory path which contains stastical data (npmi_verb.csv, npmi_noun.csv, action_freq)',
        default='val',
        type=str)
    parser.add_argument(
        '--alpha',
        default=0.6,
        type=float)
    parser.add_argument(
        '--beta',
        default=1.4,
        type=float)
    args = parser.parse_args()

    global npmi_verb, npmi_noun, action_freq
    npmi_verb = torch.from_numpy(np.loadtxt(os.path.join(args.data_dir, 'npmi_verb.csv'), delimiter=',', dtype=float))
    npmi_noun = torch.from_numpy(np.loadtxt(os.path.join(args.data_dir, 'npmi_noun.csv'), delimiter=',', dtype=float))

    npmi_verb = torch.relu(npmi_verb) + sys.float_info.epsilon
    npmi_noun = torch.relu(npmi_noun) + sys.float_info.epsilon

    action_freq = torch.from_numpy(np.loadtxt(os.path.join(args.data_dir, 'action_freq.csv'), delimiter=',', dtype=float)) + sys.float_info.epsilon

    calc(('SlowFast_Concat_input8', 'SlowFast-CLIP_Transformer', ), args)

if __name__ == '__main__':
    main()
