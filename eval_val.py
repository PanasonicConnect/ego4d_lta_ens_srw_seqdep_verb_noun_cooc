import editdistance
import json
import numpy as np
import sys
import torch

"""
python eval_val.py {outputs.json}
Check the path of annotation_file
"""

Z = 20
NUM_VERB_CLASS = 117
NUM_NOUN_CLASS = 521
annotation_file = 'fho_lta_val.json'

def edit_distance(preds, labels):
    """
    Damerau–Levenshtein edit distance from: https://github.com/gfairchild/pyxDamerauLevenshtein
    Lowest among K predictions
    """
    Z, K = preds.shape
    dists = []
    dist = min([editdistance.eval(preds[:, k], labels)/Z for k in range(K)])
    dists.append(dist)
    return np.mean(dists)

def AUED(preds, labels):
    Z, K = preds.shape
    ED = np.vstack(
        [edit_distance(preds[:z], labels[:z]) for z in range(1, Z + 1)]
    )
    AUED = np.trapz(y=ED, axis=0) / (Z - 1)

    output = {"AUED": AUED}
    output.update({f"ED_{z}": ED[z] for z in range(Z)})
    return output

def main():
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    converted_dict = {}
    prev_clip_id = ''
    for annotation in annotations['clips']:

        if annotation['clip_uid'] != prev_clip_id:
            converted_dict[annotation['clip_uid']] = [[annotation['action_clip_start_frame'], annotation['action_clip_end_frame'], annotation['verb_label'], annotation['noun_label']]]
            prev_clip_id = annotation['clip_uid']
        else:
            converted_dict[annotation['clip_uid']].append([annotation['action_clip_start_frame'], annotation['action_clip_end_frame'], annotation['verb_label'], annotation['noun_label']])


    with open(sys.argv[1], 'r') as f:
        outputs = json.load(f)

    results_list = []
    for action_id, predictions in outputs.items():
        clip_id = action_id[:36]
        step = int(action_id[37:])
        label_verb = []
        label_noun = []
        label_action = []
        for gt in converted_dict[clip_id][step:step + Z]:
            label_verb.append(gt[2])
            label_noun.append(gt[3])
            label_action.append(gt[2] * NUM_NOUN_CLASS + gt[3])
        pred_action = np.array(predictions['verb']) * NUM_NOUN_CLASS + np.array(predictions['noun'])
        
        auedit_verb = AUED(np.array(predictions['verb']).transpose(), np.array(label_verb).transpose())
        auedit_noun = AUED(np.array(predictions['noun']).transpose(), np.array(label_noun).transpose())
        auedit_action = AUED(pred_action.transpose(), np.array(label_action))

        step_result = {}
        results = {
            f"verb_" + k: v for k, v in auedit_verb.items()
        }
        step_result.update(results)
        results = {
            f"noun_" + k: v for k, v in auedit_noun.items()
        }
        step_result.update(results)
        results = {
            f"action_" + k: v for k, v in auedit_action.items()
        }
        step_result.update(results)
        results_list.append(step_result)

    for key in ['verb_AUED', 'noun_AUED', 'action_AUED']:
        metric = torch.tensor(np.array([x[key] for x in results_list])).mean()
        print(key, metric.item())
    
if __name__ == '__main__':
    main()
