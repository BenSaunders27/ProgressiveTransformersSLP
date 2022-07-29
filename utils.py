import glob
import gzip
import json
import os
import pickle
import re
from einops import rearrange

import numpy as np
import torch
import yaml

from external_metrics import sacrebleu


def load_data(fpath):
    with gzip.open(fpath, 'rb') as f:
        file = pickle.load(f)
    return file


def bleu(references, hypotheses):
    """
    Raw corpus BLEU from sacrebleu (without tokenization)

    :param hypotheses: list of hypotheses (strings)
    :param references: list of references (strings)
    :return:
    """
    bleu_scores = sacrebleu.raw_corpus_bleu(
        sys_stream=hypotheses, ref_streams=[references]
    ).scores
    scores = {}
    for n in range(len(bleu_scores)):
        scores["bleu" + str(n + 1)] = bleu_scores[n]
    return scores


def load_joint(json_file_list):
    POSE_IDX = 8 # Upper body pose
    joint_list = []
    for l in json_file_list:
        json_file = l['people'][0]
        pose = json_file['pose_keypoints_2d']
        del pose[2::3]
        del pose[POSE_IDX*2:]
        landmark = json_file['face_keypoints_2d']
        del landmark[2::3]
        right_hand = json_file['hand_right_keypoints_2d']
        del right_hand[2::3]
        left_hand = json_file['hand_left_keypoints_2d']
        del left_hand[2::3]
        
        joint_list += [pose + right_hand + left_hand + landmark]
    
    return np.array(joint_list)


def load_yaml(fpath):
    with open(fpath) as f:
        # config = yaml.load(f)
        config = yaml.safe_load(f)    
    return config


def load_dirs(fpath):
    _dirs = sorted(glob.glob(fpath), key=lambda x: int(re.findall(r'\d+', x)[-1]))
    return _dirs


def load_json(fpath):
    with open(fpath, 'r') as f:
        _file = json.load(f)
    return _file


def noised(inputs, noise_rate):
    noise = inputs.data.new(inputs.size()).normal_(0, 1)
    noised_inputs = inputs + noise * noise_rate 
    
    return noised_inputs


def postprocess(joint_feat, H, W, scale = 2.):
    t, v, c = joint_feat.size()

    joint_feat[:, :, 0] *= (W * scale)
    joint_feat[:, :, 1] *= (H * scale)

    center = torch.ones(joint_feat[:, 1, :].size())
    center[:, 0] *= (W//2)
    center[:, 1] *= (H//2)
    # center[:, 1] *= (H//2 - (H*0.1))

    joint_feat -= (joint_feat[:, 1, :] - center).unsqueeze(1).repeat(1, v, 1)

    xs = joint_feat[:, :, 0].clone()
    ys = joint_feat[:, :, 1].clone()

    return torch.cat((xs, ys), dim = -1)


def build_adj(joint_num, skeleton):
    adj_matrix = np.zeros((joint_num, joint_num))
    for line in skeleton:
        adj_matrix[line] = 1
        adj_matrix[line[1], line[0]] = 1
    return adj_matrix + np.eye(joint_num)


def make_dirs(fpath):
    if not os.path.exists(fpath):
        os.makedirs(fpath)