import glob
import gzip
import os
import pickle
import random
from einops import rearrange

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
# from torchtext import data

from utils import load_dirs, load_json


# # pylint: disable=unused-argument,global-variable-undefined
# def token_batch_size_fn(new, count, sofar):
#     """Compute batch size based on number of tokens (+padding)."""
#     global max_src_in_batch, max_tgt_in_batch
#     if count == 1:
#         max_src_in_batch = 0
#         max_tgt_in_batch = 0
#     max_src_in_batch = max(max_src_in_batch, len(new.src))
#     src_elements = count * max_src_in_batch
#     if hasattr(new, 'trg'):  # for monolingual data sets ("translate" mode)
#         max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)
#         tgt_elements = count * max_tgt_in_batch
#     else:
#         tgt_elements = 0
#     return max(src_elements, tgt_elements)
#
#
# def _collate_fn(batch):
#     return torch.nn.utils.rnn.pad_sequence([x['joint_feats'] for x in batch], batch_first=True, padding_value=0)
#
#
# def make_data_iter(dataset: Dataset,
#                    batch_size: int,
#                    batch_type: str = "sentence",
#                    train: bool = False,
#                    shuffle: bool = False):
#     """
#     Returns a torchtext iterator for a torchtext dataset.
#     :param dataset: torchtext dataset containing src and optionally trg
#     :param batch_size: size of the batches the iterator prepares
#     :param batch_type: measure batch size by sentence count or by token count
#     :param train: whether it's training time, when turned off,
#         bucketing, sorting within batches and shuffling is disabled
#     :param shuffle: whether to shuffle the data before each epoch
#         (no effect if set to True for testing)
#     :return: torchtext iterator
#     """
#
#     batch_size_fn = token_batch_size_fn if batch_type == "token" else None
#
#     if train:
#         # optionally shuffle and sort during training
#         data_iter = data.BucketIterator(
#             repeat=False, sort=False, dataset=dataset,
#             batch_size=batch_size, batch_size_fn=batch_size_fn,
#             train=True, sort_within_batch=True,
#             sort_key=lambda x: len(x.src), shuffle=shuffle)
#     else:
#         # don't sort/shuffle for validation/inference
#         # data_iter = data.BucketIterator(
#         #     repeat=False, dataset=dataset,
#         #     batch_size=batch_size, batch_size_fn=batch_size_fn,
#         #     train=False, sort=False)
#         data_iter = DataLoader(dataset, batch_size, shuffle=False, collate_fn=_collate_fn)
#
#     return data_iter


def get_random_seq(seq, seq_len):
    start = random.randrange(0, len(seq) + 1 - seq_len)
    end = start + seq_len
    return seq[start : end]


def load_data(
    dataset_type, 
    train_trans_path = None, 
    valid_trans_path = None, 
    test_trans_path = None, 
    seq_len = -1, 
    min_seq_len = -1,
    normalize = True
):
    assert dataset_type in ['phoenix', 'how2sign'], 'dataset must be selected between phoenix or how2sign.'
    
    if dataset_type == 'how2sign':
        root, _ = os.path.split(train_trans_path)

        # define joint paths
        tr_joint_path = os.path.join(root, 'train_2D_keypoints', 'json')
        val_joint_path = os.path.join(root, 'val_2D_keypoints', 'json')
        tst_joint_path = os.path.join(root, 'test_2D_keypoints', 'json')

        # tr_joint_feat_path = os.path.join(root, 'train_feat')
        # val_joint_feat_path = os.path.join(root, 'val_feat')
        # tst_joint_feat_path = os.path.join(root, 'val_feat')

        trainset = How2SignDataset(
            trans_path = train_trans_path, 
            joint_path = tr_joint_path, 
            seq_len = seq_len, 
            min_seq_len = min_seq_len,
            normalize = normalize
        )
        
        validset = How2SignDataset(
            trans_path = valid_trans_path, 
            joint_path = val_joint_path, 
            seq_len = seq_len, 
            min_seq_len = min_seq_len,
            normalize = normalize
        )

        testset = How2SignDataset(
            trans_path = test_trans_path, 
            joint_path = tst_joint_path, 
            seq_len = seq_len, 
            min_seq_len = min_seq_len,
            normalize = normalize
        )
    
    elif dataset_type == 'phoenix':
        trainset = Phoenix2014TDataset(
            fpath = train_trans_path, 
            min_seq_len = min_seq_len, 
            seq_len = seq_len,
            normalize = normalize
        )

        validset = Phoenix2014TDataset(
            fpath = valid_trans_path, 
            min_seq_len = min_seq_len, 
            seq_len = seq_len,
            normalize = normalize
        )
        
        testset = Phoenix2014TDataset(
            fpath = test_trans_path, 
            min_seq_len = min_seq_len, 
            seq_len = seq_len,
            normalize = normalize
        )
    else:
        raise NotImplementedError
    
    return trainset, validset, testset


class How2SignDataset(Dataset):
    def __init__(
        self, 
        trans_path, 
        joint_path,
        feat_path = None, 
        seq_len = -1, 
        min_seq_len = -1,
        normalize = True
    ):
        super().__init__()

        data = self._load_df(
            dataframe = pd.read_csv(trans_path, sep='\t'), 
            joint_path = joint_path, 
            min_seq_len = min_seq_len
        )

        if feat_path != None:
            num_processed = len(glob.glob(os.path.join(feat_path, '*')))
            assert len(data) == num_processed, f'These files must match: {len(data)} vs. {num_processed}.'
            
        self.max_seq_len = data.FRAME_LENGTH.max()
        self.min_seq_len = data.FRAME_LENGTH.min()
        
        self.seq_len = seq_len
        self.min_len = min_seq_len
        self.joint_path = joint_path
        self.trans_path = trans_path
        
        self.feat_path = feat_path

        self.data = data

        self.normalize = normalize
        
    def _load_df(self, dataframe, joint_path, min_seq_len):
        joint_dir_list = []
        frame_len_list = []
        
        for i in range(len(dataframe)):
            data = dataframe.iloc[i]
            skel_id = data['SENTENCE_NAME']
            skel_dir = os.path.join(joint_path, skel_id)
            frame_len = len(glob.glob(os.path.join(skel_dir, '*')))
            joint_dir_list.append(skel_dir)
            frame_len_list.append(frame_len)
        
        dataframe['JOINT_DIR'] = joint_dir_list
        dataframe['FRAME_LENGTH'] = frame_len_list

        if min_seq_len != -1:
            dataframe.drop(dataframe[dataframe.FRAME_LENGTH < min_seq_len].index, inplace = True)

        return dataframe

    def _load_skeleton(self, signs):
        POSE_IDX = 8 # Upper body pose
        skel_list = []
        
        for l in signs:
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
            
            skel_list += [pose + right_hand + left_hand + landmark]
        
        return torch.Tensor(skel_list)

    def __getitem__(self, index):
        data = self.data.iloc[index]
        
        id = data['SENTENCE_NAME']
        text = data['SENTENCE']
        joint_dir = data['JOINT_DIR']
        frame_len = data['FRAME_LENGTH']
        
        joint_dirs = load_dirs(os.path.join(joint_dir, '*'))
        joint_feats = self._load_skeleton([load_json(jd) for jd in joint_dirs])
        
        if self.seq_len != -1:
            joint_feats = get_random_seq(joint_feats, self.seq_len)
            frame_len = len(joint_feats)
        
        if self.feat_path != None:
            processed_joint_dirs = glob.glob(os.path.join(self.feat_path, id, '*'))
            processed_joint_dirs = sorted(processed_joint_dirs)

            joint_input_ids = torch.load(processed_joint_dirs[0])
            joint_input_logits = torch.load(processed_joint_dirs[1])
            joint_pad_mask = torch.load(processed_joint_dirs[2])
        else:
            joint_input_ids, joint_input_logits, joint_pad_mask = None, None, None
        
        joint_feats = rearrange(joint_feats, 't (v c) -> t v c', c = 2)
        t, v, c = joint_feats.size()
        
        if self.normalize:
            dist = (joint_feats[:, 2, :] - joint_feats[:, 5, :]).pow(2).sum(-1).sqrt()
            joint_feats /= (dist / 300).view(dist.size(0), 1, 1).repeat(1, v, c)
            
            center = torch.ones(joint_feats[:, 1, :].size())
            center[:, 0] *= (1400 // 2)
            center[:, 1] *= (1050 // 2)
            joint_feats -= (joint_feats[:, 1, :] - center).unsqueeze(1).repeat(1, v, 1)

            joint_feats[:, :, :1] /= (1400 * 2.5)
            joint_feats[:, :, 1:] /= (1050 * 2.5)

            joint_feats += 0.3 # numeric stable

        return {
            'id': id,
            'text': text,
            'joint_feats': joint_feats,
            'frame_len': frame_len,
            'joint_input_ids': joint_input_ids,
            'joint_pad_mask': joint_pad_mask,
            'joint_input_logits': joint_input_logits
        }
        
    def __len__(self):
        return len(self.data)


class Phoenix2014TDataset(Dataset):
    def __init__(
        self, 
        fpath, 
        min_seq_len = -1, 
        seq_len = -1,
        normalize = True
    ):
        super().__init__()

        self.seq_len = seq_len
        self.data = self._load_data(fpath)
        
        # filtered by a given length
        if min_seq_len != -1:
            self.data = [data for data in self.data if len(data['sign']) > min_seq_len]

        self.normalize = normalize

    def _load_data(self, fpath):
        with gzip.open(fpath, 'rb') as f:
            file = pickle.load(f)
        return file
    
    def __getitem__(self, index):
        data = self.data[index]

        _, id = os.path.split(data['name'])
        text = data['text']
        gloss = data['gloss']
        
        joint = data['sign']
        
        if self.seq_len != -1:
            joint = get_random_seq(joint, self.seq_len)
        
        joint = torch.Tensor(joint)
        
        joint = rearrange(joint, 't (v c) -> t v c', c = 2)
        
        pose = joint[:, :8, :]
        landmark = joint[:, 50:, :]
        
        landmark *= 0.4

        neck = pose[:, 0]
        nose = landmark[:, 30]

        diff = neck - nose
        diff = rearrange(diff, 't v -> t () v').repeat(1, landmark.size(1), 1)
        landmark += diff

        joint_feats = rearrange(joint, 't v c -> t v c')
        t, v, c = joint_feats.size()

        if self.normalize:
            dist = (joint_feats[:, 2, :] - joint_feats[:, 5, :]).pow(2).sum(-1).sqrt()
            joint_feats /= (dist / 0.3).view(dist.size(0), 1, 1).repeat(1, v, c)

            center = torch.ones(joint_feats[:, 1, :].size()) * 0.5
            joint_feats -= (joint_feats[:, 1, :] - center).unsqueeze(1).repeat(1, v, 1)

            joint_feats[:, :, :1] /= 1.6
            joint_feats[:, :, 1:] /= 1.6

            joint_feats += 0.1

        joint_input_ids, joint_input_logits, joint_pad_mask = None, None, None

        return {
            'id': id,
            'text': text,
            'gloss': gloss,
            'joint_feats': joint_feats,
            'frame_len': len(joint_feats),
            'joint_input_ids': joint_input_ids,
            'joint_pad_mask': joint_pad_mask,
            'joint_input_logits': joint_input_logits
        }

    def __len__(self):
        return len(self.data)

