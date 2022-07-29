import os
import math
import cv2
import numpy as np
from torch import Tensor

# ============================================================================
#                                 SKELETON
# ============================================================================

POINT_PAIR = [
    # body
    [0, 1], 
    [1, 2], 
    [2, 3],
    [3, 4],
    [1, 5],
    [5, 6],
    [6, 7], 
    # left hand
    # [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14]
    # right hand
]

POS_POINT_PAIR = [
    # body
    [0, 0], 
    [1, 2], 
    [2, 3],
    [3, 4],
    [1, 5],
    [5, 6],
    [6, 7], 
]

HAND_PAIR = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 4],

    [0, 5],
    [5, 6],
    [6, 7],
    [7, 8],

    [0, 9],
    [9, 10],
    [10, 11],
    [11, 12],

    [0, 13],
    [13, 14],
    [14, 15],
    [15, 16],

    [0, 17],
    [17, 18],
    [18, 19],
    [19, 20],
]

FACE_LINES = [
    (0, 16, False), # jaw
    (17, 21, False), # right eyebrow
    (22, 26, False), # left eyebrow
    (27, 30, False), # nose top-bottom line
    (31, 35, False), # nose right-left line
    (36, 41, True), # right eye, loop
    (42, 47, True), # left eye, loop
    (48, 59, True), # mouth outer line, loop
    (60, 67, True), # mouth inner line, loop
]

COLORS = [
    (0, 0, 0),
    (255, 0, 0), 
    (0, 255, 0), 
    (0, 0, 255), 
    (255,0,255),
    (255,255,0)
]


def draw_skeleton(sign_array, H = 256, W = 256):
    background = np.zeros((H, W, 3), np.uint8) + 255
    # H, W, C = background.shape
    
    dim = sign_array.shape[0]
    
    xs = sign_array[:dim//2]
    ys = sign_array[dim//2:]

    # pose_xs = xs[:50] * W
    # pose_ys = ys[:50] * H
    # landmark_xs = xs[50:] * W
    # landmark_ys = ys[50:] * H

    pose_xs = xs[:50]
    pose_ys = ys[:50]
    landmark_xs = xs[50:]
    landmark_ys = ys[50:]

    '''
    Draw body
    '''
    # draw pose    
    pos_idx_len = len(POS_POINT_PAIR) + 1
    body_xs = pose_xs[:pos_idx_len]
    body_ys = pose_ys[:pos_idx_len]
    for i, pair in enumerate(POS_POINT_PAIR):
        start = (int(body_xs[pair[0]]), int(body_ys[pair[0]]))
        end = (int(body_xs[pair[1]]), int(body_ys[pair[1]]))
        if i != 0:
            cv2.line(background, start, end, COLORS[0], 1)

    # draw right hand line
    rhand_idx_len = len(HAND_PAIR) + 1
    right_hand_xs = pose_xs[pos_idx_len:pos_idx_len + rhand_idx_len]
    right_hand_ys = pose_ys[pos_idx_len:pos_idx_len + rhand_idx_len]
    for idx, pair in enumerate(HAND_PAIR):
        start = (int(right_hand_xs[pair[0]]), int(right_hand_ys[pair[0]]))
        end = (int(right_hand_xs[pair[1]]), int(right_hand_ys[pair[1]]))
        cv2.line(background, start, end, COLORS[idx//4 + 1], 1)
    
    # draw left hand line
    lhand_idx_len = len(HAND_PAIR) + 1
    left_hand_xs = pose_xs[pos_idx_len + rhand_idx_len:pos_idx_len + rhand_idx_len + lhand_idx_len]
    left_hand_ys = pose_ys[pos_idx_len + rhand_idx_len:pos_idx_len + rhand_idx_len + lhand_idx_len]
    for idx, pair in enumerate(HAND_PAIR):
        start = (int(left_hand_xs[pair[0]]), int(left_hand_ys[pair[0]]))
        end = (int(left_hand_xs[pair[1]]), int(left_hand_ys[pair[1]]))
        cv2.line(background, start, end, COLORS[idx//4 + 1], 1)

    '''
    Draw landmark
    '''
    for start, end, loop in FACE_LINES:
        for i in range(start, end):
            points = [(int(landmark_xs[i]), int(landmark_ys[i])), 
                        (int(landmark_xs[i+1]), int(landmark_ys[i+1]))]
            cv2.line(background, points[0], points[1], COLORS[0], 1)        
        if loop:
            points = [(int(landmark_xs[end]), int(landmark_ys[end])), 
                        (int(landmark_xs[start]), int(landmark_ys[start]))]
            cv2.line(background, points[0], points[1], COLORS[0], 1)

    return background

def save_sign_video(
    fpath, 
    hyp, 
    ref, 
    sent, 
    H, 
    W, 
    fps = 25
):
    assert len(hyp.shape) == 2, 'Input shape must be 2D (t x d)'
    
    if type(hyp) == Tensor:
        hyp = hyp.numpy()
        ref = ref.numpy()

    pad_seq = hyp.shape[0] - ref.shape[0]
    
    # make them same length
    if pad_seq > 0:
        ref = np.concatenate((ref, np.zeros((pad_seq, ref.shape[1]), dtype = np.float32)))
    else:
        pad_seq = abs(pad_seq)
        hyp = np.concatenate((hyp, np.zeros((pad_seq, hyp.shape[1]), dtype = np.float32)))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(fpath, fourcc, float(fps), (H * 2, W), True)
    
    for i, (h, r) in enumerate(zip(hyp, ref)):
        # sign frame
        h_frame = draw_skeleton(h, H, W)
        r_frame = draw_skeleton(r, H, W)
        frame = np.concatenate((h_frame, r_frame), axis = 1)
        
        font = font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        thickness = 1
        
        cv2.putText(frame, 'Generated', (int(W * 0.3), int(H * 0.1)), 
                    font, fontScale, (255, 255, 0), thickness)
        cv2.putText(frame, 'Ground Truth', (int(W * 1.3), int(H * 0.1)), 
                    font, fontScale, (255, 255, 0), thickness)
        
        chunk = 40
        num_chunk = len(sent) // chunk
        if num_chunk > 1:
            start = 0
            for i in range(num_chunk):
                end = start + chunk
                words = sent[start : end]
                cv2.putText(frame, words, (int(W * 0.1), int(H * 0.8 + i * H * 0.1)),
                            font, fontScale, (0, 0, 0), thickness)
                start = end
        else:
            cv2.putText(frame, sent, (int(W * 0.1), int(H * 0.9)),
                        font, fontScale, (0, 0, 0), thickness)
            
        video.write(frame)
    
    video.release()


def save_sign_video_batch(
    fpath, 
    hyp_batch, 
    ref_batch, 
    sent_batch, 
    id_batch, 
    H,
    W,
    fps = 25
):
    if not os.path.exists(fpath):
        os.makedirs(fpath)
    
    assert len(hyp_batch.shape) == 3, 'Input must be 3D.'
    
    b, t, d = hyp_batch.size()
    for i in range(b):
        hyp = hyp_batch[i]
        ref = ref_batch[i]
        sent = sent_batch[i]
        id = id_batch[i]
        
        save_sign_video(
            fpath = os.path.join(fpath, f'{id}.mp4'),
            hyp = hyp,
            ref = ref,
            sent = sent,
            H = H,
            W = W
        )
