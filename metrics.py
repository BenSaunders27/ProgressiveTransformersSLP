# coding: utf-8
"""
This module holds various MT evaluation metrics.
"""

from external_metrics import sacrebleu
from external_metrics import mscoco_rouge
import logging
import numpy as np

def wer(hypotheses, references):
    wer = del_rate = ins_rate = sub_rate = 0
    n_seq = len(hypotheses)

    for h, r in zip(hypotheses, references):
        res = wer_single(r=r, h=h)
        wer += res["wer"] / n_seq
        del_rate += res["del"] / n_seq
        ins_rate += res["ins"] / n_seq
        sub_rate += res["sub"] / n_seq

    return {"wer": wer, "del": del_rate, "ins": ins_rate, "sub": sub_rate}

def wer_single(r, h):
    edit_distance_matrix = editDistance(r=r, h=h)
    step_list = getStepList(r=r, h=h, d=edit_distance_matrix)

    min_distance = float(edit_distance_matrix[len(r)][len(h)])
    num_del = float(np.sum([s == "d" for s in step_list]))
    num_ins = float(np.sum([s == "i" for s in step_list]))
    num_sub = float(np.sum([s == "s" for s in step_list]))

    word_error_rate = round((min_distance / len(r) * 100), 4)
    del_rate = round((num_del / len(r) * 100), 4)
    ins_rate = round((num_ins / len(r) * 100), 4)
    sub_rate = round((num_sub / len(r) * 100), 4)

    return {"wer": word_error_rate, "del": del_rate, "ins": ins_rate, "sub": sub_rate}

# Calculate ROUGE scores
def rouge(hypotheses, references):
    rouge_score = 0
    n_seq = len(hypotheses)

    for h, r in zip(hypotheses, references):
        rouge_score += mscoco_rouge.calc_score(hypotheses=[h], references=[r]) / n_seq

    return rouge_score

# Calculate CHRF scores
def chrf(hypotheses, references):
    """
    Character F-score from sacrebleu

    :param hypotheses: list of hypotheses (strings)
    :param references: list of references (strings)
    :return:
    """
    return sacrebleu.corpus_chrf(hypotheses=hypotheses, references=references)

# Calculate BLEU scores
def bleu(hypotheses, references, all=False):
    """
    Raw corpus BLEU from sacrebleu (without tokenization)

    :param hypotheses: list of hypotheses (strings)
    :param references: list of references (strings)
    :return:
    """
    logger = logging.getLogger(__name__)
    bleu_scores = sacrebleu.raw_corpus_bleu(
        sys_stream=hypotheses, ref_streams=[references]
    ).scores
    scores = {}
    for n in range(len(bleu_scores)):
        scores["bleu" + str(n + 1)] = bleu_scores[n]

    rouge_score = rouge(hypotheses, references) * 100
    #wer_d = wer(hypotheses, references)
    #werStr = "WER: " +  str(wer_d["wer"])

    bleu1Str = "BLEU-1: " + str(scores["bleu1"])
    bleu2Str = "BLEU-2: " + str(scores["bleu2"])
    bleu3Str = "BLEU-3: " + str(scores["bleu3"])
    bleu4Str = "BLEU-4: " + str(scores["bleu4"])
    rougeStr = "Rouge: " + str(rouge_score)

    logger.info(bleu1Str)
    logger.info(bleu2Str)
    logger.info(bleu3Str)
    logger.info(bleu4Str)
    logger.info(rougeStr)

    print("--------------- New Scores ------------")
    print(bleu1Str)
    print(bleu2Str)
    print(bleu3Str)
    print(bleu4Str)
    print(rougeStr)
    #print(werStr)

    return bleu_scores[3]