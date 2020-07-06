# Progressive Transformers for End-to-End Sign Language Production

Source code for "Progressive Transformers for End-to-End Sign Language Production" (Ben Saunders, Necati Cihan Camgoz, Richard Bowden) 

Paper accepted for publication at ECCV 2020, pre-print available at https://arxiv.org/abs/2004.14874




Source code currently under review, and shall be available in time for ECCV20 conference.


# Abstract

The goal of automatic Sign Language Production (SLP) is to translate spoken language to a continuous stream of sign language video at a level comparable to a human translator. If this was achievable, then it would revolutionise Deaf hearing communications. Previous work on predominantly isolated SLP has shown the need for architectures that are better suited to the continuous domain of full sign sequences.

In this paper, we propose Progressive Transformers, a novel architecture that can translate from discrete spoken language sentences to continuous 3D skeleton pose outputs representing sign language. We present two model configurations, an end-to-end network that produces sign direct from text and a stacked network that utilises a gloss intermediary.

Our transformer network architecture introduces a counter that enables continuous sequence generation at training and inference. We also provide several data augmentation processes to overcome the problem of drift and improve the performance of SLP models. We propose a back translation evaluation mechanism for SLP, presenting benchmark quantitative results on the challenging RWTH-PHOENIX-Weather-2014T(PHOENIX14T) dataset and setting baselines for future research.


# Reference

If you with to reference Progressive Transformers in a publication or thesis, please cite the following [paper](https://arxiv.org/abs/2004.14874):

```
@article{saunders2020progressive,
  title={Progressive Transformers for End-to-End Sign Language Production},
  author={Saunders, Ben and Camgoz, Necati Cihan and Bowden, Richard},
  journal={arXiv preprint arXiv:2004.14874},
  year={2020}
}
```
