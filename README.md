# Progressive Transformers for End-to-End Sign Language Production

Source code for "Progressive Transformers for End-to-End Sign Language Production" (Ben Saunders, Necati Cihan Camgoz, Richard Bowden) 

Paper published at ECCV 2020, available at https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123560664.pdf


# Usage

Prepare Phoenix (or other sign language dataset) data as .txt files for .skel, .gloss, .txt and .files. Data format should be parallel .txt files for src, trg and files. Each line of the .txt file represents a new sequence, in the same order in each file. src file should contain a new source input on each line
trg file should contain skeleton data, with each line a new sequence, each frame following on from the previous. Each joint value should be separated by a space; " ". Each frame is partioned using the known trg_size length, which includes all joints (In 2D or 3D) and the counter. Files file should contain the name of each sequence on a new line. Examples in /Data/tmp. Data path must be specified in config file.

To run, start __main__.py with arguments "train" and ".\Configs\Base.yaml":

`python __main__.py train ./Configs/Base.yaml` 

Back Translation model created from https://github.com/neccam/slt. Back Translation evaluation code coming soon.

# Abstract

The goal of automatic Sign Language Production (SLP) is to translate spoken language to a continuous stream of sign language video at a level comparable to a human translator. If this was achievable, then it would revolutionise Deaf hearing communications. Previous work on predominantly isolated SLP has shown the need for architectures that are better suited to the continuous domain of full sign sequences.

In this paper, we propose Progressive Transformers, a novel architecture that can translate from discrete spoken language sentences to continuous 3D skeleton pose outputs representing sign language. We present two model configurations, an end-to-end network that produces sign direct from text and a stacked network that utilises a gloss intermediary.

Our transformer network architecture introduces a counter that enables continuous sequence generation at training and inference. We also provide several data augmentation processes to overcome the problem of drift and improve the performance of SLP models. We propose a back translation evaluation mechanism for SLP, presenting benchmark quantitative results on the challenging RWTH-PHOENIX-Weather-2014T(PHOENIX14T) dataset and setting baselines for future research.


# Reference

If you wish to reference Progressive Transformers in a publication or thesis, please cite the following [paper](https://arxiv.org/abs/2004.14874):

```
@article{saunders2020progressive,
  title={Progressive Transformers for End-to-End Sign Language Production},
  author={Saunders, Ben and Camgoz, Necati Cihan and Bowden, Richard},
  journal={arXiv preprint arXiv:2004.14874},
  year={2020}
}
```

## Acknowledgements
<sub>This work received funding from the SNSF Sinergia project 'SMILE' (CRSII2 160811), the European Union's Horizon2020 research and innovation programme under grant agreement no. 762021 'Content4All' and the EPSRC project 'ExTOL' (EP/R03298X/1). This work reflects only the authors view and the Commission is not responsible for any use that may be made of the information it contains. We would also like to thank NVIDIA Corporation for their GPU grant. </sub>
