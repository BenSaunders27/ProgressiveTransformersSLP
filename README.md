# Progressive Transformers for End-to-End Sign Language Production

Source code for "Progressive Transformers for End-to-End Sign Language Production" (Ben Saunders, Necati Cihan Camgoz, Richard Bowden - ECCV 2020) 

Conference video available at https://twitter.com/BenMSaunders/status/1336638886198521857

# Usage

Install required packages using the requirements.txt file.

`pip install -r requirements.txt`

To run, start __main__.py with arguments "train" and ".\Configs\Base.yaml":

`python __main__.py train ./Configs/Base.yaml` 

An example train.log file can be found in ".\Configs\train.log" and a validation file at ".\Configs\validations.txt"

Back Translation model created from https://github.com/neccam/slt. Back Translation evaluation code coming soon.

# Data

Pre-processed Phoenix14T data can be requested via email at b.saunders@surrey.ac.uk. If you wish to create the data yourself, please follow below:

Phoenix14T data can be downloaded from https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/ and skeleton joints can be extracted using OpenPose at  https://github.com/CMU-Perceptual-Computing-Lab/openpose and lifted to 3D using the 2D to 3D Inverse Kinematics code at https://github.com/gopeith/SignLanguageProcessing under 3DposeEstimator.

Prepare Phoenix14T (or other sign language dataset) data as .txt files for .skel, .gloss, .txt and .files. Data format should be parallel .txt files for "src", "trg" and "files", with each line representing a new sequence:

- The "src" file contains source sentences, with each line representing new sentence. 

- The "trg" file contains skeleton data of each frame, with a space separating frames. The joints should be divided by 3 to match the scaling I used. Each frame contains 150 joint values and a subsequent counter value, all separated by a space. Each sequence should be separated with a new line. If your data contains 150 joints per frame, please ensure that trg_size is set to 150 in the config file. 

- The "files" file should contain the name of each sequence on a new line. 

Examples can be found in /Data/tmp. Data path must be specified in config file.

# Reference

If you use this code in your research, please cite the following [papers](https://arxiv.org/abs/2004.14874):

```
@inproceedings{saunders2020progressive,
	title		=	{{Progressive Transformers for End-to-End Sign Language Production}},
	author		=	{Saunders, Ben and Camgoz, Necati Cihan and Bowden, Richard},
	booktitle   	=   	{Proceedings of the European Conference on Computer Vision (ECCV)},
	year		=	{2020}}

@inproceedings{saunders2020adversarial,
	title		=	{{Adversarial Training for Multi-Channel Sign Language Production}},
	author		=	{Saunders, Ben and Camgoz, Necati Cihan and Bowden, Richard},
	booktitle   	=   	{Proceedings of the British Machine Vision Conference (BMVC)},
	year		=	{2020}}

@inproceedings{saunders2021continuous,
	title		=	{{Continuous 3D Multi-Channel Sign Language Production via Progressive Transformers and Mixture Density Networks}},
	author		=	{Saunders, Ben and Camgoz, Necati Cihan and Bowden, Richard},
	booktitle   	=   	{International Journal of Computer Vision (IJCV)},
	year		=	{2021}}

```

## Acknowledgements
<sub>This work received funding from the SNSF Sinergia project 'SMILE' (CRSII2 160811), the European Union's Horizon2020 research and innovation programme under grant agreement no. 762021 'Content4All' and the EPSRC project 'ExTOL' (EP/R03298X/1). This work reflects only the authors view and the Commission is not responsible for any use that may be made of the information it contains. We would also like to thank NVIDIA Corporation for their GPU grant. </sub>
