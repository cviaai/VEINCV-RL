[![Python](https://img.shields.io/badge/python-3.6-blue.svg)](https://python.org)
[![Python](https://img.shields.io/badge/pytorch-1.6.0-red)](https://pytorch.org)
[![Python](https://img.shields.io/badge/openai%20gym-0.2.3-blue)](https://openai.com/)
[![Python](https://img.shields.io/badge/paper-published-red)](https://ieeexplore.ieee.org/document/9305503)

# Near-Infrared-to-Visible Vein Imaging via Convolutional Neural Networks and Reinforcement Learning

This repository is an official PyTorch implementation of the paper "Near-Infrared-to-Visible Vein Imaging via Convolutional Neural Networks and Reinforcement  Learning", accepted to the 16th International Conference on Control, Automation, Robotics and Vision, ICARCV 2020, Shenzhen, China, December 13-15, 2020.

## Motivation

Peripheral Difficult Venous Access (PDVA) is a commonplace problem in clinical practice which results in repetitive punctures, damaged veins, and significant discomfort to the patients. Nowadays, the poor visibility of subcutaneous vasculature in the visible part of the light spectrum is overcome by near-infrared (NIR) imaging and a returned projection of the recognized vasculature back to the arm of the patient. Here we introduce a closed-loop hardware system that optimizes cross-talk between the virtual mask generated from the NIR measurement and the projected augmenting image through CNNs and RL.

Experimental setup            |  RL image adjustment (example)
:-------------------------:|:-------------------------:
<img src="https://github.com/cviaai/NIR-VISIBLE-IMAGING-WITH-CNN-RL/blob/master/img/Experimental_setup_scheme.png" width="600"></img> | <img src="https://github.com/cviaai/NIR-VISIBLE-IMAGING-WITH-CNN-RL/blob/master/img/example.gif" width="600"></img>

## Segmentation pipeline
![Segmentation pipeline](https://github.com/cviaai/NIR-VISIBLE-IMAGING-WITH-CNN-RL/blob/master/img/Segmentation_pipeline.png)

</p>
<p align="center">
<em> Fig. 2. Segmentation pipeline featuring Frangi vesselness filter, attention U-Net and clDICE loss </em><br>
</p>

## Installation as a project repository:

```
git clone https://github.com/cviaai/VEINCV-RL.git
```
In this case, you need to manually install the dependencies.

## Requirements
To install requirements:

```setup
pip install -r requirements.txt
```

## Dataset
Dataset contains 90 manually collected and annotated NIR images of forearms. Annotation was done with Frangi vesselness filter and [Computer Vision Annotation Tool (CVAT)](https://github.com/openvinotoolkit/cvat). 

We make our dataset publicly available for other researchers. If you use it in your research, please [cite us](https://github.com/cviaai/VEINCV-RL/blob/master/README.md#citing) 

## Training

To train the models used in the paper, run this command:

```python3
python main.py
```

## Code structure 
Folder ```/server``` - main folder with experiment files
* ```/dataset_90``` - 90 forearm snapshots and 90 corresponding masks, 75 of them for train, 15 for validation
* ```/ignite_new``` - segmentation and alignment experiments
* ```/img_check``` - 1 random training sample to check; 4 snapshots in it - original snapshot, ground true mask, predicted image, predicted mask (binarized predicted image)
```
.
├───dataset_90
├───ignite_new
│   ├───architectures
│   │   ├───attention_unet_architectures.py
│   │   ├───base_unet_architecture.py
│   │   └───encoders_unet_architectures.py
│   ├───pretrained_models
│   ├───VeinsDataset.py
│   ├───dice_helpers.py
│   ├───inference.ipynb
│   ├───iou.py
│   ├───loss.py
│   ├───main.py
│   ├───miou.py
│   ├───precision.py
│   ├───recall.py
│   ├───ssim.py
│   ├───train.py
│   ├───tversky_loss.py
│   └───utils.py
└───img_check
```

## Citing
If you use this package in your publications or in other work, please cite it as follows:
```
@INPROCEEDINGS{VeinCVNIR,  
  author={V. M. {Leli} and A. {Rubashevskii} and A. {Sarachakov} and O. {Rogov} and D. V. {Dylov}},  
  booktitle={2020 16th International Conference on Control, Automation, Robotics and Vision (ICARCV)},   
  title={Near-Infrared-to-Visible Vein Imaging via Convolutional Neural Networks and Reinforcement Learning},   
  year={2020},  
  volume={},  
  number={},  
  pages={434-441},  
  doi={10.1109/ICARCV50220.2020.9305503}
}
```
```
@patent{VCV_Patent,
  author      = {Dylov, Dmitry V. and Rogov, Oleg Y. and Leli, Vito M. and Sarachakov, Aleksandr Y. and Rubashevskii, Aleksandr  A.},
  title       = {Noise-resilient vasculature localization method with regularized segmentation},
  nationality = {United States},
  number      = {US63045376},
  day         = {06},
  month       = {01},
  year        = {2022},
  dayfiled    = {29},
  monthfiled  = {06},
  yearfiled   = {2020},
  url         = {https://patents.google.com/patent/WO2022005336A1,
}
```

## Dataset
```https://drive.google.com/file/d/1rYlKY8HwF2c44mFbnY9aWFDyIj_UP48z/view?usp=sharing```

## Maintainers

[Oleg Rogov](https://github.com/olegrgv)
