[![Python](https://img.shields.io/badge/python-3.7-blue.svg)](https://python.org)

# VeinCV
## Motivation
Peripheral Difficult Venous Access (PDVA) is a commonplace problem in clinical practice which results in repetitive punctures, damaged veins, and significant discomfort to the patients. Nowadays, the poor visibility of subcutaneous vasculature in the visible part of the light spectrum is overcome by near-infrared (NIR) imaging and a returned projection of the recognized vasculature back to the arm of the patient. We introduce the first “smart” engine to govern the components of such imagers in a mixed reality setting. Namely, we introduce a closed-loop hardware system that optimizes cross-talk between the virtual mask generated from the NIR measurement and the projected augmenting image. Such real-virtual image translation is accomplished by several steps. First, the NIR vein segmentation task is solved using U-Net-based network architecture and the Frangi vesselness filter. The generated mask is then transformed and translated into the visible domain by a projector that adjusts for distortions and misalignment with the true vasculature using the paradigm of Reinforcement Learning (RL). We propose a new class of mixed reality reward functions that guarantees proper alignment of the projected image regardless of angle, translation, and scale offsets between the NIR measurement and the visible projection.
![Experimental setup scheme](https://github.com/cviaai/NIR-VISIBLE-IMAGING-WITH-CNN-RL/blob/master/img/Experimental_setup_scheme.png)

</p>
<p align="center">
<em> Fig. 1. Experimental setup </em><br>
</p>

## Segmentation pipeline
Segmentation pipeline featuring Frangi vesselness filter, attention U-Net and clDICE loss.
![Segmentation pipeline](https://github.com/cviaai/NIR-VISIBLE-IMAGING-WITH-CNN-RL/blob/master/img/Segmentation_pipeline.png)

</p>
<p align="center">
<em> Fig. 2. Segmentation pipeline featuring Frangi vesselness filter, attention U-Net and clDICE loss </em><br>
</p>

## Requirements
To install requirements:

```setup
pip install -r requirements.txt
```

## Training

To train the models used in the paper, run this command:

```train
python train.py --config <path_to_config_file>
```
---Will be edited--- 

## Evaluation

To evaluate models, run:

```eval
python eval.py --config <path_to_config_file>
```
---Will be edited--- 

## Code structure 
Folder "server" - main folder with experiment files
[UPDATE WITH RL FILES]

Option 1.

```
.
├───dataset_90
├───ignite_new
│   ├───architectures
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

Option 2.

* dataset_90 - 90 forearm snapshots and 90 corresponding masks, 75 of them for train, 15 for validation
* ignite_new - experiment
  * architectures - all used convolutional neural network architectures for semantic segmentation task
  * pretrained_models - here saved pre-trained models
  * VeinsDataset.py - preparing data
  * dice_helpers.py - helper function for dice
  * inference.ipynb - Model Inference experiments (draft)
  * iou.py - Intersection over Union metric
  * loss.py - Loss functions (on the following: Binary Cross Entropy, Dice, clDice)
  * main.py - full experiment: paths, parameters, model choice, etc.
  * miou.py - mean Intersection over Union metric
  * precision.py - Precision metric
  * recall.py - Recall metric
  * run.ipynb - run the experiment
  * ssim.py - Structure Similarity Loss function
  * train.py - main training cycle
  * tversky_loss.py - Tversky loss
  * utils.py - utils functions
* img_check - 1 random training sample to check; 4 snapshots in it - original snapshot, ground true mask, predicted image, predicted mask (binarized predicted image)

## Citing
If you use this package in your publications or in other work, please cite it as follows:
```
@misc{rubashevskii2020nir,
    title={Near-Infrared-to-Visible Vein Imaging via Convolutional Neural Networks and Reinforcement  Learning},
    author={Aleksandr Rubashevskii and Vito M. Leli and Aleksandr Sarachakov and Oleg Y. Rogov and Dmitry V. Dylov},
    year={2020},
    eprint={...},
    archivePrefix={arXiv},
    primaryClass={eess.IV}
}
```

## Maintainers
Aleksandr Rubashevskii (main contributor) @rubaha96

Vito Michele Leli @vitomichele

Oleg Rogov @olegrgv
