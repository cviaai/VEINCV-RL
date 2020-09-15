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
* Python 3.7
* Pytorch
* Pytorch-Ignite
* OpenAI Gym
* Stable-baselines

To install requirements:

```setup
pip install -r requirements.txt
```

## Code structure 
Server folder - main folder with experiment files
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
  * train.py - main training cycle
* img_check - 1 random training sample to check; 4 snapshots in it - original snapshot, ground true mask, predicted image, predicted mask (binarized predicted image)
## Maintainers
Aleksandr Rubashevskii (main contributor) @rubaha96

Vito Michele Leli @vitomichele

Oleg Rogov @olegrgv
