# Conservative Prediction via Data-Driven Confidence Minimization
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/rlworkgroup/metaworld/blob/master/LICENSE)

This repository contains the official code for ["Conservative Prediction via Data-Driven Confidence Minimization"](https://arxiv.org/abs/2306.04974) by [Caroline Choi*](https://www.linkedin.com/in/caroline-choi-4a915012a/), [Fahim Tajwar*](https://tajwarfahim.github.io/), [Yoonho Lee*](https://yoonholee.com/), [Huaxiu Yao](https://www.huaxiuyao.io/), [Ananya Kumar](https://ananyakumar.wordpress.com/), and [Chelsea Finn](https://ai.stanford.edu/~cbfinn/).

Any correspondence about the code should be addressed to Caroline Choi (cchoi1@stanford.edu) or Fahim Tajwar (ftajwar@andrew.cmu.edu).

## Citing our paper

If you use our code, you can cite our paper as follows:

```
@misc{choi2023conservative,
      title={Conservative Prediction via Data-Driven Confidence Minimization}, 
      author={Caroline Choi and Fahim Tajwar and Yoonho Lee and Huaxiu Yao and Ananya Kumar and Chelsea Finn},
      year={2023},
      eprint={2306.04974},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Setup Conda environment for the experiments

If you have not set up conda, please use [appropriate instructions](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) to download and set up conda on your local device. Use the following commands in a shell terminal.

```
conda create -n dcm python=3.9
conda activate dcm
pip install -r requirements.txt
```

In case some packages are missing when trying to run the experiments or version mismatches happen, please install them on your own. Most of our code assumes one GPU is available, and the code is not guaranteed to work without any GPU. Please make necessary changes in the code if this is the case. Also, please make sure to follow the instructions in [pytorch](https://pytorch.org/get-started/locally/) or [pytorch previous versions](https://pytorch.org/get-started/previous-versions/) to download the pytorch version that matches the cuda version of the local device, the torch version provided in the requirements file might not always be suitable.

## Download datasets

Please see the instructions in [OOD Detection Inconsistency](https://github.com/tajwarfahim/OOD_Detection_Inconsistency), [WILDS](https://github.com/p-lambda/wilds), [Robustness](https://github.com/hendrycks/robustness) and [Group DRO](https://github.com/kohpangwei/group_DRO) repositories to download the required datasets.

## Pre-trained model weights

Pre-trained model weights for OOD detection experiments can be found in the following link: [link](https://drive.google.com/drive/folders/1fDfVdyFtMdArI1H2i4zLdPe3bT5c1IoW?usp=sharing). 

Similarly, weights for the selective classification experiments can be found [here](https://drive.google.com/drive/folders/1Wg-bznMcdu6dcgFBzGdvJP1A_I3wOD7y?usp=sharing).

## Example scripts for OOD detection

To train models, run the following command:

```
cd ood_detection
bash train_script.sh
```

To test pre-trained models (in this case, on CIFAR-10), run the following command after downloading appropriate datasets and pre-trained models:

```
bash test_script.sh
```

Inside these scripts are example python commands that can be used to reproduce our experiment results.


## Example scripts for selective classification

Similarly, to run the selective classification experiments on CIFAR-10, run the following:

```
cd selective_classification
bash exp_script.sh
```

## Acknowledgements
We gratefully acknowledge authors of the following repositories:

1. [Outlier Exposure](https://github.com/hendrycks/outlier-exposure),
2. [Energy based Out-of-distribution Detection](https://github.com/wetliu/energy_ood),
3. [Mahalanobis Method for OOD Detection](https://github.com/pokaxpoka/deep_Mahalanobis_detector) and
4. [Siamese Network](https://github.com/fangpin/siamese-pytorch)
5. [Deep Gamblers](https://github.com/Z-T-WANG/NIPS2019DeepGamblers)
6. [Self-Adaptive Training](https://github.com/LayneH/self-adaptive-training)
7. [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar)
8. [OOD Detection Inconsistency](https://github.com/tajwarfahim/OOD_Detection_Inconsistency)

We thank the authors of these repositories for providing us with easy-to-work-with codebases.



