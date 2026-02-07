<h1 align='center' style="text-align:center; font-weight:bold; font-size:2.0em;letter-spacing:2.0px;"> ReHARK: Refined Hybrid Adaptive RBF Kernels for Robust One-Shot Vision-Language Adaptation </h1>
<p align="center" style="margin: 0;">🚀 Achieving State-of-the-Art in Training-Free One-Shot Adaptation 🚀</p>

<p align='center' style="text-align:center;font-size:1.25em;">
<a href="mailto:yassir.bendou@gmail.com" target="_blank" style="text-decoration: none;">Yassir Bendou</a>



<em>IMT Atlantique</em>
</p>

<p align='center' style="text-align:center;font-size:2.5 em;">
<b>
<a href="https://arxiv.org/abs/2501.11175" target="_blank" style="text-decoration: none;">[Paper]</a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#" target="_blank" style="text-decoration: none;">[Project Page]</a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#" target="_blank" style="text-decoration: none;">[Code]</a>
</b>
</p>

Abstract

The adaptation of large-scale Vision-Language Models (VLMs) like CLIP to downstream tasks in the one-shot regime is often hindered by a significant "Stability-Plasticity" dilemma. ReHARK (Refined Hybrid Adaptive RBF Kernels) is a synergistic training-free framework that reinterprets few-shot adaptation through global proximal regularization in a Reproducing Kernel Hilbert Space (RKHS).

The proposed pipeline consists of:

(1) Hybrid Prior Construction: Seamlessly fusing CLIP and GPT-3 textual knowledge with visual prototypes to form robust semantic-visual anchors.

(2) Support Set Augmentation (Bridging): Generating intermediate samples to smooth the transition between visual and textual modalities.

(3) Adaptive Distribution Rectification: Aligning test feature statistics with the augmented support set to mitigate domain shifts.

(4) Multi-Scale RBF Kernels: Utilizing an ensemble of kernels to capture complex feature geometries across diverse scales.

ReHARK establishes a new state-of-the-art for one-shot adaptation, achieving an average accuracy of 65.83% across 11 benchmarks.

Requirements

Installation

Using conda

Create a conda environment and install dependencies:

conda create -n rehark python=3.9
conda activate rehark

pip install -r requirements.txt


Using uv

If you prefer to use uv:

uv venv --python 3.9
source .venv/bin/activate
uv pip install -r requirements.txt


Dataset

Follow DATASET.md to install ImageNet and the other 10 benchmarks (Caltech101, DTD, EuroSAT, FGVCAircraft, Food101, OxfordFlowers, OxfordPets, StanfordCars, SUN397, UCF101) referring to CoOp.

Get Started

Configs

The running configurations, including the Optuna search space and kernel parameters, can be modified directly in the configs/ directory.

Running

For one-shot classification with the standard 1,000 trial search budget:

python main.py --method ReHARK --shots 1 --dataset caltech101 --augment-epoch 10


If GPU memory is saturated, consider using fewer data augmentations via --augment-epoch.

Running Options

Multiple methods are implemented for comparison:
| Name | Details |
| :--- | :--- |
| ZeroShot | CLIP |
| TIP | Tip-Adapter: Training-free CLIP-Adapter for Better Vision-Language Modeling |
| GDA | A Hard-to-Beat Baseline for Training-free CLIP-based Adaptation |
| ProKeR | A Kernel Perspective on Training-Free Few-Shot Adaptation |
| ReHARK | ReHARK (ours) - Refined Hybrid Adaptive RBF Kernels |

Performance & Analysis

Method

Avg. Accuracy (1-Shot)

Linear Baseline

55.45%

Laplacian Kernel

60.84%

ReHARK (Proposed)

65.83%

Acknowledgement

This repository benefits from ProKeR, Tip-Adapter, and GDA.

Citation

@article{ReHARK2026,
  title={Refined Hybrid Adaptive RBF Kernels for Robust One-Shot Vision-Language Adaptation},
  author={Bendou, Yassir},
  journal={arXiv preprint},
  year={2026},
  url={[https://arxiv.org/abs/2501.11175](https://arxiv.org/abs/2501.11175)}
}


Contact

If you have any questions, feel free to contact:

Maintainer: Md ajhidul Islam (2006123@eee.buet.ac.bd)
