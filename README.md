# CDRNet: Change Discriminability Restoration Network for Foggy Remote Sensing Change Detection


<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue">
  <img src="https://img.shields.io/badge/PyTorch-1.10+-red">
  <img src="https://img.shields.io/badge/Task-Remote%20Sensing%20Change%20Detection-green">
  <img src="https://img.shields.io/badge/Scene-Foggy%20Remote%20Sensing-purple">
</p>

---

## Introduction

This repository provides the implementation of **CDRNet**, a **Change Discriminability Restoration Network** for **foggy remote sensing change detection**.

Fog interference usually weakens structural cues of real changed regions and induces pseudo-change responses in unchanged backgrounds. To address this issue, CDRNet restores change discriminability by compensating fog-weakened structural cues and disentangling fog-induced pseudo-change responses during progressive change decoding.

---

## Framework

<p align="center">
  <img src="CDRNet_framework.jpg" width="900">
</p>

CDRNet consists of the following components:

- **Weight-sharing PVT-v2-B2 Backbone**  
  Extracts hierarchical bi-temporal features from two temporal remote sensing images.

- **Structural Cue Compensation Module (SCCM)**  
  Compensates fog-weakened structural cues through spatial semantic preservation and wavelet-domain structural modeling.

- **Multi-scale Structural Edge Guidance Head (MSEGH)**  
  Aggregates multi-scale structure-enhanced features to generate edge guidance for boundary-aware decoding.

- **Fog-induced Pseudo-change Disentanglement Module (FPDM)**  
  Suppresses fog-induced pseudo-change responses and progressively decodes reliable changed regions.

---

## Highlights

- A degradation-aware framework for foggy remote sensing change detection.
- Structural cue compensation for restoring real changed regions degraded by fog.
- Frequency-domain reliable-change enhancement based on wavelet decomposition.
- Edge-guided pseudo-change disentanglement for suppressing fog-induced false responses.
- Progressive coarse-to-fine decoding for accurate binary change prediction.

---

## Repository Structure

```text
CDRNet/
├── network/
│   ├── CDRNet.py
│   ├── SCC.py
│   ├── FPD.py
│   ├── Edge.py
│   ├── tools.py
│   └── backbones/
│       └── pvtv2.py
├── utils/
│   ├── dataloader.py
│   ├── metrics.py
│   └── tools.py
├── pretrained_model/
│   └── pvt_v2_b2.pth
├── figures/
│   └── CDRNet_framework.png
├── train_v2.py
├── test.py
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/your-username/CDRNet.git
cd CDRNet

conda create -n cdrnet python=3.8
conda activate cdrnet

pip install -r requirements.txt
```

A typical environment includes:

```text
python >= 3.8
torch >= 1.10
torchvision
numpy
opencv-python
tqdm
scikit-learn
Pillow
```

Please install a PyTorch version compatible with your CUDA version.

---

## Pretrained Backbone

CDRNet adopts **PVT-v2-B2** as the weight-sharing backbone.

Please download the pretrained PVT-v2-B2 model and place it under:

```text
./pretrained_model/pvt_v2_b2.pth
```

The default path is:

```python
path = './pretrained_model/pvt_v2_b2.pth'
```

---

## Dataset Preparation

Please organize the dataset as follows:

```text
Dataset/
├── train/
│   ├── T1/
│   ├── T2/
│   └── GT/
├── val/
│   ├── T1/
│   ├── T2/
│   └── GT/
└── test/
    ├── T1/
    ├── T2/
    └── GT/
```

Each sample contains:

- `T1`: image at the first time point
- `T2`: image at the second time point
- `GT`: binary change mask

The ground-truth mask should follow:

```text
0: unchanged
1: changed
```

---

## Training

Modify the dataset path and training configuration in `train_v2.py`, then run:

```bash
python train_v2.py \
  --data_name LEVIR-CD-HTC136_v10 \
  --epoch 200 \
  --batchsize 8 \
  --trainsize 256 \
  --lr_cdrnet 1e-4 \
  --edge_weight 0.1 \
  --reg_weight 1e-4
```

Main options:

```text
--data_name       dataset name
--epoch           number of training epochs
--batchsize       batch size
--trainsize       input image size
--lr_cdrnet       learning rate
--edge_weight     weight of edge supervision
--reg_weight      weight of regularization
```

The trained model will be saved to:

```text
./train_output/CDRNet/{data_name}/
```

---

## Testing

After training, run:

```bash
python test.py \
  --data_name LEVIR-CD-HTC136_v10 \
  --model_path ./train_output/CDRNet/LEVIR-CD-HTC136_v10/Seg_epoch_best.pth
```

The predicted change maps will be saved in the configured output directory.



## Citation

If you find this repository useful, please consider citing our paper:

```bibtex
@article{cdrnet2026,
  title={Change Discriminability Restoration Network for Foggy Remote Sensing Change Detection},
  author={},
  journal={},
  year={2026}
}
```

---

## Acknowledgement

This project uses **PVT-v2** as the backbone. We thank the authors of PVT and the open-source remote sensing change detection community for their valuable contributions.

---


