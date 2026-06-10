# CDRNet: Change Discriminability Restoration Network for Foggy Remote Sensing Change Detection


<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue">
  <img src="https://img.shields.io/badge/PyTorch-2.10-red">
  <img src="https://img.shields.io/badge/Scene-Foggy%20Remote%20Sensing-purple">
  <img src="https://img.shields.io/badge/Task-Remote%20Sensing%20Change%20Detection-green">
</p>



## Introduction

This repository provides the implementation of a **Change Discriminability Restoration Network (CDRNet)** for **foggy remote sensing change detection**.

Fog interference usually weakens structural cues of real changed regions and induces pseudo-change responses in unchanged backgrounds. To address this issue, CDRNet restores change discriminability by compensating fog-weakened structural cues and disentangling fog-induced pseudo-change responses during progressive change decoding.



<p align="center">
  <img src="figures/CDRNet_framework.jpg" width="900">
</p>

CDRNet consists of the following components:

- **Structural Cue Compensation Module (SCCM)**  
  Compensates fog-weakened structural cues through spatial semantic preservation and wavelet-domain structural modeling.

- **Structural Edge Guidance (SEG)**  
  Aggregates multi-scale structure-enhanced features to generate edge guidance for boundary-aware decoding.

- **Fog-induced Pseudo-change Disentanglement Module (FPDM)**  
  Suppresses fog-induced pseudo-change responses and progressively decodes reliable changed regions.

---

## Installation

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


```bash
git clone https://github.com/your-username/CDRNet.git
cd CDRNet

conda create -n cdrnet python=3.8
conda activate cdrnet

pip install -r requirements.txt
```

Please install a PyTorch version compatible with your CUDA version.
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





CDRNet adopts **PVT-v2-B2** as the weight-sharing backbone. 
Please download the pretrained PVT-v2-B2 model and place it under:

```text
./pretrained_model/pvt_v2_b2.pth
```

The default path is:

```python
path = './pretrained_model/pvt_v2_b2.pth'
```


## Dataset Preparation

The new foggy RSCD datasets can be obtained from the [Cloud Drive](https://pan.jiangnan.edu.cn/link/AA0298DDB05F1E43F6A70CF8A4D664AAA3) [RSCD].

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



## Testing

After training, run:

```bash
python test.py \
  --data_name LEVIR-CD-HTC136_v10 \
  --model_path ./train_output/CDRNet/LEVIR-CD-HTC136_v10/Seg_epoch_best.pth
```

The predicted change maps will be saved in the configured output directory.

---

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


## Acknowledgement
:clap::clap::clap: Thanks to the authors of remote sensing change detection for their excellent work！



