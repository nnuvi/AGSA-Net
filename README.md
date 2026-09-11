# AGSA-Net [IEEE TGRS 2026]

This is the official PyTorch implementation of [AGSA-Net: Abundance-Guided Self-Attention Network for Spectral Unmixing-Aware Hyperspectral Remote Sensing Image Classification](https://ieeexplore.ieee.org/document/11675920) by Nafisa Anjum, Satavisa Dey Borno, Ananna Saha, [Mir Faiyaz Hossain](https://scholar.google.com/citations?user=bKzBdFwAAAAJ&hl=en&oi=ao), [Sifat Momen](https://scholar.google.com/citations?user=sGVZEaAAAAAJ&hl=en&oi=ao), [Nabeel Mohammed](https://scholar.google.com/citations?user=w5djOYsAAAAJ&hl=en), and [Shafin Rahman](https://scholar.google.com/citations?hl=en&user=Pe8C-SUAAAAJ&view_op=list_works&sortby=pubdate).

## Overview

AGSA-Net is an abundance-guided self-attention framework for hyperspectral remote sensing image classification. The proposed model leverage **spectral unmixing** to estimate physically meaningful abundance maps, which represent the material composition of each pixel. These abundance maps are used to guide a **self-attention transformer**, allowing the network to focus on **class-discriminative spectral interactions** and long-range dependencies.

<p align="center">
  <img src="./architecture_final.png" width="750">
</p>
<p align="center">
  <small><em>Figure: Architecture of the proposed Abundance-Guided Self-Attention Network (AGSA-Net) for spectral unmixing-aware hyperspectral image classification.</em></small>
</p>


The network first estimates physically meaningful abundance maps under non-negativity and sum-to-one constraints. These maps are refined using a hybrid linear–nonlinear reconstruction decoder and used to build an abundance affinity prior. This prior guides the spectral self-attention transformer to focus on class-discriminative spectral interactions. Finally, transformer features are fused with compact abundance descriptors for classification, combining spectral unmixing insights with self-attention modeling.


## Directory Structure

```text
├── data/
|  ├── IndianPine.mat                 
|  ├── Augburg/           
|  ├── Berlin/   
├── result/               # Trained model
├── output/               # generated outputs
├── dataset.py
├── main.py
├── metrics.py
├── model.py
├── train.py
├── utils.py
```

All quantitative and qualitative outputs are **automatically created when the code is executed**.

The results reported in the manuscript were obtained by running this implementation under the experimental settings described in the paper.

## Dataset

The adopted Augsburg and Berlin datasets can be downloaded from (https://pan.baidu.com/s/1fOzt4CJ0FQwDjGSP3DOU9w?pwd=c258) (code: c258).

## Run

### Train
Main command to train the model on a specified dataset. Dataset and patch size can be changed via `--dataset` and `--patches`.

```bash
python main.py --dataset Indian --flag_test train --patches 7
```

### Test 
Command to evaluate the trained model on a specified dataset. Saved model needs to be loaded before testing.

```bash
python main.py --dataset Indian --flag_test test --patches 7
```
## ⭐ Citation

If you find this repository or **AGSA-Net** useful in your research, please kindly cite our work:

```bibtex
@ARTICLE{11675920,
  author={Anjum, Nafisa and Borno, Satavisa Dey and Saha, Ananna and Hossain, Mir Faiyaz and Momen, Sifat and Mohammed, Nabeel and Rahman, Shafin},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  title={AGSA-Net: Abundance-Guided Self-Attention Network for Spectral Unmixing-Aware Hyperspectral Remote Sensing Image Classification},
  year={2026},
  volume={},
  number={},
  pages={1-1},
  keywords={Modeling;Materials;Pixel;Transformers;Labeling;Training;Hyperspectral imaging;Image classification;Accuracy;Convolutional neural networks;Hyperspectral imaging;remote sensing;spectral unmixing;abundance estimation;geoscience;deep learning;transformers;self-attention;hyperspectral image classification},
  doi={10.1109/TGRS.2026.3729910}
}

## Acknowledgement

Part of this code is based on the implementation of [DSNet](https://github.com/hanzhu97702/DSNet). Many thanks to the authors for their valuable work.


