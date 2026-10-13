# X-Net

### Brain Stroke Lesion Segmentation with Depthwise Separable Convolution and Long-Range Dependencies

[![MICCAI 2019](https://img.shields.io/badge/MICCAI-2019-2455a4)](https://doi.org/10.1007/978-3-030-32248-9_28)
[![arXiv](https://img.shields.io/badge/arXiv-1907.07000-b31b1b)](https://arxiv.org/abs/1907.07000)
[![Framework](https://img.shields.io/badge/framework-Keras%20%2F%20TensorFlow-orange)](#requirements)

Research implementation of **X-Net**, a network for segmenting stroke lesions in brain MRI. X-Net combines depthwise separable convolutions with a **Feature Similarity Module (FSM)** to capture long-range contextual information.

**[Paper](https://arxiv.org/abs/1907.07000) · [Publication](https://doi.org/10.1007/978-3-030-32248-9_28) · [Data preprocessing](https://github.com/Andrewsher/ATLAS-dataset-generate-h5file) · [Citation](#citation) · [中文版](README_CN.md)**

**Authors:** Kehan Qi, Hao Yang, Cheng Li, Zaiyi Liu, Meiyun Wang, Qiegen Liu, and Shanshan Wang.

## Overview

- **Efficient feature extraction:** depthwise separable convolutions reduce the network's parameter footprint.
- **Long-range context:** FSM models feature similarity across spatial locations.
- **Stroke lesion segmentation:** the implementation processes 2D MRI slices and evaluates predictions at the patient level.

## Results

The original repository reports the following results on **229 ATLAS cases**, using **five-fold cross-validation**:

| Dice | IoU | Precision | Recall | Parameters |
| ---: | ---: | ---: | ---: | ---: |
| 0.4867 | 0.3723 | 0.6000 | 0.4752 | 15.1M |

These are the results documented in the original README, not newly reproduced measurements.

## Requirements

The original implementation uses standalone Keras with a TensorFlow 1.x backend.

| Component | Original version |
| --- | --- |
| Operating system | Ubuntu 16.04 |
| Keras | 2.2.4 |
| TensorFlow | 1.14.0 |
| scikit-learn | 0.19.1 |
| pandas | 0.20.3 |
| h5py | 2.7.0 |

NumPy and Matplotlib are also imported by the code; their original versions, along with the Python version, are not specified in the repository. Use a Python environment compatible with the packages above. The existing APIs may require changes to run with TensorFlow 2.x or modern Keras.

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/Andrewsher/X-Net.git
cd X-Net
```

### 2. Prepare the ATLAS dataset

Use the companion [ATLAS preprocessing repository](https://github.com/Andrewsher/ATLAS-dataset-generate-h5file) to convert the MRI images and lesion annotations into HDF5 format. Follow its dataset and dependency instructions before running:

```bash
python generate_h5.py --dataset-path /path/to/ATLAS_R1.1
```

Run this command from the preprocessing repository. Its current script writes `train.h5` under the dataset directory. Set `data_file_path` in X-Net's `main.py` to the generated file.

The X-Net data loader expects:

| Item | Expected format |
| --- | --- |
| HDF5 keys | `data` for images; `label` for lesion masks |
| Image intensities | Normalized to `[0, 1]` |
| Slice ordering | 189 consecutive slices per patient |
| Model input | Single-channel slices of shape `(224, 192, 1)` |
| Mask values | `0` = background; `1` = lesion |

The loader crops stored slices with `[5:229, 2:194]`. Preserve the preprocessing layout and patient ordering: the training script assumes 229 patients, and patient indices are converted directly into slice indices.

### 3. Configure the experiment

Edit these settings near the top of [`main.py`](main.py):

```python
data_file_path = '/path/to/ATLAS_R1.1/train.h5'
pretrained_weights_file = None
input_shape = (224, 192, 1)
batch_size = 8
num_folds = 5
```

Leave `pretrained_weights_file` as `None` to train from scratch, or supply a compatible weights file.

Also adjust the `CUDA_VISIBLE_DEVICES` assignment at the bottom of `main.py` to select an available GPU; its current value is `'1'`. For example, use `'0'` for the first GPU. For reliable device selection, place this assignment before imports that initialize the TensorFlow backend.

### 4. Train and evaluate

```bash
python main.py
```

The script runs patient-level cross-validation and evaluates each fold after training.

## Outputs

Each fold writes to `fold_0/`, `fold_1/`, and so on:

| File | Contents |
| --- | --- |
| `ep=...-loss=...-val_loss=....h5` | Best validation-loss weight checkpoints |
| `trained_final_weights.h5` | Weights at the end of training |
| `record.csv` | Training and validation history |
| `score_record.csv` | Per-patient segmentation scores |
| TensorBoard event files | Training logs |

The aggregate scores are saved to `x_net_final_score.csv` in the working directory. Evaluation in `main.py` uses the final in-memory model; it does not reload the best validation-loss checkpoint.

## Code guide

| File | Purpose |
| --- | --- |
| [`model.py`](model.py) | X-Net architecture |
| [`FSM.py`](FSM.py) | Feature Similarity Module |
| [`data.py`](data.py) | HDF5 loading and slice generators |
| [`loss.py`](loss.py) | Binary cross-entropy plus Dice loss |
| [`main.py`](main.py) | Cross-validation, training, and evaluation |
| [`utils.py`](utils.py) | Patient-level segmentation metrics |

## Citation

If you use X-Net in your research, please cite:

```bibtex
@inproceedings{qi2019xnet,
  title     = {X-Net: Brain Stroke Lesion Segmentation Based on Depthwise
               Separable Convolution and Long-Range Dependencies},
  author    = {Qi, Kehan and Yang, Hao and Li, Cheng and Liu, Zaiyi
               and Wang, Meiyun and Liu, Qiegen and Wang, Shanshan},
  booktitle = {Medical Image Computing and Computer Assisted Intervention
               -- MICCAI 2019},
  series    = {Lecture Notes in Computer Science},
  volume    = {11766},
  pages     = {247--255},
  year      = {2019},
  publisher = {Springer},
  doi       = {10.1007/978-3-030-32248-9_28}
}
```

**Dataset reference:** Liew, Sook-Lei, et al. “A large, open source dataset of stroke anatomical brain images and manual lesion segmentations.” *Scientific Data* 5, 180011 (2018).

## Questions

For implementation questions or reproducibility issues, please [open an issue](https://github.com/Andrewsher/X-Net/issues) with your environment details and the relevant error message.
