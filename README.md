# Synthetic Functional Yeast Promoter Sequence Design Using Autoregressive Language Models
This is a python implementation of the synthetic promoter sequence generation method (namely **Gen-DNA-TCN**) reported in:
```
@article{...,
  title={Synthetic Functional Yeast Promoter Sequence Design Using Autoregressive Language Models},
  author={Alsaggaf, Ibrahim and Freitas, Alex A. and de Magalhães, João Pedro and Wan, Cen},
  journal={...},
  pages={...},
  year={...},
  publisher={...},
  note={Under review}
}
```

<p align="center">
  <img src="images/Flowchart.png" width="700" title="Gen-DNA-TCN flowchart">
</p>

# Usage
This repository contains the implementation of the Gen-DNA-TCN model. The implementation is built in Python3 (version 3.10.12) using Scikit-learn, Levenshtein, and the deep learning library Pytorch.

## Requirements
- torch==2.1.2
- scikit-learn==1.4.0
- Levenshtein==0.26.1
- pandas==2.2.3
- numpy==1.26.4

## Tutorial 
To train the Gen-DNA-TCN (**Model 1**) that directly exploits the embedding layers of the pre-trained Pre-DNA-TCN model from the recent Random Promoter DREAM Challenge [(Rafi et al., 2025)](https://www.nature.com/articles/s41587-024-02414-w) execute:
```
code
```

To train Gen-DNA-TCN (**Model 2**) that exploits the 56,879 yeast promoter DNA training sequences for both Pre-DNA-TCN model pre-training and for the second stage of the Gen-DNA-TCN model training execute:
```
code
```

To train Gen-DNA-TCN (**Model 3**) that does not use the pre-trained Pre-DNA-TCN model and directly exploits the 56,879 yeast promoter DNA training sequences to train the Gen-DNA-TCN model execute:
```
code
```

To generate synthetic yeast promoter DNA sequences using the **first type** of starting nucleotides** (i.e. starting nucleotides are extracted from the real 56,879 training sequences) execute:
```
code
```

To generate synthetic yeast promoter DNA sequences using the **second type** of starting nucleotides (i.e. starting nucleotides are made
by random permutations) execute:
```
code
```
To select synthetic yeast promoter DNA sequences using bin-based sampling execute:
```
code
```

To compute pair-wise Levenshtein distance between real and synthetic yeast promoter DNA sequences execute:
```
code
```

## The code
explain each file in the code directory

# Availability
The models' checkpoints and the generated synthetic yeast promoter DNA sequences used in this work can be downloaded from [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18182470.svg)](https://doi.org/10.5281/zenodo.18182470)

# Acknowledgment
The authors acknowledge the support by the School of Computing and Mathematical Sciences and the Birkbeck GTA programme.
