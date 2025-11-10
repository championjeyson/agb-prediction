[![CI](https://github.com/championjeyson/agb-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/championjeyson/agb-prediction/actions/workflows/ci.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/championjeyson/agb-prediction/blob/main/notebooks/demo_inference.ipynb)

[![CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey?logo=creative-commons&logoColor=white)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Paper: Sialelli 2025](https://img.shields.io/badge/Paper-Sialelli+2025-green?logo=document&logoColor=white)](https://isprs-annals.copernicus.org/articles/X-G-2025/829/2025/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-AGBD-yellow?logo=huggingface)](https://huggingface.co/datasets/prs-eth/AGBD)
[![GitHub Repo](https://img.shields.io/badge/GitHub-AGBD-lightgrey?logo=github)](https://github.com/ghjuliasialelli/AGBD)

![image header](figures/sialelli+2025_fig4_mod-01.png.png)

# Above-Ground Biomass (AGB) Prediction

This repository implements the inference part of the best model described and trained on the **AGBD** (Above-Ground Biomass Dataset) described in:

> **Sialelli, G., Peters, T., Wegner, J. D., and Schindler, K. (2025).**  
> *AGBD: A Global-scale Biomass Dataset.*  
> *ISPRS Ann. Photogramm. Remote Sens. Spatial Inf. Sci.*, X-G-2025, 829–838.  
> [https://doi.org/10.5194/isprs-annals-X-G-2025-829-2025](https://doi.org/10.5194/isprs-annals-X-G-2025-829-2025)

---
## 🚀 Overview

This project implements the model pipeline for **biomass inference from multi-sensor satellite imagery**, using reproducible Python workflows and GEE preprocessing scripts.

It includes:
- Configurable model setup (`configs/`)
- Pretrained model weights for the best architecture and model 
- Example notebook for inference, with example data
- CI-tested modular Python codebase

Not directly included into this repository, it also links to the Google Earh Engine code that extracts the input as required for any area of intererest:
[Google Earth Engine Code](https://code.earthengine.google.com/af9dcf4d48f154082386ae82de1c69f6)

## OLD

This repository implements the best model for AGB inference described in:

**Sialelli, G., Peters, T., Wegner, J. D., and Schindler, K. (2025).** AGBD: A Global-scale Biomass Dataset. ISPRS Ann. Photogramm. Remote Sens. Spatial Inf. Sci., X-G-2025, 829–838. https://doi.org/10.5194/isprs-annals-X-G-2025-829-2025

## Google Earth Engine code to extract the input for your area of interest
https://code.earthengine.google.com/af9dcf4d48f154082386ae82de1c69f6

Note: inputs, outputs, and pretrained weights may be large files so they are stored through LFS. Make sure you have that installed to be able to download everything.
If not working, you'll be able to find the pretrained weights here:...
