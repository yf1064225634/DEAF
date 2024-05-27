# DEAF: An Adaptive Feature Aggregation Model for Predicting Soil CO2 Flux
## Table of Contents
- [1. About The Project](#Project) 
-  [2. Usage](#directory-structure) 
- [3. Directory Structure](#usage) 
- [4. Structure Declaration](#Structure-declaration)
- [5. Getting Started](#Getting-Started) 
- [6. Citation](#contact)
- [7. Contact](#contact)
## 1. About The Project
This project proposes a time-series adaptive feature model for predicting soil respiration CO2 flux. The aim is to uncover potential correlations between channels in the raw sequences and effectively mitigate the impact of external environmental noise, striving for efficient, accurate, and non-destructive prediction of soil respiration CO2 flux.

Related paper: [DEAF: An Adaptive Feature Aggregation Model for Predicting Soil CO2 Flux](linking)

## 2. Usage
this repo provides several features:
- You can preprocess your data and load time-series feature data in the `data_provider` folder.
- You can find some publicly available datasets used in the current project in the `dataset` folder. These datasets are open source.
- You can view some models used, including public models like Transformer, Autoformer, DLinear, etc., as well as the DEAF model proposed in this project. All of these are placed in the `models` folder.
- Some modules used by the models are also in the `layers` folder, including existing modules (AutoCorrelation), improved modules (ECA), and self-proposed models (Adain).
- Once the environment is set up, you can proceed with model training, testing, and validation, with some visual information provided.

If you have any questions about the models, feel free to engage in technical discussions. Please note that there are still many possible improvements between models and modules.

## 3. Directory Structure
The repository is structured as follows:
DEAF/
├── data_provider/
│  &nbsp; &nbsp;├── data_factory.py
│  &nbsp; &nbsp;└── data_loader.py
├── dataset/
│  &nbsp; &nbsp;├── ETT
│  &nbsp; &nbsp;├── EX
│  &nbsp; &nbsp;├── ILL
│  &nbsp; &nbsp;└──  WEA
├── exp/
│&nbsp; &nbsp; ├── exp_basic.py
│ &nbsp; &nbsp;└── exp_main.py
├── layers/
│ &nbsp; &nbsp;├── Adain.py
│ &nbsp; &nbsp;├── Adain2.py
│&nbsp; &nbsp; ├── AutoCorrelation.py
│ &nbsp; &nbsp;├── Autoformer_EncDec.py
│&nbsp; &nbsp; ├── DishTS.py
│&nbsp; &nbsp; ├── ECA.py
│ &nbsp; &nbsp;├── Embed.py
│&nbsp; &nbsp; ├── REVIN.py
│&nbsp; &nbsp; ├── SefAttention_Family.py
│&nbsp; &nbsp; └── Transformer_EncDec.py
├── models/
│ &nbsp; &nbsp;├── Autoformer.py
│ &nbsp; &nbsp;├── DEAF.py
│ &nbsp; &nbsp;├── DLinear.py
│ &nbsp; &nbsp;├── Informer.py
│&nbsp; &nbsp; ├── Linear.py
│&nbsp; &nbsp; ├── NLinear.py
│ &nbsp; &nbsp;├── Stat_models.py
│ &nbsp; &nbsp;└── Tranformer.py
├── scripts/
├── utils/
│ &nbsp; &nbsp;├── masking.py
│ &nbsp; &nbsp;├── metrcs.py
│ &nbsp; &nbsp;├── timefeatures.py
│ &nbsp; &nbsp;└── tools.py
└── run.py
└── README.md

## 4. Directory Structure
**data_provider/**
The folders contain files for reading data:
- **data_factory.py**: Data partitioning and parameter settings.
- **data_loader.py**: Data reading and loading settings.

**dataset/**
This folder contains datasets required for the models.

**exp/**
This folder is used for executing model training, testing, and validation.
- **exp_basic.py**: Model device acquisition (CPU, GPU).
- **exp_main.py**: Model training, testing, and validation, including calculation of related metrics.

**layers/**
This folder contains the network modules required by various models, stored modularly.

**models/**
This folder contains integrated models of various modules, including comparison models and DEAF.

**scripts/**
This folder contains parameter scripts for model training, testing, and validation to help run quickly.

**utils/**
This folder contains functions for calculating feature matrices, masks, evaluation metrics, etc.

**run/**
Unified interface for model execution.

## 5. Getting Started

1. Install Python 3.10, PyTorch 1.9.0, and Shell environment.
2. Prepare the data. Some data is already provided in the project.
3. Train the model. We provide the experiment scripts for all benchmarks under the folder `./scripts`. You can reproduce the experiment results by running:
   ```
   sh ./scripts/Exp_DEAF_diff5/ETTh1.sh
   ```
4. You can dive deeper into your results with several tools.

## 6. Citation

If you find this repository useful, please cite our paper... Incoming.

## 7. Contact

If you have any questions, please contact [ydsaYF@163.com](mailto:ydsaYF@163.com).
