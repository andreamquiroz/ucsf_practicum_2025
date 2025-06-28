# UCSF Cancer Practicum 2024-2025

As per the practicum deliverables from our USF mentor, Cody Carroll, we have been tasked with the following: "Along with the write up of your practicum, you should also submit a GitHub page with a complete and well-organized ReadMe which explains the details of your code. If you used branches to divide your work, they should be integrated so that the final code can be found on one main branch. The code must function such that an independent third party is able to reproduce the analysis, figures, and final results in your report."

At UCSF, we have been working on leveraging large language models (LLMs) to enhance cancer patient outcomes and prediction of cancer. The biggest hurdle we have faced is ensuring that the electronic health records (EHRs) notes aren't truncated by the models we used, and thus the problem of the long context window has come to the forefront in our project. The repository is organized as follows:

- A folder of all the literature that has helped us throughout the practicum, both for our reports and the methods we ended up taking for our project.
- The scripts containing relevant code of our methods and approaches

**Note:** Due to the nature of the data, and in compliance with the Health Insurance Portability and Accountability Act (HIPAA), we only provide the base code and will not be including any scripts, notebooks, and the datasets themselves that will contain sensitive information. We ensure that anything that could potentially include that data is censored.

# SeerAttention Medical Classification

Medical text classification using SeerAttention sparse attention. Clone SeerAttention repo first, then copy our scripts into it.

## Quick Setup

1. **Clone SeerAttention repo:**
```bash
git clone https://github.com/microsoft/SeerAttention.git
cd SeerAttention
```

2. **Setup environment:**
```bash
conda create -yn seer python=3.11
conda activate seer
pip install torch==2.4.0
pip install -r requirements.txt
pip install -e .
pip install scikit-learn pandas matplotlib seaborn tqdm
```

3. **Copy the scripts** into the SeerAttention directory

4. **Run experiments:**
```bash
# Baseline SeerAttention
python simplified_train.py \
    --train_file your_train.pkl \
    --test_file your_test.pkl \
    --output_dir ./output

# Biomedical SeerAttention  
python bio_simplified_train.py \
    --train_file your_train.pkl \
    --test_file your_test.pkl \
    --output_dir ./bio_output
```

## Data Format

Your pickle files need pandas DataFrames with columns:
- `text`: Medical document text
- `label`: Binary classification (0/1)
- `text_length`: Character count
- `overallsurvival`: Survival time (optional)
- `stage_grade`: Cancer stage '0-2', '3', '4' (optional)

## Hardware

Tested on 4x NVIDIA RTX A6000 GPUs (48GB each). Baseline model took around ~18 hours per dataset, biomedical model took less than 20 minutes for each dataset

## Files

- `simplified_train.py`: Baseline SeerAttention training
- `bio_simplified_train.py`: Biomedical model training
- `simplified_model.py`: Baseline model class
- `bio_simplified_model.py`: Biomedical model class
- `simple_dataset.py`: Dataset loader
- `bio_simplified_dataset.py`: Biomedical dataset loader

## Citation

Based on SeerAttention: https://github.com/microsoft/SeerAttention
