# A549 Single-Cell RNA Analysis Pipeline

This repository contains a collection of Jupyter notebooks and Python scripts for analyzing single-cell RNA sequencing data from A549 cell lines under various perturbations.

## Installation

### Prerequisites
- Conda or Miniconda (recommended)
- Git

### Environment Setup


Create and activate the conda environment:
```bash
conda env create -f python.yml
conda activate scbasecamp_env
```


## Project Structure

- `downloaddata.ipynb`: Downloads and processes raw data for A549 cell line under various perturbations
- `data_analysis.ipynb`: Main analysis notebook for data exploration and visualization
- `mrMI.ipynb` & `mrMI_a549.ipynb`: Multi-reference metric learning analysis notebooks
- `mrMI.py`: Python script containing utility functions for MrVI analysis
- `task1.ipynb`: Specific analysis tasks for A549 cell perturbations

## Data Structure

The analysis expects data in AnnData (`.h5ad`) format with the following key fields:

- `adata.obs['sample_id']`: Sample identifiers
- `adata.obs['is_control']`: Control/treatment labels
- `adata.obs['drug']`: Drug treatment labels
- `adata.obs['perturbation']`: Perturbation information

## Output

The analysis pipeline generates several outputs:

- Preprocessed data files (`.h5ad` format)
- Quality control metrics and visualizations
- UMAP embeddings colored by various metadata
- Differential expression results
- Drug effect analysis results





