# A549 Single-Cell RNA Analysis Pipeline

This repository contains a comprehensive pipeline for analyzing single-cell RNA sequencing data from A549 cell lines under various perturbations, including viral infections, drug treatments, and radiation exposure.

## Overview

The project focuses on analyzing cellular responses to different perturbations using single-cell RNA sequencing data. It includes tools for data processing, visualization, and analysis using state-of-the-art methods like MrVI (Multi-reference Variational Inference).

## Installation

### Prerequisites
- Conda or Miniconda (recommended)
- Git
- Python 3.13

### Environment Setup

1. Clone the repository:
```bash
git clone [repository-url]
cd A549-Cell-Response-to-Drug-Perturbations
```

2. Create and activate the conda environment:
```bash
conda env create -f python.yml
conda activate scbasecamp_env
```

## Project Structure

### Core Analysis Files
- `downloaddata.ipynb`: Downloads and processes raw data for A549 cell line under various perturbations
- `data_analysis.ipynb`: Main analysis notebook for data exploration and visualization
- `task1.ipynb`: Specific analysis tasks for A549 cell perturbations

### MrVI Analysis
- `mrVI_simple.py`: Basic implementation of MrVI analysis
- `mrVI_v2.py`: Enhanced version of MrVI analysis with additional features


## Data Structure

The analysis expects data in AnnData (`.h5ad`) format with the following key fields:

- `adata.obs['sample_id']`: Sample identifiers
- `adata.obs['is_control']`: Control/treatment labels
- `adata.obs['drug']`: Drug treatment labels
- `adata.obs['perturbation']`: Perturbation information

## Analysis Pipeline

1. **Data Download and Preprocessing**
   - Use `downloaddata.ipynb` to fetch and process raw data
   - Data is converted to AnnData format for downstream analysis

2. **Exploratory Analysis**
   - Run `data_analysis.ipynb` for initial data exploration
   - Generate quality control metrics and basic visualizations

3. **MrVI Analysis**
   - Use `mrVI_v2.py` for advanced analysis
   - Generate embeddings and identify perturbation effects

4. **Task-Specific Analysis**
   - Execute `task1.ipynb` for specific analysis tasks
   - Generate custom visualizations and results

5. **Causal Analysis**
   - Implement causal inference methods from `new_code`
   - Identify causal relationships between perturbations and gene expression
   - Generate causal network visualizations and pathway analysis

## Output

The pipeline generates various outputs:

- Preprocessed data files (`.h5ad` format)
- Quality control metrics and visualizations
- UMAP embeddings for different perturbations
- Differential expression results
- Drug effect analysis results
- Custom visualizations in PNG format

## Dependencies

The project uses the following key Python packages:
- scanpy 1.10
- anndata 0.11
- pandas 2.2
- Jupyter and related packages
- Google Cloud Storage integration tools

## Contributing

Feel free to submit issues and enhancement requests.

## License

[Add appropriate license information]





