import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from scvi.external import MRVI

# Set up GPU for JAX
os.environ["JAX_PLATFORM_NAME"] = "gpu"

# Create output directories
results_dir = "a549_mrvi_results_simple"
os.makedirs(results_dir, exist_ok=True)

# Load data
print("Loading dataset...")
adata = sc.read('a549_combined_data.h5ad')
print(f"Dataset shape: {adata.shape}")

# Basic preprocessing
# print("\nPreprocessing...")
# # Use log1p layer if available, otherwise compute it
# if 'log1p' not in adata.layers:
#     sc.pp.normalize_total(adata)
#     sc.pp.log1p(adata)
# else:
#     adata.X = adata.layers['log1p']

# Identify control samples based on perturbation
control_terms = ['mock']
adata.obs['is_control'] = adata.obs['perturbation'].str.lower().apply(
    lambda x: any(term in str(x).lower() for term in control_terms)
)

# Print summary of controls
n_control = adata.obs['is_control'].sum()
print(f"\nFound {n_control} control cells out of {len(adata)} total cells")

# Select highly variable genes
sc.pp.highly_variable_genes(
    adata,
    n_top_genes=10000,
    flavor="seurat_v3",
    # layer='log1p' if 'log1p' in adata.layers else None
)
adata = adata[:, adata.var.highly_variable]
print(f"Selected {adata.shape[1]} highly variable genes")

# Scale the data
# sc.pp.scale(adata, max_value=10)

# Setup MrVI
print("\nSetting up MrVI...")
MRVI.setup_anndata(
    adata,
    batch_key="batch",
    labels_key="disease",  # Using perturbation as labels
    sample_key='sample_id'
)

# Create and train model
print("\nTraining MrVI model...")
model = MRVI(
    adata,
    # n_latent=30,  # Adjust based on data complexity
    # n_hidden=128,
    # n_layers=2
)

model.train(
    max_epochs=200,
    early_stopping=True,
    early_stopping_patience=20,
    plan_kwargs={"lr": 5e-4},
    accelerator="gpu",
    devices=1
)

# Get latent representations
print("\nExtracting latent representations...")
adata.obsm["X_mrvi"] = model.get_latent_representation()

# Compute neighbors and UMAP
print("\nComputing UMAP...")
sc.pp.neighbors(adata, use_rep="X_mrvi", n_neighbors=15)
sc.tl.umap(adata)

# Basic visualization
print("\nCreating visualizations...")

# Create plots directory
plots_dir = os.path.join(results_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)

# Plot by control status
plt.figure(figsize=(10, 8))
sc.pl.umap(adata, color='is_control', show=False, title='Control vs Treatment')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'umap_control_status.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot by perturbation
plt.figure(figsize=(10, 8))
sc.pl.umap(adata, color='perturbation', show=False, title='Perturbations')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'umap_perturbation.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot by batch
plt.figure(figsize=(10, 8))
sc.pl.umap(adata, color='batch', show=False, title='Batch')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'umap_batch.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot by cell line
plt.figure(figsize=(10, 8))
sc.pl.umap(adata, color='cell_line', show=False, title='Cell Line')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'umap_cell_line.png'), dpi=300, bbox_inches='tight')
plt.close()

# Save results
print("\nSaving results...")
adata.write(os.path.join(results_dir, 'mrvi_results.h5ad'))
model.save(os.path.join(results_dir, 'mrvi_model'))

print("Analysis complete!") 