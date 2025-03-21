import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, optimal_leaf_ordering
from scipy.spatial.distance import squareform
from matplotlib.colors import to_hex
import scvi

# Import MrVI from scvi-tools external modules
from scvi.external import MRVI
import os
os.environ["JAX_PLATFORM_NAME"] = "gpu"  # Set this to use GPU

import jax
print("JAX version:", jax.__version__)
print("Available JAX devices:", jax.devices())
print("JAX backend:", jax.config.x64_enabled)

# If devices show GPU, then proceed with your MRVI training
scvi.settings.seed = 0
adata = sc.read('a549_combined_data.h5ad')
print(f"Loaded dataset: {adata.shape[0]} cells and {adata.shape[1]} genes")

# Basic preprocessing
print("Performing basic preprocessing...")
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=10)
# Check if preprocessing has already been done
preprocessing_needed = True

if "log1p" in adata.uns:
    print("Dataset appears to be already normalized")
    preprocessing_needed = False

if preprocessing_needed:
    print("Performing basic preprocessing...")
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=10)
    print(f"After filtering: {adata.shape[0]} cells, {adata.shape[1]} genes")
    
    # Normalize the data
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    print("Normalization complete")
else:
    print("Skipping preprocessing as data appears to be already processed")

# Identify highly variable genes if needed
if "highly_variable" not in adata.var.columns:
    print("Identifying highly variable genes...")
    sc.pp.highly_variable_genes(
        adata, 
        n_top_genes=10000,
        inplace=True, 
        subset=True, 
        flavor="seurat_v3"
    )
else:
    print("Using existing highly variable genes annotation")
    # Subset to highly variable genes if not already done
    if adata.shape[1] > 10000:
        adata = adata[:, adata.var.highly_variable]
        print(f"Subsetted to {adata.shape[1]} highly variable genes")
print("Dataset dimensions:", adata.shape[0], "cells ×", adata.shape[1], "genes")
print("\nObservation metadata columns:")
for col in adata.obs.columns:
    print(f"- {col}")

print("\nVariable metadata columns:")
for col in adata.var.columns:
    print(f"- {col}")
# We can directly use the columns from your data structure
sample_key = "sample_id"  # Your sample identifier
batch_key = "batch" if "batch" in adata.obs.columns else None  # Use batch if available

# Your data already has 'is_control' and 'drug' columns which are perfect for MrVI
print(f"\nUsing '{sample_key}' as sample key and {batch_key} as batch key")

if "is_control" in adata.obs.columns:
    print("Found 'is_control' column - will use for control/treatment comparisons")
    
if "drug" in adata.obs.columns:
    print("\nDrug distribution in dataset:")
    drug_counts = adata.obs["drug"].value_counts()
    for drug, count in drug_counts.items():
        print(f"- {drug}: {count} cells ({count/adata.shape[0]*100:.1f}%)")
    
# Check preprocessing status
if "log1p" in adata.uns:
    print("\nData already normalized with log1p transformation")

# Setup anndata for MrVI with your existing columns
print("\nSetting up anndata for MrVI...")
print(f"Using '{sample_key}' as sample key")
if batch_key:
    print(f"Using '{batch_key}' as batch key")

# Make sure to include batch_key only if it's defined
if batch_key:
    MRVI.setup_anndata(adata, sample_key=sample_key, batch_key=batch_key)
else:
    MRVI.setup_anndata(adata, sample_key=sample_key)

model = MRVI(adata)
print("\nTraining MrVI model...")
model.train(
    max_epochs=400,
    early_stopping=True,
    early_stopping_patience=20,
    plan_kwargs={"lr": 1e-3},  # Slightly higher learning rate
    accelerator="gpu",
    devices=1,  # Explicitly use 1 GPU for stability
    deterministic=True  # For reproducibility
)
results_dir = "a549_mrvi_results3"
fig_dir = os.path.join(results_dir, "figures")
data_dir = os.path.join(results_dir, "data")
report_dir = os.path.join(results_dir, "reports")
model_dir = os.path.join(results_dir, "models")
os.makedirs(model_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
os.makedirs(fig_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)
os.makedirs(report_dir, exist_ok=True)
# Plot training history
plt.figure(figsize=(10, 6))
plt.plot(model.history["elbo_validation"].iloc[5:])
plt.xlabel("Epoch")
plt.ylabel("Validation ELBO")
plt.title("MrVI Training Convergence")
plt.savefig(f"{results_dir}/mrvi_training_convergence.png")
plt.close()

# Get latent representation (u) - capturing broad cell states
print("Extracting latent representations...")
u = model.get_latent_representation()
adata.obsm["u"] = u

# Compute neighbors and UMAP embedding
print("Computing neighbors and UMAP embedding...")
sc.pp.neighbors(adata, use_rep="u")
sc.tl.umap(adata, min_dist=0.3)



if "X_umap" not in adata.obsm:
    compute_umap = True
    
    # We need neighbors first
    if "neighbors" not in adata.uns:
        print("Computing neighbors...")
        sc.pp.neighbors(adata, use_rep="X_pca" if "X_pca" in adata.obsm else None)
    
    print("Computing UMAP embedding...")
    sc.tl.umap(adata, min_dist=0.3)
else:
    print("Using existing UMAP embedding")
    compute_umap = False

# Determine what to use for cell grouping
if "initial_clustering" in adata.obs.columns:
    cell_type_column = "initial_clustering"
    print(f"Using existing 'initial_clustering' annotation")
elif "leiden" in adata.obs.columns:
    cell_type_column = "leiden"
    print(f"Using existing 'leiden' clustering")
elif "louvain" in adata.obs.columns:
    cell_type_column = "louvain"
    print(f"Using existing 'louvain' clustering")
else:
    # Run clustering if no cell type annotation exists
    print("No clustering found, running Leiden clustering...")
    
    # Make sure we have neighbors computed
    if "neighbors" not in adata.uns:
        sc.pp.neighbors(adata, use_rep="X_pca" if "X_pca" in adata.obsm else None)
        
    # Run Leiden clustering
    sc.tl.leiden(adata, resolution=0.5)
    cell_type_column = "leiden"
    print(f"Created new cell clusters using Leiden algorithm")
    
# Display cluster information
if cell_type_column in adata.obs.columns:
    n_clusters = adata.obs[cell_type_column].nunique()
    print(f"\nFound {n_clusters} cell clusters/types")
    cluster_counts = adata.obs[cell_type_column].value_counts()
    
    # Show top 10 clusters by size
    print("Top 10 clusters by size:")
    for i, (cluster, count) in enumerate(cluster_counts.iloc[:10].items()):
        print(f"- Cluster {cluster}: {count} cells ({count/adata.shape[0]*100:.1f}%)")

print("Creating UMAP visualizations...")
def sanitize_filename(name):
    # Replace invalid characters with underscores
    invalid_chars = '<>:"/\\|?*(),'
    sanitized = ''.join('_' if c in invalid_chars else c for c in name)
    # Replace spaces with underscores and remove multiple consecutive underscores
    sanitized = '_'.join(filter(None, sanitized.split()))
    return sanitized.strip('_')
    

# Plot clusters
sc.pl.umap(
    adata,
    color=[cell_type_column],
    frameon=False,
    save=f"_cell_types3.png" 
)

# Plot drug conditions - create a separate plot for each drug
if "drug" in adata.obs.columns:
    # Get unique drugs, excluding control
    drugs = [d for d in adata.obs["drug"].unique() if d != "control"]
    
    # First plot the overall drug distribution
    sc.pl.umap(
        adata,
        color=["drug"],
        frameon=False,
        save=f"_all_drugs3.png"     
        )
    
    # Then create a custom plot to compare each drug vs control
    for drug in drugs:#[:10]:  
        plt.figure(figsize=(12, 5))
        
        # Subset data
        drug_mask = adata.obs["drug"] == drug
        control_mask = adata.obs["drug"] == "control"
        
        # Plot control cells
        plt.subplot(1, 2, 1)
        plt.scatter(
            adata.obsm["X_umap"][~drug_mask & ~control_mask, 0],
            adata.obsm["X_umap"][~drug_mask & ~control_mask, 1],
            c="lightgray", s=5, alpha=0.5, label="Other"
        )
        plt.scatter(
            adata.obsm["X_umap"][control_mask, 0],
            adata.obsm["X_umap"][control_mask, 1],
            c="blue", s=10, alpha=0.7, label="Control"
        )
        plt.title("Control Cells")
        plt.legend()
        plt.axis("off")
        
        # Plot drug cells
        plt.subplot(1, 2, 2)
        plt.scatter(
            adata.obsm["X_umap"][~drug_mask & ~control_mask, 0],
            adata.obsm["X_umap"][~drug_mask & ~control_mask, 1],
            c="lightgray", s=5, alpha=0.5, label="Other"
        )
        plt.scatter(
            adata.obsm["X_umap"][drug_mask, 0],
            adata.obsm["X_umap"][drug_mask, 1],
            c="red", s=10, alpha=0.7, label=drug
        )
        plt.title(f"{drug} Cells")
        plt.legend()
        plt.axis("off")
        
        plt.tight_layout()
        safe_filename = sanitize_filename(drug)
        plt.savefig(f"{fig_dir}/umap_control_vs_{safe_filename}.png")
        plt.close()
    
    print(f"Created UMAP comparison plots for top drugs")

# Sample distances
print("Computing sample distances...")
dists = model.get_local_sample_distances(
    keep_cell=False, 
    groupby=cell_type_column,  # Group by cell types
    batch_size=32
)


model_file = os.path.join(model_dir, "mrvi_model3")
model.save(model_file)

# Save results
print("Saving results...")
adata.write(f"{results_dir}/mrvi_analyzed_data.h5ad")
# Create a directory structure for results
os.makedirs(results_dir, exist_ok=True)
os.makedirs(fig_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)
os.makedirs(report_dir, exist_ok=True)

# Create comprehensive visualizations
print("\nGenerating visualizations...")

# 1. Basic UMAP visualization
if "X_umap" in adata.obsm:
    # Plot clusters
    sc.pl.umap(
        adata,
        color=[cell_type_column],
        frameon=False,
        save=f"{cell_type_column}_3.png",
        show=False
    )
    
    # Plot drug if available
    if "drug" in adata.obs.columns:
        sc.pl.umap(
            adata,
            color=["drug"],
            frameon=False,
            save="_drug3.png",
            show=False
        )
    
    # Plot condition if available
    if "is_control" in adata.obs.columns:
        sc.pl.umap(
            adata,
            color=["is_control"],
            frameon=False,
            save="_is_control3.png",
            show=False
        )
    
    # Plot perturbation if available
    if "perturbation" in adata.obs.columns:
        sc.pl.umap(
            adata,
            color=["perturbation"],
            frameon=False,
            save="_perturbation3.png",
            show=False
        )
    
    # Plot overall drug effect if available
    if "overall_drug_effect" in adata.obs.columns:
        sc.pl.umap(
            adata,
            color=["overall_drug_effect"],
            frameon=False,
            vmax=np.quantile(adata.obs["overall_drug_effect"], 0.95),
            cmap="viridis",
            save="_overall_drug_effect3.png",
            show=False
        )
    
    # Plot individual drug effects
    effect_columns = [col for col in adata.obs.columns if col.endswith("_effect_size")]
    if effect_columns:
        sc.pl.umap(
            adata,
            color=effect_columns[:min(9, len(effect_columns))],  # Up to 9 drugs
            frameon=False,
            ncols=3,
            vmax=0.5,  # Set a consistent scale
            cmap="viridis",
            save="_drug_effects3.png",
            show=False
        )
    
    # Move figures to the results directory
    try:
        from glob import glob
        import shutil
        
        # Find all generated figures
        figures = glob("figures/*.png")
        for fig in figures:
            # Copy to our results directory
            shutil.copy(fig, fig_dir)
            
        print(f"Saved UMAP visualizations to {fig_dir}")
    except Exception as e:
        print(f"Error copying figures: {e}")
else:
    print("No UMAP embedding found, skipping UMAP visualizations")

