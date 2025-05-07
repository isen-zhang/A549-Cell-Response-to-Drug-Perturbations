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
results_dir = "a549_mrvi_results"
plots_dir = os.path.join(results_dir, "plots")
perturbation_plots_dir = os.path.join(plots_dir, "perturbation_comparison")
os.makedirs(results_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(perturbation_plots_dir, exist_ok=True)

# Load data
print("Loading dataset...")
adata = sc.read('a549_combined_data.h5ad')
print(f"Dataset shape: {adata.shape}")

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
    flavor="seurat_v3"
)
adata = adata[:, adata.var.highly_variable]
print(f"Selected {adata.shape[1]} highly variable genes")

# Setup MrVI
print("\nSetting up MrVI...")
MRVI.setup_anndata(
    adata,
    # batch_key="batch",
    labels_key="perturbation",  # Using perturbation as labels
    sample_key='sample_id'
)

# Create and train model
print("\nTraining MrVI model...")
model = MRVI(
    adata,
)

model.train(
    max_epochs=150,
    early_stopping=True,
    early_stopping_patience=20,
    plan_kwargs={"lr": 5e-4},
    accelerator="gpu",
    devices=1
)
# Plot training history
print("\nPlotting training history...")
plt.figure(figsize=(10, 6))
plt.plot(model.history["elbo_validation"].iloc[5:])
plt.xlabel("Epoch")
plt.ylabel("Validation ELBO")
plt.title("MrVI Training Convergence")
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'mrvi_training_curve.png'), dpi=300, bbox_inches='tight')
plt.close()

# Get latent representations
print("\nExtracting latent representations...")
adata.obsm["X_mrvi"] = model.get_latent_representation()

# Compute neighbors and UMAP
print("\nComputing UMAP...")
sc.pp.neighbors(adata, use_rep="X_mrvi", n_neighbors=15)
sc.tl.umap(adata, min_dist=0.3)

# Basic visualization
print("\nCreating visualizations...")

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
# Plot by disease
plt.figure(figsize=(10, 8))
sc.pl.umap(adata, color='disease', show=False, title='Disease')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'umap_disease.png'), dpi=300, bbox_inches='tight')
plt.close()

# --- Add perturbation comparisons ---
print("\nCreating perturbation comparison visualizations...")

# 1. Compare individual perturbations using separate UMAPs
perturbations = adata.obs['perturbation'].unique().tolist()
control_perts = [p for p in perturbations if any(term in p.lower() for term in control_terms)]

if control_perts:
    control_pert = control_perts[0]
    
    # Create individual comparison plots
    for pert in perturbations:
        if pert != control_pert:
            # Subset data to focus on just this perturbation vs control
            subset_mask = adata.obs['perturbation'].isin([pert, control_pert])
            adata_subset = adata[subset_mask].copy()
            
            # Create plot
            plt.figure(figsize=(10, 8))
            sc.pl.umap(
                adata_subset, 
                color='perturbation', 
                show=False, 
                title=f'{pert} vs {control_pert}'
            )
            plt.tight_layout()
            
            # Sanitize filename by replacing problematic characters
            safe_pert_name = pert.replace('/', '_').replace(' ', '_').replace('(', '').replace(')', '')
            plt.savefig(os.path.join(perturbation_plots_dir, f'umap_{safe_pert_name}_vs_control.png'), dpi=300, bbox_inches='tight')
            plt.close()

# 2. Create a heatmap showing distances between perturbations in latent space
print("\nCreating perturbation distance heatmap...")

# Calculate mean latent representation for each perturbation
pert_means = {}
for pert in perturbations:
    pert_mask = adata.obs['perturbation'] == pert
    if np.sum(pert_mask) > 0:  # Ensure there are cells with this perturbation
        pert_means[pert] = np.mean(adata[pert_mask].obsm["X_mrvi"], axis=0)

# Compute distances between all pairs of perturbations
n_perts = len(pert_means)
pert_names = list(pert_means.keys())
distance_matrix = np.zeros((n_perts, n_perts))

for i, pert1 in enumerate(pert_names):
    for j, pert2 in enumerate(pert_names):
        distance_matrix[i, j] = np.linalg.norm(pert_means[pert1] - pert_means[pert2])

# Create heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(
    distance_matrix,
    xticklabels=pert_names,
    yticklabels=pert_names,
    cmap="viridis",
    annot=True
)
plt.title("Distances Between Perturbations in Latent Space")
plt.tight_layout()
plt.savefig(os.path.join(perturbation_plots_dir, "perturbation_distance_heatmap.png"), dpi=300, bbox_inches='tight')
plt.close()

# 3. Create a combined UMAP highlighting multiple perturbations
# Select top 5 perturbations with most cells (excluding control)
if control_perts:
    # Count cells per perturbation
    pert_counts = adata.obs['perturbation'].value_counts()
    
    # Get top 5 non-control perturbations
    non_control_perts = [p for p in pert_counts.index if p not in control_perts]
    top_perts = non_control_perts[:5].tolist() + control_perts
    
    # Subset data
    top_mask = adata.obs['perturbation'].isin(top_perts)
    adata_top = adata[top_mask].copy()
    
    # Create plot
    plt.figure(figsize=(10, 8))
    sc.pl.umap(
        adata_top, 
        color='perturbation', 
        show=False, 
        title='Top Perturbations vs Control'
    )
    plt.tight_layout()
    plt.savefig(os.path.join(perturbation_plots_dir, "top_perturbations.png"), dpi=300, bbox_inches='tight')
    plt.close()

# 4. Create perturbation trajectory visualization
if control_perts:
    print("\nCreating perturbation trajectory visualization...")
    control_pert = control_perts[0]
    
    # Calculate centroid for each perturbation in UMAP space
    pert_centroids = {}
    for pert in perturbations:
        pert_mask = adata.obs['perturbation'] == pert
        if np.sum(pert_mask) > 0:  # Ensure there are cells
            pert_umap = adata[pert_mask].obsm['X_umap']
            pert_centroids[pert] = np.mean(pert_umap, axis=0)
    
    # Create scatter plot
    plt.figure(figsize=(12, 10))
    
    # Plot all cells colored by perturbation (with alpha for visibility)
    for i, pert in enumerate(perturbations):
        pert_mask = adata.obs['perturbation'] == pert
        if np.sum(pert_mask) > 0:  # Ensure there are cells
            pert_umap = adata[pert_mask].obsm['X_umap']
            plt.scatter(
                pert_umap[:, 0], 
                pert_umap[:, 1], 
                s=5, 
                alpha=0.5, 
                label=pert
            )
    
    # Add arrows from control to other perturbations
    if control_pert in pert_centroids:
        control_centroid = pert_centroids[control_pert]
        for pert, centroid in pert_centroids.items():
            if pert != control_pert:
                plt.arrow(
                    control_centroid[0], 
                    control_centroid[1],
                    centroid[0] - control_centroid[0], 
                    centroid[1] - control_centroid[1],
                    head_width=0.2, 
                    head_length=0.3, 
                    fc='black', 
                    ec='black',
                    length_includes_head=True, 
                    alpha=0.7
                )
    
    # Add centroid markers
    for pert, centroid in pert_centroids.items():
        plt.scatter(
            centroid[0], 
            centroid[1], 
            s=100, 
            marker='o', 
            edgecolor='black', 
            linewidth=2,
            label=f"{pert} centroid"
        )
    
    plt.title("Perturbation Trajectories from Control")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(perturbation_plots_dir, "perturbation_trajectories.png"), dpi=300, bbox_inches='tight')
    plt.close()

# 5. Differential expression analysis if possible
try:
    print("\nPerforming differential expression analysis...")
    # Define sample covariates of interest
    sample_cov_keys = ["perturbation"]
    
    # Run differential expression
    de_results = model.differential_expression(
        sample_cov_keys=sample_cov_keys,
        batch_size=128
    )
    
    # Save DE results
    de_results.to_dataframe().to_csv(os.path.join(results_dir, "differential_expression_results.csv"))
    print("Differential expression analysis completed.")
except Exception as e:
    print(f"Differential expression analysis failed: {str(e)}")

# Save results
print("\nSaving results...")
adata.write(os.path.join(results_dir, 'mrvi_results.h5ad'))
model.save(os.path.join(results_dir, 'mrvi_model'))

print("Analysis complete!")