# Drug-Likeness Prediction Using a Feature-Enriched GNN

A graph neural network project for **Blood-Brain Barrier Penetration
(BBBP) prediction** using molecular structure together with global
physicochemical descriptors.

The central idea is to represent each molecule at three levels:

-   **Atom-level features:** 150-dimensional node representations
-   **Bond-level features:** 10-dimensional edge representations
-   **Molecule-level descriptors:** 18 global physicochemical
    descriptors

These representations are combined in a feature-enriched GNN with early
fusion and attention-based graph pooling.

------------------------------------------------------------------------

## Project Overview

Predicting whether a molecule can cross the blood-brain barrier is an
important problem in early-stage CNS drug discovery. Molecular graphs
naturally represent atoms as nodes and chemical bonds as edges, but
graph structure alone may not capture useful global molecular
properties.

This project therefore enriches graph representations with molecular
descriptors such as:

-   Molecular weight
-   LogP
-   Topological polar surface area (TPSA)
-   Hydrogen-bond donors and acceptors
-   Rotatable bonds
-   Ring counts
-   Aromatic/aliphatic ring counts
-   Bertz complexity
-   QED
-   Heteroatom count
-   Fraction Csp3
-   Other RDKit-derived descriptors

The notebook compares the feature-enriched GNN against conventional and
graph-based baselines.

------------------------------------------------------------------------

## Dataset

**Dataset:** BBBP (Blood-Brain Barrier Penetration) from
DeepChem/MoleculeNet

-   **2,039 molecules**
-   Binary classification task
-   Target: whether a molecule penetrates the blood-brain barrier
-   Molecular input: SMILES strings
-   Invalid SMILES are removed before graph construction
-   Random seed: `42`
-   Train / validation / test split: `80% / 16% / 20%` as implemented
    through two successive splits in the notebook

> Note: the notebook code uses `train_test_split` with stratification
> for the actual split. The notebook contains a comment referring to a
> scaffold split, but the executed implementation is a stratified random
> split.

------------------------------------------------------------------------

## Method

### 1. Molecular Feature Extraction

Each molecule is converted into a graph using RDKit and PyTorch
Geometric.

### Node features

Each atom is represented using properties including:

-   Atomic number
-   Degree
-   Formal charge
-   Hybridization
-   Number of attached hydrogens
-   Aromaticity
-   Ring membership
-   Atomic mass

The executed notebook reports **150 node-feature dimensions**.

### Edge features

Each bond is represented using:

-   Single / double / triple / aromatic bond type
-   Conjugation
-   Ring membership
-   Stereo configuration

The notebook uses **10 edge-feature dimensions**.

### Molecular descriptors

The model also uses **18 global molecular descriptors**, including
physicochemical and structural properties such as LogP, TPSA, QED,
molecular weight, hydrogen bonding, rings, and other RDKit descriptors.

------------------------------------------------------------------------

## Model Architecture

The final evaluated model is the **Feature-Enriched GNN with early
fusion and multi-head attention**.

Conceptually:

``` text
                    Molecular SMILES
                           │
                           ▼
                    RDKit Processing
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
     Atom Features     Bond Features    Molecular
      150 dims           10 dims        Descriptors
                                            18 dims
          │                │                │
          └────────────────┼────────────────┘
                           ▼
                Feature-Enriched GNN
                           │
                 Message Passing /
                 Graph Representation
                           │
                           ▼
                Multi-Head Attention
                     Pooling
                           │
                           ▼
                 Fully Connected Layers
                           │
                           ▼
                  BBBP Prediction
```

### Key design choices

-   **Early fusion** of molecular descriptors with graph representations
-   **Multi-head attention pooling** to learn which atoms contribute
    most to the molecular representation
-   **3 graph/message-passing layers**
-   Hidden dimension: `128`
-   Attention heads: `2`
-   Attention dimension: `64`
-   Dropout: approximately `0.305`
-   Binary cross-entropy with logits
-   AdamW optimizer
-   Cosine Annealing Warm Restarts scheduler
-   Gradient clipping
-   Early stopping based on validation ROC-AUC

The final training run reports **341,036 trainable parameters**.

------------------------------------------------------------------------

## Hyperparameter Optimization

Optuna was used to search over:

-   Hidden dimension: `128, 256, 384`
-   Number of attention heads: `2, 4, 8`
-   Dropout: `0.1–0.4`
-   Learning rate: `5e-4–2e-3`
-   Weight decay: `1e-6–1e-3`

**20 trials** were run, with pruning enabled.

The best Optuna validation ROC-AUC reported in the notebook was:

**0.9077**

Best configuration:

``` text
hidden_dim    = 128
num_heads     = 2
dropout       = 0.3045
learning_rate = 0.0005027
weight_decay  = 0.0008125
```

------------------------------------------------------------------------

## Results

### Final Test Performance

The executed final model evaluation reports:

  Model                           ROC-AUC     Accuracy           F1
  -------------------------- ------------ ------------ ------------
  Random Forest                **0.9312**   **0.9020**   **0.9394**
  Basic GNN                        0.9040       0.8897       0.9313
  Late-Fusion GNN                  0.9124       0.8824       0.9250
  **Feature-Enriched GNN**     **0.9271**   **0.8995**   **0.9354**

### What the results show

The Feature-Enriched GNN:

-   Improves ROC-AUC by approximately **2.6% relative to the Basic GNN**
-   Improves ROC-AUC by approximately **1.6% relative to the Late-Fusion
    GNN**
-   Achieves **0.9271 ROC-AUC** and **0.9354 F1** on the test set
-   Performs very competitively with the Random Forest baseline

Importantly, the Random Forest baseline has slightly higher test ROC-AUC
and F1 than the Feature-Enriched GNN in the executed comparison.
Therefore, the result is best described as **competitive with the
strongest baseline**, rather than claiming that the GNN outperforms
every baseline.

------------------------------------------------------------------------

## Visualizations Generated

The notebook generates the following analysis artifacts:

1.  `1_property_distributions.png` --- molecular property distributions
2.  `2_sample_molecules.png` --- example BBB-penetrating and
    non-penetrating molecules
3.  `3_feature_enrichment.png` --- before/after feature enrichment
4.  `4_training_curves.png` --- training and validation behavior
5.  `5_confusion_matrix.png` --- test-set confusion matrix
6.  `6_roc_curve.png` --- ROC curve
7.  `7_model_comparison.png` --- comparison against baseline models

These visualizations make the notebook useful as an evaluation artifact
as well as an implementation notebook.

------------------------------------------------------------------------

## Baselines

Three baselines are evaluated:

### Random Forest

Uses **2048-bit Morgan fingerprints** as molecular representations.

### Basic GNN

Uses graph structure without the full feature-enrichment strategy.

### Late-Fusion GNN

Combines learned graph representations with molecular descriptors later
in the network.

The main experiment investigates whether **richer molecular features and
earlier integration** improve graph-based BBBP prediction.

------------------------------------------------------------------------

## Tech Stack

-   Python
-   PyTorch
-   PyTorch Geometric
-   RDKit
-   DeepChem
-   scikit-learn
-   NumPy
-   pandas
-   Matplotlib
-   Seaborn
-   Optuna

------------------------------------------------------------------------

## Reproducibility

The notebook sets:

``` python
torch.manual_seed(42)
np.random.seed(42)
```

The project can be run in Google Colab or a local Python environment
with the required dependencies installed.

### Main dependencies

``` bash
pip install torch torch-geometric deepchem scikit-learn pandas matplotlib seaborn rdkit optuna
```

The notebook downloads the BBBP dataset when it is not already available
locally.

------------------------------------------------------------------------

## Notebook

The main project artifact is:

**`afml_project_v1.ipynb`**

The notebook contains the complete workflow:

``` text
Dataset loading
      ↓
Exploratory analysis
      ↓
Molecular feature extraction
      ↓
SMILES → graph conversion
      ↓
Feature-enriched GNN
      ↓
Hyperparameter optimization
      ↓
Training + early stopping
      ↓
Test evaluation
      ↓
Baseline comparison
      ↓
Visualization + results
```

------------------------------------------------------------------------

## Key Takeaway

The project explores whether **combining molecular graph structure with
global physicochemical information** can improve BBB penetration
prediction.

The final Feature-Enriched GNN achieves:

> **0.9271 ROC-AUC \| 0.8995 Accuracy \| 0.9354 F1**

and provides a structured comparison against both graph-based and
fingerprint-based baselines.
