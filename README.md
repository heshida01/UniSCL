# UniSCL

UniSCL is a unified multi-task framework for **predicting the subcellular localization of mRNA and lncRNA** and for **analysing informative sequence features**.

Based on RNALocate v3.0, we build a curated human dataset with rigorous quality control and label harmonization into three major compartments: **chromatin**, **cytoplasm**, and **extracellular exosome**. UniSCL integrates:

- **120 RF-selected hand-crafted features**
- **CNN head–tail sequence embeddings** (fixed-length 5′ and 3′ segments)
- A **shared MLP backbone** with **task-specific heads** for mRNA and lncRNA

Experiments show that UniSCL achieves higher weighted and macro F1 than:
- MLPs using only hand-crafted features
- LSTM / Transformer encoders using only sequence embeddings

Feature importance and group-wise analyses highlight higher-order k-mer and physicochemical pseudo k-mer features as major contributors, providing an interpretable view of the model’s decision basis.

---

![architecture](https://github.com/lijingtju/UniSCL/blob/main/flowchart.jpg)

---

## Requirements

- Python ≥ 3.8  
- PyTorch (GPU recommended)
- NumPy
- pandas
- scikit-learn

You can install all dependencies with:

```bash
pip install -r requirements.txt
