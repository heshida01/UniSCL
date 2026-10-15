# UniSCL

UniSCL is a unified multi-task framework for **predicting the subcellular localization of mRNA and lncRNA** and for **analysing informative sequence features**.

Based on RNALocate v3.0, we build a curated human dataset with rigorous quality control and label harmonization into three major compartments: **chromatin**, **cytoplasm**, and **extracellular exosome**. UniSCL integrates:

- **120 RF-selected hand-crafted features**
- **CNN head–tail sequence embeddings** (fixed-length 5′ and 3′ segments)
- A **shared MLP backbone** with **task-specific heads** for mRNA and lncRNA


---

![architecture](https://github.com/lijingtju/UniSCL/blob/main/flowchart.jpg)

---

## Requirements

- Python ≥ 3.8  
- PyTorch (GPU recommended)
- NumPy
- pandas
- scikit-learn

Repository structure:

```
.
├── data/
│   ├── SCL_test_80.fasta       # Example test FASTA (sequences)
│   └── ...                     # Other raw / processed data
├── feature/
│   ├── test_top120.csv         # Example 120-dim hand-crafted feature file
│   └── ...                     # Feature matrices
├── models/
│   ├── best_model.pt           # Trained UniSCL weights
│   ├── scaler_mean.npy         # StandardScaler mean (120-dim)
│   └── scaler_scale.npy        # StandardScaler scale (120-dim)
├── select_features.py          # Random forest-based feature selection
├── test.py                     # Inference script (no training)
└── ...
```

Data format
1. Sequence file (--input_fasta)

FASTA format (recommended):
```
>seq_0001
ACGUGA...
>seq_0002
UUGC...
```

Or CSV with a sequence column (and optionally task, label):
```
id,sequence,task,label
seq_0001,ACGUGA...,lncRNA,0
seq_0002,UUGC...,mRNA,1
...
```

If task / label are present in the CSV they will be used automatically.

2. Feature file (--input_fea)

test_top120.csv must contain 120 hand-crafted features per sample.
Non-feature columns such as task, label, id, sequence are allowed and will be ignored when building the feature matrix.

Feature selection

UniSCL computes a rich panel of sequence-derived descriptors and then uses a random forest to select the top 120 informative features.

To run feature selection:
```
python select_features.py
```

This script:

Loads the full feature set.

Slect top 120 features based on *feature_importance_ranking.csv*.

Once you have:
```
SCL_test_80.fasta (or an equivalent sequence file)

test_top120.csv (120-dim selected feature file)
```
you can run:
```
python test.py --input_fasta ./data/SCL_test_80.fasta --input_fea ./feature/test_top120.csv --model_path  ./models/best_model.pt --scaler_scale ./models/scaler_scale.npy --scaler_mean ./models/scaler_mean.npy --output o.csv
```

Arguments
```
--input_fasta
```
Path to the sequence file (FASTA or CSV with a sequence column).
```
--input_fea
```
Path to the 120-dim feature CSV.
```
--model_path
```
Path to the trained UniSCL model (state_dict).
```
--scaler_mean, --scaler_scale
```
Paths to the stored StandardScaler parameters used during training.
If omitted, test.py will automatically look for them in the same directory as model_path.
```
--output
```
Output CSV file. It contains:
```
id

sequence

task

(optional) label — if provided in the input

pred_label — predicted class (0: chromatin, 1: cytoplasm, 2: extracellular exosome)

prob_class_0, prob_class_1, prob_class_2 — predicted probabilities
```
On labeled data, the script also prints macro / weighted F1 and accuracy on the console.

