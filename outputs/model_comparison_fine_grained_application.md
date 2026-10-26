# Model Comparison: Fine_Grained_Application

| Model | Accuracy | Macro-F1 | Precision (Macro) | Recall (Macro) | Weighted-F1 | Test Samples |
| --- | --- | --- | --- | --- | --- | --- |
| Random Forest (Baseline) | 0.8034 | 0.7251 | 0.7825 | 0.7038 | 0.7877 | 178 |
| MLP (Baseline) | 0.7191 | 0.4370 | 0.8588 | 0.5098 | 0.6072 | 178 |
| GraphSAGE (GNN) | 0.8596 | 0.8068 | 0.8683 | 0.7784 | 0.8499 | 178 |
| GAT (GNN) | 0.8596 | 0.7966 | 0.9021 | 0.7608 | 0.8449 | 178 |


*Evaluation performed under identical train/test splits with zero label leakage.*
