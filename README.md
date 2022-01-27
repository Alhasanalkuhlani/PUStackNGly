# PUStackNGly: Positive-Unlabeled and Stacking Learning for N-linked Glycosylation Site Prediction.
ABSTRACT:
N-linked glycosylation is one of the most common protein post-translation modifcations (PTMs) in humans where the Asparagine (N) amino acid of the protein is attached to the glycan. It is involved in most biological processes and associated with various human diseases as diabetes, cancer, coronavirus, inﬂuenza, and Alzheimer’s. Accordingly, identifying N-linked glycosylation sites will be benefcial to understanding the system and mechanism of glycosylation. Due to the experimental challenges of glycosylation site identifcation, machine learning becomes very important to predict the glycosylation sites. This paper proposes a novel N-linked glycosylation predictor based on bagging positive-unlabeled (PU) learning and stacking ensemble machine learning (PUStackNGly). In the proposed PUStackNGly, comprehensive sequence and structural-based features are extracted using different feature extraction descriptors. Then, ensemble-based feature selection is employed to select the most signifcant and stable features. The ensemble bagging PU learning selects the reliable negative samples from the unlabeled samples using four supervised learning methods (support vector machines, random forest, logistic regression, and XGBoost). Then, stacking ensemble learning is applied using four base classifers: logistic regression, artifcial neural networks, random forest, and support vector machine. The experiments results show that PUStackNGly has a promising predicting performance compared to supervised learning methods. Furthermore, the proposed PUStackNgly outperforms the existing N-linked glycosylation prediction tools on an independent dataset with 95.11% accuracy, 100% recall 80.7% precision, 89.32% F1 score, 96.93% AUC, and 0.87 MCC.
