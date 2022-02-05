# PUStackNGly: Positive-Unlabeled and Stacking Learning for N-linked Glycosylation Site Prediction.
PUStackNGly is a novel N-linked glycosylation predictor based on bagging positive-unlabeled (PU) learning and stacking ensemble machine learning (PUStackNGly). In the proposed PUStackNGly, comprehensive sequence and structural-based features are extracted using different feature extraction descriptors. Then, ensemble-based feature selection is employed to select the most signifcant and stable features. The ensemble bagging PU learning selects the reliable negative samples from the unlabeled samples using four supervised learning methods (support vector machines, random forest, logistic regression, and XGBoost). Then, stacking ensemble learning is applied using four base classifers: logistic regression, artifcial neural networks, random forest, and support vector machine. The experiments results show that PUStackNGly has a promising predicting performance compared to supervised learning methods. 

The data extraction, preprocessing, and samples construction are implemented using R 4.0.3 while the remaining stages of PUStackNGly are implemented in Python 3.8.5.
# Cite as
A. Alkuhlani, W. Gad, M. Roushdy and A. -B. M. Salem, "PUStackNGly: Positive-Unlabeled and Stacking Learning for N-Linked Glycosylation Site Prediction," in IEEE Access, vol. 10, pp. 12702-12713, 2022, doi: 10.1109/ACCESS.2022.3146395.
