# 🎯 Section 3: Classification

Predicting a category: will this user buy after seeing an ad? Six classifiers on the same Social Network Ads dataset, then a head-to-head comparison.

> Open `ML-Classics-in-R.Rproj` in RStudio first: it makes paths such as `data/mall.csv` resolve. Packages are loaded with `library()` at the top of each script.

| # | Script | Topic | Dataset | Notes |
|:-:|--------|-------|---------|-------|
| 1 | [`01-logistic-regression.R`](01-logistic-regression.R) | Logistic regression (with SVM and random forest for comparison) | `social-network-ads.csv` |  |
| 2 | [`02-kernel-svm.R`](02-kernel-svm.R) | Kernel SVM | `social-network-ads.csv` | ⚠️ the decision-boundary plot needs `ElemStatLearn` (removed from CRAN) |
| 3 | [`03-naive-bayes.R`](03-naive-bayes.R) | Naïve Bayes classifier | `social-network-ads.csv` |  |
| 4 | [`04-decision-tree-classification.R`](04-decision-tree-classification.R) | Decision tree classifier | `social-network-ads.csv` |  |
| 5 | [`05-knn-random-forest-xgboost.R`](05-knn-random-forest-xgboost.R) | KNN, random forest and XGBoost | `social-network-ads.csv` |  |
| 6 | [`06-model-comparison-knn-rf-xgboost.R`](06-model-comparison-knn-rf-xgboost.R) | Model comparison: KNN vs random forest vs XGBoost | `social-network-ads.csv` |  |

---

[← Regression](../02-regression/) &nbsp;·&nbsp; [Clustering →](../04-clustering/)

*Built by [Evgenii Matveev](https://github.com/evgeniimatveev) · Data Analyst / Analytics & MLOps Engineer*
