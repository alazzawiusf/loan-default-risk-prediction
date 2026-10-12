# Loan Default Risk Prediction

Predicting small-business loan default from application-time data with a leakage-safe machine-learning workflow: K-Means applicant segmentation, Decision Tree, Neural Network and Logistic Regression models, cross-validated selection and a held-out test evaluation.

**Author:** Ahmed Alazzawi — graduate student, Computer Science / AI, University of South Florida
**Reports:** [Executive report (Markdown)](Loan_Default_Risk_Prediction_Executive_Report.md) · [Executive report (PDF)](Loan_Default_Risk_Prediction_Executive_Report.pdf) · [Notebook](Loan_Default_Risk_Prediction.ipynb)

---

## Business problem

A lender reviewing small-business loan applications wants to flag higher-risk applicants **before** credit is approved, so analysts can focus their review time where it matters most. The project estimates each applicant's probability of default and asks three questions:

1. How well can default be predicted from information available at application?
2. Which characteristics does the model rely on?
3. Are there natural applicant segments with different risk levels?

The output is an analytical prototype that ranks applications by risk. It is not a production credit-decision system.

## Dataset

- 2,000 anonymised small-business loan applications, 37.1% of which defaulted.
- Eight predictors: business age, annual revenue, employees, industry sector (Retail, Services, Manufacturing, Tech), credit score, missed payments in the prior year, owner experience and requested loan amount. Target: `Defaulted` (1/0).
- Supplied as university course material. No missing values, no duplicates, no personal identifiers. **The CSV is not redistributed** because its licence is undocumented — see [`data/README.md`](data/README.md).

## Methods and tools

Python 3.13 · pandas · NumPy · scikit-learn · matplotlib · seaborn · Jupyter

| Step | What was done |
|---|---|
| Data quality | Missing / duplicate / range / consistency / IQR-outlier checks; outliers retained as plausible |
| Split | Stratified 80/20 train/test split **before** any learned transformation |
| EDA | Default rates and correlations on the training set only |
| Segmentation | K-Means on standardised application features (target excluded); k chosen by silhouette |
| Models | Majority-class baseline, Logistic Regression, Decision Tree (depth 4), Neural Network (MLP 3-5); Full (8 features) and Reduced (4 features) variants |
| Validation | scikit-learn `Pipeline` + 5-fold stratified CV on the training set; selection rule fixed in advance (highest CV ROC-AUC, prefer the more interpretable model within 0.01) |
| Final test | Selected model evaluated once on the 400-row hold-out set; bootstrap 95% CI for ROC-AUC |
| Interpretation | Permutation importance, tree rules, logistic coefficients, decision-threshold trade-off (out-of-fold) |

## Key findings (verified)

- **Selected model: Logistic Regression (Full).** Test ROC-AUC **0.831** (95% CI 0.784–0.873), accuracy 77.8%, recall 75.7%, precision 67.9%, specificity 79.0% at a 0.5 threshold.
- **Model families are statistically tied.** Logistic Regression, Decision Tree and Neural Network all score 0.803–0.806 CV ROC-AUC (SD ≈ 0.03), so the simplest, most explainable model was chosen.
- **Credit score is the dominant signal, owner experience second.** Shuffling credit score cuts test ROC-AUC by 0.166; owner experience by 0.064; every other feature by less than 0.01.
- **Tech's high raw default rate (68%) is explained by its applicant profile** (young firms, weaker credit): sector adds no predictive value once other attributes are known.
- **Four applicant segments** with default rates of 8%, 16%, 52% and 74% (training), holding up on the unseen test set (8%, 8%, 54%, 75%).
- **Threshold is a policy lever.** At 0.5 the model flags 39% of applicants and catches 68% of defaulters (out-of-fold); at 0.30 it flags 50% and catches 84%.

All results are predictive associations, not causal effects.

## Model performance

**Cross-validation (training set, 5 folds, mean)** — used for selection

| Model | ROC-AUC | Accuracy | Precision | Recall | Specificity | F1 |
|---|---|---|---|---|---|---|
| Baseline (majority class) | 0.500 | 0.629 | 0.000 | 0.000 | 1.000 | 0.000 |
| **Logistic Regression (Full)** ✔ | **0.803** | 0.741 | 0.644 | 0.682 | 0.775 | 0.661 |
| Decision Tree (Full) | 0.806 | 0.748 | 0.649 | 0.700 | 0.776 | 0.673 |
| Decision Tree (Reduced) | 0.793 | 0.742 | 0.641 | 0.695 | 0.769 | 0.666 |
| Neural Network (Full) | 0.804 | 0.748 | 0.647 | 0.719 | 0.764 | 0.679 |
| Neural Network (Reduced) | 0.784 | 0.747 | 0.630 | 0.781 | 0.727 | 0.697 |

**Hold-out test set (400 applications)** — selected model is the headline; others shown for transparency only

| Model | ROC-AUC | Accuracy | Precision | Recall | Specificity | F1 |
|---|---|---|---|---|---|---|
| **Logistic Regression (Full)** ✔ | **0.831** | 0.778 | 0.679 | 0.757 | 0.790 | 0.716 |
| Decision Tree (Full) | 0.841 | 0.780 | 0.695 | 0.723 | 0.813 | 0.709 |
| Decision Tree (Reduced) | 0.843 | 0.773 | 0.682 | 0.723 | 0.802 | 0.702 |
| Neural Network (Full) | 0.837 | 0.775 | 0.667 | 0.784 | 0.770 | 0.720 |
| Neural Network (Reduced) | 0.820 | 0.783 | 0.656 | 0.865 | 0.734 | 0.746 |

Test-set differences between models are within sampling noise (all fall inside the selected model's 95% CI) and were not used for selection.

<p align="center">
  <img src="figures/fig06_cv_model_comparison.png" width="49%" alt="Cross-validated ROC-AUC by model">
  <img src="figures/fig05_segment_default_rates.png" width="49%" alt="Default rate by applicant segment">
</p>
<p align="center">
  <img src="figures/fig07_test_roc_and_confusion_matrix.png" width="90%" alt="Test ROC curves and confusion matrix">
</p>

## Audit and corrections

This repository is a corrected version of a course submission. The audit found:

| Issue in original | Correction |
|---|---|
| `Defaulted` (the target) was included in the K-Means input, and each run's cluster labels were fed into the next run | Clustering uses application features only; fitted on training data |
| Cluster labels (derived from the target) were used as predictors in the "Full" models | Removed from all models |
| Scaler fitted on all data before cross-validation | Preprocessing inside `Pipeline`, refit per fold |
| No hold-out test set; winner chosen on a 0.002 AUC gap | 80/20 split; pre-declared selection rule; single test evaluation with CI |
| "Probability Measure" (mean predicted probability) and "Entropy" (computed from precision/NPV) reported as performance metrics | Replaced with standard metrics and confusion matrix |
| Reduced feature set described as chosen by "predictive importance" without evidence | Re-described as an analyst-chosen simplicity trade-off |
| Personal student ID used as random seed and Google Drive paths | Seed 42, relative paths |

Re-running the original code showed the leak **did not inflate** the original model scores (the tree never used the cluster columns), but it invalidated the segmentation results. Corrected scores are of similar size and are now backed by a held-out test. Details are in Section 5 of the notebook.

## Repository structure

```
├── Loan_Default_Risk_Prediction.ipynb                 # executed, documented analysis
├── Loan_Default_Risk_Prediction_Executive_Report.md   # business report (GitHub-readable)
├── Loan_Default_Risk_Prediction_Executive_Report.pdf  # same report, 7-page PDF
├── data/README.md                                     # how to obtain the dataset
├── figures/                                           # all charts, generated by the notebook
├── results/                                           # metrics.json + CV / test result tables
├── requirements.txt
└── .gitignore
```

## Reproduce

```bash
git clone <this-repo-url> && cd <repo>
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# place the dataset at data/LoanDefaultPredictionDataset_Classification.csv (see data/README.md)
jupyter nbconvert --to notebook --execute --inplace Loan_Default_Risk_Prediction.ipynb
```

All randomness is seeded (`SEED = 42`); two consecutive full runs produced byte-identical `results/metrics.json`. Runtime is under a minute on a laptop.

## Limitations

Undocumented (possibly synthetic) data; small sample; random rather than out-of-time split; original hyper-parameters kept without tuning; calibration and fairness not assessed; timing of `MissedPaymentsLastYear` assumed to precede the application. See the [executive report](Loan_Default_Risk_Prediction_Executive_Report.md#6-limitations-and-next-steps) for details.
