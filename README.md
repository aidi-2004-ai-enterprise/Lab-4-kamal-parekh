# Lab-4- binary classification for bankruptcy predition.
dataset:  Company Bankruptcy Prediction – (https://www.kaggle.com/datasets/fedesoriano/company-bankruptcy-prediction

risk assesment domain

predicting weather a company will go bankrupt.
## 1. Choosing Initial Models
- **Benchmark**: Logistic Regression → interpretable, fast, and a baseline for comparison.
- **Advanced Models**: XGBoost + Random Forest → capture non-linear relationships, handle noisy financial data.
- **No Clustering**: Labels available, supervised learning more appropriate.

---

## 2. Data Pre-processing
- Standardize numeric features → needed for Logistic Regression’s stability.
- Tree-based models unaffected but scaling applied for pipeline consistency.
- No encoding required → dataset features are numeric.

---

## 3. Handling Class Imbalance
- Class skewed toward non-bankruptcy → prevents majority bias.
- Use class weighting + `scale_pos_weight` in XGBoost.
- Avoid heavy SMOTE → can distort borderline risky cases.
- Stratified splits → preserve class distribution.

---

## 4. Outlier Detection & Treatment
- Preserve true extreme values → may indicate genuine distress patterns.
- Remove only if data entry errors or impossible ratios.
- Keep rare bankruptcy points for minority class signal.

---

## 5. Sampling Bias Across Train/Test
- Apply PSI to detect shifts in feature distributions.
- Prevents models from exploiting dataset artifacts.

---

## 6. Data Normalization
- Apply `StandardScaler` for Logistic Regression.
- Tree-based models don’t require it, but keeps processing uniform.

---

## 7. Testing for Normality
- Not essential for chosen models (LR with regularization, tree models).
- Skewness handled via splits or coefficients.

---

## 8. Dimensionality Reduction (PCA)
- **Pro**: Mitigates noise & multicollinearity, may improve generalization.
- **Con**: Loses feature interpretability → problematic for regulatory cases.
- Decision: Skip PCA initially, rely on model-based selection.

---

## 9. Feature Engineering
- Possible: Financial ratio interactions.
- Limit to domain-relevant transformations to avoid noise.

---

## 10. Testing & Addressing Multicollinearity
- Use correlation matrix + VIF to detect redundancy.
- Drop highly correlated features for LR to improve interpretability.

---

## 11. Feature Selection
- Start with all features → filter via correlation & model importances.
- Lasso for LR, gain importance for XGBoost.
- Avoid too many features (overfit) or too few (underfit).

---

## 12. Hyperparameter Tuning
- Random Search for broad exploration.
- Bayesian Optimization for final fine-tuning.

---

## 13. Cross-Validation Strategy
- Stratified K-Fold → preserves class balance in folds.
- Reduces variance in performance estimates.

---

## 14. Evaluation Metrics
- Rare class → prioritize F1 and ROC-AUC.
- F1 balances precision & recall; ROC-AUC assesses ranking.
- Use `predict_proba` for ROC-AUC and threshold optimization.

---

## 15. Evaluating Drift & Model Degradation
- PSI between train/test for early drift detection.
- Enables timely retraining when features shift.

---

## 16. Interpreting Model Results
- SHAP values → explainable at local & global level.
- Critical for transparency in finance.

---











