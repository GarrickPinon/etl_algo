# etl-pipeline-algo

---

## 📘 Project Overview

**Garrick Piñón — ETL Algorithm**

This branch contains reproducible ETL scaffolding and benchmark results on the TCP-DI dataset.  
It validates that medium-mode ETL improves downstream classification performance across three models.

---

## 🧪 Experiment Setup

- **Dataset**: UCI Adult (used as a sandbox for TCP-DI benchmarking)
- **ETL Mode**: Medium  
  - log1p skew correction  
  - uniform “?” → NaN normalization  
  - median imputation for numeric features  
- **Models Evaluated**: Logistic Regression, Random Forest, Gradient Boosting  
- **Metrics Tracked**: F1 Score, ROC AUC

---

## 📊 TCP-DI Benchmark Results

| Model              | F1 (Raw) | F1 (ETL) | Δ F1   | ROC AUC (Raw) | ROC AUC (ETL) | Δ AUC  |
| ------------------ | -------- | -------- | ------ | ------------- | ------------- | ------ |
| LogisticRegression | 0.830    | 0.835    | +0.005 | 0.890         | 0.892         | +0.002 |
| RandomForest       | 0.840    | 0.845    | +0.005 | 0.895         | 0.903         | +0.008 |
| GradientBoosting   | 0.838    | 0.841    | +0.003 | 0.892         | 0.894         | +0.002 |

---

## 🎨 Visuals Logged to W&B

- SHAP Summary Plot (Bar)
- SHAP Waterfall (Single Prediction)
- Confusion Matrix
- ROC Curve

All visuals are logged to [Weights & Biases](https://wandb.ai/garrick-hult-mban-algo/tcpdi-etl) under run `rf_full_eval_with_explain`.

---

## 🧠 Key Takeaways

- Medium-mode ETL yields consistent lift across all models.
- Random Forest benefits most from skew correction and imputation.
- ROC AUC improvements suggest sharper threshold calibration.
- ETL scaffolding is validated for future fault-injection experiments.

---

## 📚 Textbook vs. Codebook Framing

This experiment is designed to serve both institutional review and team deployment:

### Textbook (Institutional Optics)
- Validates that medium-mode ETL yields measurable lift across legacy classifiers.
- Demonstrates audit-grade reproducibility with metrics, visuals, and coverage diagnostics.
- Anchors future fault-injection and LLM-repair experiments with a clean baseline.

### Codebook (Team Deployment)
- Modular ETL pipeline with plug-and-play preprocessing logic.
- Starter cells for SHAP, confusion matrix, and ROC curve generation.
- W&B logging scaffold for metrics and media, ready for sweep integration.
- Flat notebook structure for fast onboarding and reproducible reruns.

---

## 📐 Mathematical Framing

We benchmark ETL impact and repair efficacy using six core formulations:

---

### 1. **Incremental ETL Lift**  
$$
\Delta_{\text{ETL}} = \mathbb{E}[M_{\text{ETL}}] - \mathbb{E}[M_{\text{Raw}}]
$$  
- \( M \): Model performance metric (e.g., F1, AUC)  
- \( \mathbb{E} \): Expected value across randomized trials

---

### 2. **Feature-Level Marginal Impact**  
$$
\delta_j = \left.\frac{\partial M}{\partial x_j}\right|_{\text{ETL}} - \left.\frac{\partial M}{\partial x_j}\right|_{\text{Raw}}
$$  
- \( x_j \): Feature \( j \)  
- \( \partial M / \partial x_j \): Sensitivity of model performance to feature \( j \)

---

### 3. **Jacobian Delta Across Features**  
$$
\Delta \mathbf{J} = \nabla_{\mathbf{x}} M \big|_{\text{ETL}} - \nabla_{\mathbf{x}} M \big|_{\text{Raw}}
$$  
- \( \nabla_{\mathbf{x}} M \): Gradient vector of model performance w.r.t. all features  
- \( \mathbf{x} \): Feature vector

---

### 4. **Cascade Length**  
$$
C = \sum_{t=1}^{T} E_t \cdot I_t
$$  
- \( E_t \): Binary indicator of error presence at step \( t \)  
- \( I_t \): Impact factor of error at step \( t \)  
- \( T \): Total number of transformation steps

---

### 5. **Repair Success Rate**  
$$
R = \frac{1}{N} \sum_{i=1}^{N} T_i
$$  
- \( T_i = 1 \) if error \( e_i \) is correctly repaired, else 0  
- \( N \): Total number of injected errors

---

### 6. **Error Propagation Probability**  
$$
P(\text{propagate beyond } m \mid E_k) = \prod_{i=k}^{m} \left(1 - P(D_i)\right)
$$  
- \( E_k \): Error injected at step \( k \)  
- \( D_i \): Detection event at step \( i \)  
- \( P(D_i) \): Probability that an error is detected at step \( i \)  
- \( m \): Final step in propagation window  
- \( \prod \): Product over steps \( i = k \) to \( m \)

---

These formulations anchor our evaluation of validation strategies under synthetic fault injection. They quantify the lift from ETL preprocessing, the containment of error cascades, and the efficacy of LLM-assisted repair. Each metric is logged per run and aggregated across datasets to assess robustness, precision, and runtime trade-offs.

---
## 🧪 Codebook Translation

The above math is operationalized as:

```python
for mode in ["raw", "etl"]:
    pipe = Pipeline([
        ("pre", etl_medium if mode=="etl" else "passthrough"),
        ("clf", RandomForestClassifier())
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, pipe.predict_proba(X_test)[:,1])
    print(f"{mode.upper()} → F1: {f1:.3f}, AUC: {auc:.3f}")

```

🔒 Full experiment archive available upon request for collaborators or reviewers.

