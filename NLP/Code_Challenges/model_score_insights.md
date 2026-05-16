# Credit Card Approval Model — Score Insights

## Model Score Summary

| Metric | Value |
|---|---|
| Accuracy | 0.8333 |
| Precision | 0.8462 |
| Recall | 0.8088 |
| F1-Score | 0.8271 |
| AUC-ROC | 0.8866 |

---

## Reconstructing the Confusion Matrix

From Precision and Recall we can work backwards to exact counts on the 138 test samples:

```
                  Predicted: Approved   Predicted: Rejected
Actual: Approved       TP = 55               FN = 13
Actual: Rejected       FP = 10               TN = 60
```

Every insight below flows from these four numbers.

---

## Insights

### 1. The Model is Genuinely Good, Not Just Lucky

Accuracy of **83.3%** on a dataset where 55.5% of applicants are approved (383 approved vs 307 rejected across 690 rows). A naive model that always predicts "approve" would score only 55.5%. Our model is 28 percentage points above that baseline — the learning is real.

---

### 2. The Model is More Cautious About False Approvals Than False Rejections

| Error Type | Count | Meaning |
|---|---|---|
| False Positives (FP) | 10 | Approved someone who should be rejected |
| False Negatives (FN) | 13 | Rejected someone who should be approved |

The model makes more mistakes in the *"miss good customers"* direction than the *"let bad customers through"* direction. The higher **Precision (0.846)** vs **Recall (0.809)** confirms this — when it says *approve*, it is right more often than it catches *all* approvable applicants.

> **Business read:** the model leans slightly conservative, which is generally the right instinct for credit risk — approving a defaulter costs more than turning away a creditworthy applicant.

---

### 3. AUC-ROC of 0.887 is the Strongest Signal of Quality

Accuracy is measured at one threshold (0.5). AUC-ROC at **0.887** means: pick any random approved and any random rejected applicant — the model assigns the approved one a higher probability **88.7% of the time**.

This robustness across all possible thresholds tells you the model has genuinely learned the underlying pattern, not just memorised a boundary.

| AUC-ROC | Interpretation |
|---|---|
| 1.00 | Perfect separation |
| **0.887** | **Strong — our model** |
| 0.50 | No better than random guessing |

---

### 4. The Precision–Recall Gap Points to Where Improvement Would Come

```
Precision = 0.846   Recall = 0.809   Gap = 0.037
```

A gap of 3.7 points means the model is leaving **13 creditworthy applicants** unserved (the FNs). To close this gap you could:

- **Lower the decision threshold** from 0.5 → e.g. 0.4 — approve more borderline cases (Recall ↑ but Precision ↓)
- **Add more informative features** so the model better distinguishes borderline-good from borderline-bad applicants
- **Try a more powerful model** (e.g. Random Forest, Gradient Boosting) that can learn non-linear boundaries

---

### 5. F1 of 0.827 Confirms the Model is Well-Balanced

F1 is the harmonic mean of Precision and Recall — it collapses toward zero if either metric is poor. A score of **0.827** confirms the model handles both the *quality of approvals* and *coverage of good applicants* reasonably well, with no severe trade-off in either direction.

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
   = 2 × (0.846 × 0.809) / (0.846 + 0.809)
   = 0.827
```

---

## Verdict

> The model is a solid, conservative credit screener that correctly handles **83% of applications**, rarely approves bad applicants (**Precision 84.6%**), and misses roughly 1 in 5 creditworthy applicants (**Recall 80.9%**) — a sensible trade-off for a bank prioritising default risk over growth.

---

## Metric Decision Guide

| Question | Metric to Use |
|---|---|
| Overall, how often is it right? | Accuracy |
| When it approves, can I trust it? | Precision |
| Does it catch all good applicants? | Recall |
| Balance between the two? | F1-Score |
| How good is it across all thresholds? | AUC-ROC |
