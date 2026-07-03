## Prerequisites
 
You have the following assets ready:
 
| Asset | Quantity | Description |
|-------|----------|-------------|
| Target model `f_target` | 1 | The model you want to attack |
| Shadow models | 640 (or 32 per target sample) | Models with identical architecture to target |
| Target samples | 1000 (example) | 500 real members + 500 real non-members (labels known to you for evaluation) |
 
**Key assumption about shadow model organization:**
For **each** target sample `x_i`, there are 32 shadow models:
- **16 IN models**: trained on datasets that **include** `x_i`
- **16 OUT models**: trained on datasets that **exclude** `x_i`
 
So `640 = 20 target samples × 32 shadow models`, or you may have trained 32,000 shadow models for 1,000 target samples. The following steps assume you have the IN/OUT shadow model groups ready for each target sample.
 
---
 
## Step 1: Collect Target Model Outputs for All Target Samples
 
```python
# Pseudocode
target_losses = {}
 
for x_i in Target_All:  # all 1000 target samples
    # Compute cross-entropy loss of target model on x_i
    loss_i = f_target.evaluate_loss(x_i, true_label_i)
    target_losses[x_i] = loss_i
```
 
**Result**: For each target sample `x_i`, you have a single loss value `ℓ_target_i` from the target model.
 
---
 
## Step 2: Compute IN/OUT Loss Distributions Using Shadow Models
 
For **each** target sample `x_i`:
 
```python
# For the 16 IN shadow models of this sample
losses_IN_i = []
for shadow_model in shadow_models_IN[x_i]:  # 16 models
    loss = shadow_model.evaluate_loss(x_i, true_label_i)
    losses_IN_i.append(loss)
 
# For the 16 OUT shadow models of this sample
losses_OUT_i = []
for shadow_model in shadow_models_OUT[x_i]:  # 16 models
    loss = shadow_model.evaluate_loss(x_i, true_label_i)
    losses_OUT_i.append(loss)
```
 
**Result**: For each `x_i`, you now have:
- `losses_IN_i`: a list of length 16
- `losses_OUT_i`: a list of length 16
 
---
 
## Step 3: Fit Gaussian Distributions and Compute Likelihood Ratios
 
For each target sample `x_i`:
 
```python
from scipy.stats import norm
 
# Fit Gaussian distributions via maximum likelihood estimation
mu_in, std_in = norm.fit(losses_IN_i)
mu_out, std_out = norm.fit(losses_OUT_i)
 
# Compute likelihood of the target model's loss under each distribution
likelihood_in = norm.pdf(target_losses[x_i], loc=mu_in, scale=std_in)
likelihood_out = norm.pdf(target_losses[x_i], loc=mu_out, scale=std_out)
 
# Compute likelihood ratio score
lambda_i = likelihood_in / likelihood_out
```
 
**Result**: For each target sample `x_i`, you obtain a **continuous leakage score** `Λ(x_i)`.
 
> **Why likelihood ratio instead of raw loss difference?**
>
> Suppose a sample `x_i` is inherently easy to classify (e.g., a clear white cat). Even if the model has never seen it, the loss would still be low. Judging by the absolute value of `ℓ_target_i` alone would falsely flag it as a member.
>
> The likelihood ratio answers: **Is this loss value more likely to come from the IN distribution or the OUT distribution?** It automatically calibrates for baseline differences caused by sample difficulty.
 
---
 
## Step 4: Evaluate Attack at a Controlled FPR
 
You now have two sets of likelihood ratio scores:
- `Λ_MEM`: scores for 500 real member samples
- `Λ_NON`: scores for 500 real non-member samples
 
### 4.1 Compute Threshold at a Specified FPR
 
```python
import numpy as np
 
target_fpr = 0.01  # 1% false positive rate
 
# Find threshold in non-member scores: only 1% of non-members exceed this
threshold = np.percentile(Λ_NON, 100 * (1 - target_fpr))
# Equivalent to: P(Λ_non > threshold) = 0.01
```
 
### 4.2 Compute Corresponding TPR
 
```python
# Fraction of member scores exceeding the threshold
tpr = np.mean(Λ_MEM > threshold)
print(f"TPR @ {target_fpr*100}% FPR: {tpr:.4f}")
```
 
### 4.3 Compute AUC (Optional but Recommended)
 
```python
from sklearn.metrics import roc_auc_score
 
# Construct labels: member = 1, non-member = 0
y_true = [1] * 500 + [0] * 500
y_score = list(Λ_MEM) + list(Λ_NON)
 
auc = roc_auc_score(y_true, y_score)
print(f"AUC: {auc:.4f}")
```
 
---
 
## Step 5: Visualization and Diagnosis
 
### 5.1 Single-Sample Diagnosis (Pick a Typical Member)
 
```python
import matplotlib.pyplot as plt
 
# Take a certain member sample x_k as an example
x_range = np.linspace(
    min(min(losses_IN_k), min(losses_OUT_k), target_losses[x_k]) - 0.5,
    max(max(losses_IN_k), max(losses_OUT_k), target_losses[x_k]) + 0.5,
    200
)
 
plt.plot(x_range, norm.pdf(x_range, mu_in, std_in), 'r-', label='IN distribution')
plt.plot(x_range, norm.pdf(x_range, mu_out, std_out), 'b-', label='OUT distribution')
plt.axvline(target_losses[x_k], color='black', linestyle='--',
            label=f'Target loss = {target_losses[x_k]:.3f}')
plt.legend()
plt.title(f'Sample x_k: Λ = {lambda_k:.2f}')
plt.savefig('single_sample_diagnosis.png')
```
 
If the IN and OUT distributions heavily overlap and the target loss falls in the overlapping region, leakage for that sample is low. Conversely, if they are clearly separated, leakage is high.
 
### 5.2 Global Score Distribution
 
```python
plt.hist(Λ_NON, bins=30, alpha=0.5, label='Non-members', density=True)
plt.hist(Λ_MEM, bins=30, alpha=0.5, label='Members', density=True)
plt.axvline(threshold, color='red', linestyle='--',
            label=f'Threshold (FPR=1%)')
plt.xlabel('Likelihood Ratio Λ')
plt.ylabel('Density')
plt.legend()
plt.savefig('score_distribution.png')
```
 
### 5.3 ROC Curve
 
```python
from sklearn.metrics import roc_curve
 
fpr_vals, tpr_vals, thresholds = roc_curve(y_true, y_score)
plt.plot(fpr_vals, tpr_vals)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xscale('log')  # Better visualization of low-FPR region
plt.title(f'ROC (AUC = {auc:.3f})')
plt.savefig('roc_curve.png')
```
 
---
 
## Step 6: Output Final Quantification Report
 
```python
def compute_tpr_at_fpr(mem_scores, non_scores, fpr_target):
    threshold = np.percentile(non_scores, 100 * (1 - fpr_target))
    return np.mean(mem_scores > threshold)
 
print("=" * 50)
print("Membership Inference Attack Report")
print("=" * 50)
print(f"Target model: Centralized Training (No DP)")
print(f"Number of target samples: {len(Target_All)}")
print(f"  - Members: {len(Target_MEM)}")
print(f"  - Non-members: {len(Target_NON)}")
print(f"Shadow models per sample: 32 (16 IN + 16 OUT)")
print("-" * 50)
print(f"TPR @ 0.1% FPR: {compute_tpr_at_fpr(Λ_MEM, Λ_NON, 0.001):.4f}")
print(f"TPR @ 1% FPR:   {tpr:.4f}")
print(f"TPR @ 10% FPR:  {compute_tpr_at_fpr(Λ_MEM, Λ_NON, 0.1):.4f}")
print(f"AUC: {auc:.4f}")
print("=" * 50)
```
 
---
 
## Summary
 
| Step | Action | Output |
|------|--------|--------|
| 1 | Compute target model loss on all target samples | `ℓ_target_i` for each sample |
| 2 | Compute IN/OUT shadow model losses per sample | `losses_IN_i`, `losses_OUT_i` (16 values each) |
| 3 | Fit Gaussians and compute likelihood ratios | `Λ(x_i)` continuous leakage score per sample |
| 4 | Compute threshold at fixed FPR and evaluate TPR/AUC | `TPR@1%FPR`, `AUC` |
| 5 | Visualization and diagnosis | Distribution plots, ROC curve |
| 6 | Final report | Quantified leakage metrics |
 
---
 
At this point, you have completed membership leakage quantification for a specific target model. To compare across different scenarios (e.g., centralized with DP, federated learning), simply swap in a new target model while **reusing the same set of shadow models** (provided shadow training configurations match the target training paradigm), and repeat Steps 1–6 to obtain comparable TPR@1%FPR and AUC values.