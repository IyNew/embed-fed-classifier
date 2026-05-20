
# Shadow Model Training Dataset Splitting Protocol

## 1. Overall Data Pool Definition

Assume a total sample pool of 6000 samples, to be divided into three mutually non-overlapping sub-pools:

| Pool Name | Symbol | Sample Count | Purpose |
|-----------|--------|-------------|---------|
| Member Pool | D_mem | 3600 | Training the target model (source of member samples) |
| Non-member Pool | D_non | 1200 | Samples unseen by the target model (source of non-member samples) |
| Adversary Auxiliary Pool | D_aux | 1200 | Fully held by the adversary, used for training shadow models |

**Constraints**:
- D_mem ∩ D_non ∩ D_aux = ∅
- D_mem ∪ D_non ∪ D_aux = All 6000 samples
- The label distributions of the three pools should remain approximately consistent (can be ensured via stratified sampling)

---

## 2. Target Sample Selection

### 2.1 Member Target Samples

Randomly sample 10 samples from D_mem:

```
T_mem = {x_m1, x_m2, ..., x_m10} ⊂ D_mem
```

### 2.2 Non-member Target Samples

Randomly sample 10 samples from D_non:

```
T_non = {x_n1, x_n2, ..., x_n10} ⊂ D_non
```

### 2.3 Total Target Sample Set

```
T_all = T_mem ∪ T_non = {x_1, x_2, ..., x_20}
```

| Target Sample | Ground Truth Label | Source |
|---------------|-------------------|--------|
| x_1 ... x_10 | MEMBER | D_mem |
| x_11 ... x_20 | NON-MEMBER | D_non |

> **Important**: Once these 20 target samples are selected, they remain fixed throughout the entire shadow model training process and must never appear in the training data of shadow models (except when deliberately included as the IN condition).

---

## 3. Construction of the Shadow Base Pool

### 3.1 Shadow Base Pool

Randomly sample 800 samples from D_aux as the base data source for shadow model training:

```
Pool_shadow ⊂ D_aux,  |Pool_shadow| = 800
```

This pool has absolutely no intersection with T_all (because D_aux is independent of both D_mem and D_non).

### 3.2 Pre-generation of Shadow Training Subsets

Randomly draw 32 training subsets from Pool_shadow with replacement, each subset of size 500:

```
S_1, S_2, ..., S_32,  where each |S_j| = 500,  S_j ⊂ Pool_shadow
```

**Sampling Rules**:
- Each S_j is independently and randomly sampled from Pool_shadow by drawing 500 samples
- Overlap between different S_j is allowed (i.e., the same auxiliary sample may appear in multiple subsets)
- Use a fixed random seed to ensure experimental reproducibility

These 32 subsets will serve as the shared base training data for all 20 target samples.

---

## 4. Construction of Shadow Model Training Sets per Target Sample

For each target sample x_i ∈ T_all (i = 1, 2, ..., 20), 32 shadow training sets need to be constructed.

### 4.1 IN Training Sets (16 sets)

Used for training 16 shadow models that "have seen x_i":

| Training Set ID | Composition | Sample Count |
|----------------|-------------|-------------|
| Train_IN_i_1 | S_1 ∪ {x_i} | 501 |
| Train_IN_i_2 | S_2 ∪ {x_i} | 501 |
| ... | ... | ... |
| Train_IN_i_16 | S_16 ∪ {x_i} | 501 |

### 4.2 OUT Training Sets (16 sets)

Used for training 16 shadow models that "have not seen x_i":

| Training Set ID | Composition | Sample Count |
|----------------|-------------|-------------|
| Train_OUT_i_1 | S_17 | 500 |
| Train_OUT_i_2 | S_18 | 500 |
| ... | ... | ... |
| Train_OUT_i_16 | S_32 | 500 |

### 4.3 Totals

| Item | Count |
|------|-------|
| Target samples | 20 |
| Training sets per target sample | 32 (16 IN + 16 OUT) |
| Total training sets | 20 × 32 = 640 |
| Total shadow models | 640 |

---

## 5. Dataset Panorama

```
All Samples (6000)
├── D_mem (3600) ───────────────────────────── Target model training
│   └── T_mem (10) ─── Member target samples {x_1...x_10}
│
├── D_non (1200) ──────────────────────────── Target model testing
│   └── T_non (10) ─── Non-member target samples {x_11...x_20}
│
└── D_aux (1200) ──────────────────────────── Held by adversary
    └── Pool_shadow (800) ─────────────────── Shadow model base pool
        ├── S_1  ──→ Train_IN_i_1  = S_1  ∪ {x_i}
        ├── S_2  ──→ Train_IN_i_2  = S_2  ∪ {x_i}
        ├── ...
        ├── S_16 ──→ Train_IN_i_16 = S_16 ∪ {x_i}
        ├── S_17 ──→ Train_OUT_i_1  = S_17
        ├── S_18 ──→ Train_OUT_i_2  = S_18
        ├── ...
        └── S_32 ──→ Train_OUT_i_16 = S_32

(The above mapping from S_j to Train is repeated 20 times, once for each x_i)
```

---

## 6. Key Constraints and Verification Checklist

Before commencing training, the following conditions must be verified:

- [ ] D_mem ∩ D_non = ∅
- [ ] D_aux ∩ (D_mem ∪ D_non) = ∅
- [ ] T_all ∩ Pool_shadow = ∅
- [ ] For any i, x_i ∉ S_j (naturally satisfied when S_j is used for OUT; when S_j is used for IN, x_i originates from D_mem or D_non while S_j originates from D_aux, hence also satisfied)
- [ ] S_1 through S_32 are all independently and randomly sampled from Pool_shadow
- [ ] All shadow models have an architecture identical to the target model
- [ ] All hyperparameters (learning rate, number of epochs, batch size, optimizer) of shadow model training are identical to those of the target model

---

## 7. Suggested File Organization After Shadow Model Training

```
shadow_models/
├── target_x_01/
│   ├── IN/
│   │   ├── model_in_01.pt
│   │   ├── model_in_02.pt
│   │   └── ...
│   │   └── model_in_16.pt
│   └── OUT/
│       ├── model_out_01.pt
│       ├── model_out_02.pt
│       └── ...
│       └── model_out_16.pt
├── target_x_02/
│   └── ...
└── target_x_20/
    └── ...

losses/
├── target_x_01/
│   ├── losses_in.npy    # shape: (16,)
│   └── losses_out.npy   # shape: (16,)
├── target_x_02/
│   └── ...
└── target_x_20/
    └── ...
```

