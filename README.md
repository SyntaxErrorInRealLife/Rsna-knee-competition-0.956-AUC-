# 🦴 RSNA Knee Abnormality Detection
## CoAtNet + ConvNeXt-Small Ensemble

### 📌 Overview

This notebook implements an ensemble of **CoAtNet and ConvNeXt-Small** for the [RSNA Knee Abnormality Detection competition](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection).

The pipeline combines complementary pretrained-model inference workflows to generate **study-level predictions for 12 knee abnormalities** from MRI data. It uses retrained model checkpoints, rank-based prediction blending, and supporting preprocessing assets to produce a competition-compatible submission.

### 🔬 Methodology

- **CoAtNet + Multiple-Instance Learning (MIL)** — Study-level prediction using the Raptor CoAtNet/MIL inference approach.
- **ConvNeXt-Small** — A complementary image-model pipeline for additional predictive information.
- **Retrained checkpoints** — The model weights used here are our retrained checkpoints, rather than the original public baseline weights.
- **Ensemble blending** — Combines predictions from complementary model pipelines using rank-based blending.
- **Study-level inference** — Aggregates MRI evidence to predict all 12 target abnormalities.
- **Submission validation** — Checks prediction values and formats the output for the competition.

### 🎯 Prediction Targets

The ensemble predicts the following abnormalities:

| Category | Targets |
|:--|:--|
| Ligaments | ACL, MCL |
| Menisci | Medial Meniscus, Lateral Meniscus |
| Osteoarthritis | Medial OA, Lateral OA, PF OA |
| Other abnormalities | Effusion, Synovitis, Baker's cyst, Contusion, Fracture |

---

## 🙏 Credits & Acknowledgements

### 🧠 Model Architecture & Inference

This work builds on the Raptor CoAtNet/MIL inference recipe and public baseline weights provided by [dreaddevelopment](https://www.kaggle.com/datasets/dreaddevelopment/raptor-knee-widedense).

**The checkpoints used in this notebook are our retrained models, not the original public baseline weights.**

The CoAtNet backbone is provided through [`timm`](https://github.com/huggingface/pytorch-image-models).

### 📝 Report-Derived Weak Supervision

Our weak-supervision work draws on the following community resources, combined with our own report-derived labels:

- [**stevenleehans** — RSNA Knee LLM Report Labels](https://www.kaggle.com/datasets/stevenleehans/rsna-knee-llm-report-labels)
- [**pilkwang** — RSNA Knee LLM Labels](https://www.kaggle.com/datasets/pilkwang/rsna-knee-llm-labels)
- [**riadmohamed42** — JEV Knee Labels Verification](https://www.kaggle.com/code/riadmohamed42/jev-knee-labels-verification)

These resources informed the development and verification of report-derived supervision. **Source reports and study-level label tables are not included in this notebook.**

### 🏥 Osteoarthritis Initiative (OAI) Acknowledgement

Data and/or research tools used in the preparation of this work were obtained and analyzed from the controlled-access datasets distributed through the **Osteoarthritis Initiative (OAI)**, a data repository housed within the NIMH Data Archive (NDA).

OAI is a collaborative informatics resource created by the **National Institute of Mental Health (NIMH)** and the **National Institute of Arthritis and Musculoskeletal and Skin Diseases (NIAMS)** to accelerate biomarker identification, scientific investigation, and osteoarthritis drug development.

**Dataset identifier:** `10.15154/0hcg-f676`

- [NDA Shared Study](https://nda.nih.gov/study.html?id=3407)
- [Dataset DOI: 10.15154/0hcg-f676](https://doi.org/10.15154/0hcg-f676)
- [Official NDA Manuscript Preparation and Acknowledgement Guidance](https://nda.nih.gov/nda/manuscript-preparation)

### 📜 Competition Rules & Data

Competition data is used in accordance with the [RSNA Knee Abnormality Detection competition rules](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules).

External datasets, pretrained weights, and supporting assets remain subject to their respective access conditions, licensing terms, and applicable competition rules.

---

## ⚙️ Reproducibility

To reproduce inference:

1. Attach the official competition dataset.
2. Attach all required model checkpoints and supporting assets.
3. Enable GPU acceleration.
4. Ensure the required dependencies and preprocessing resources are available.
5. Run the notebook and verify the generated submission.

Once all required assets are attached, the final inference workflow is intended to run without internet access.

**Note:** Actual results depend on the checkpoint versions, preprocessing configuration, and execution environment. The public leaderboard performance of the referenced approaches should not be interpreted as a guarantee of the score obtained from this notebook.
