#  LIME Stability Analysis for Text Classification

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Team:** ModelMiners (Abdul Aahad Qureshi, Khyzar Baig)  
**Course:** Interpretable Machine Learning (iML) - Winter 2025/26  
**Supervisor:** Prof. Dr. Marius Lindauer, Lukas Fehring  
**Institution:** Leibniz Universität Hannover

---

##  Project Overview

This project systematically analyzes the **stability of LIME (Local Interpretable Model-agnostic Explanations)** for text classification tasks. LIME explanations suffer from instability due to random sampling, which undermines trust in high-stakes applications.

### Research Questions
1. How does the number of perturbation samples affect LIME stability?
2. Does sentence length influence explanation consistency?
3. Are explanations more stable on simple vs. complex models?
4. How stable are feature effect magnitudes (not just rankings)?

### Key Findings
 **Sample Size:** Need ≥1000 samples for stable explanations (94% agreement)  
 **Sentence Length:** Medium-length sentences (8-15 words) most stable (99%)  
 **Model Complexity:** Simple models 7% more stable than complex models  
 **Feature Effects:** High sign consistency (91%) but moderate magnitude stability

---

##  Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/khyzar/iml_lime_stability.git
cd iml_lime_stability

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Experiments

**Option 1: Run All Experiments**
```bash
python experiments/run_all.py
```

**Option 2: Run Individual Experiments**
```bash
# Experiment 1: Sample size effect
python experiments/exp1_samples.py

# Experiment 2: Sentence length effect
python experiments/exp2_length.py

# Experiment 3: Model complexity effect
python experiments/exp3_models.py

# Experiment 4: Feature effects stability
python experiments/exp4_features.py
```

**Option 3: Use Jupyter Notebook** (Recommended for exploration)
```bash
jupyter notebook notebooks/LIME_Stability_Analysis.ipynb
```

---

## 📊 Results

All results are saved in `results/`:

- **Figures:** `results/figures/*.png` (300 DPI, publication quality)
- **Metrics:** `results/metrics/*.csv` (raw data for all experiments)

### Sample Results

| Experiment | Key Finding | Metric |
|------------|-------------|--------|
| Exp 1: Sample Size | 2000 samples → 94% agreement | Top-3 Agreement |
| Exp 2: Sentence Length | Medium (8-15 words) best | 99.0% agreement |
| Exp 3: Model Complexity | LogReg 7% more stable | 92% vs 85% |
| Exp 4: Feature Effects | Sign stable, magnitude varies | 91% sign consistency |

---

## 📂 Repository Structure
```
iml_lime_stability/
├── notebooks/           # Jupyter notebook with complete analysis
├── src/                # Core Python modules
├── experiments/        # Experiment scripts
├── results/           # All outputs (figures + CSV)
├── docs/              # Report and poster
└── requirements.txt   # Dependencies
```

---

##  Methodology

### Dataset
- **SST-2** (Stanford Sentiment Treebank)
- Binary sentiment classification
- 67,349 training samples, 872 test samples

### Models
1. **Logistic Regression** (simple baseline)
   - TF-IDF features (5000 dimensions)
   - Linear decision boundary

2. **DistilBERT** (complex model)
   - Pre-trained on SST-2
   - 66M parameters

### LIME Configuration
- **Perturbation method:** Random word masking
- **Surrogate model:** Linear regression (bag-of-words)
- **num_samples tested:** 100, 250, 500, 1000, 2000
- **Runs per sentence:** 30 (to measure stability)
- **Seeds:** 5 different random seeds [42, 123, 456, 789, 1000]

### Stability Metrics
1. **Top-K Agreement:** % overlap of top-3 words across runs
2. **Rank Correlation:** Spearman ρ of word rankings
3. **Coefficient of Variation:** CV of importance scores

---

##  Experiments

### Experiment 1: Effect of num_samples
- **Goal:** Find optimal number of perturbation samples
- **Setup:** 50 sentences × 5 sample sizes × 30 runs × 5 seeds
- **Result:** Strong positive correlation (p < 0.001)

### Experiment 2: Effect of Sentence Length
- **Goal:** Test if input complexity affects stability
- **Setup:** 3 length groups (short/medium/long) × 15 sentences × 30 runs × 5 seeds
- **Result:** Medium-length most stable (ANOVA p < 0.01)

### Experiment 3: Effect of Model Complexity
- **Goal:** Compare simple vs. complex models
- **Setup:** 2 models × 15 sentences × 30 runs × 5 seeds
- **Result:** LogReg significantly more stable (t-test p < 0.001)

### Experiment 4: Feature Effect Stability
- **Goal:** Analyze magnitude stability (not just rankings)
- **Setup:** 20 sentences × 30 runs
- **Result:** High sign consistency (91%), moderate magnitude CV (0.18)

---

##  References

1. **Ribeiro et al. (2016).** "Why Should I Trust You?": Explaining the Predictions of Any Classifier. [arXiv:1602.04938](https://arxiv.org/abs/1602.04938)

2. **Slack et al. (2020).** Fooling LIME and SHAP: Adversarial Attacks on Post-hoc Explanation Methods. [arXiv:1911.02508](https://arxiv.org/abs/1911.02508)

3. **Alvarez-Melis & Jaakkola (2018).** On the Robustness of Interpretability Methods. [arXiv:1806.08049](https://arxiv.org/abs/1806.08049)

4. **Krishna et al. (2022).** The Disagreement Problem in Explainable Machine Learning. [arXiv:2202.01602](https://arxiv.org/abs/2202.01602)

---

##  Team

**ModelMiners**
- Abdul Aahad Qureshi - [GitHub](https://github.com/Ahad002)
- Khyzar Baig

**Supervisor:** Prof. Dr. Marius Lindauer, Lukas Fehring

---

##  License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

**If you found this useful, please star the repository!**
