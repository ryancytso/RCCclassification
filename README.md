RCC Classification

This project implements a decision tree classifier from scratch in Python with a focus on interpretability and pathology-informed domain knowledge. The primary goal is to support renal cell carcinoma (RCC) subtype classification by explicitly modeling how morphological features and immunohistochemistry (IHC) features reduce diagnostic uncertainty.

Unlike off-the-shelf libraries (ex. scikit-learn), this implementation exposes  mathematical and algorithmic step—making ideal for explainability, debugging, and domain-driven customization.

----------------------------------------------------------------------------------------------------------------------------------

Adjust approach and pursued a "feature-gated" morphology first decision tree followed by IHC refinement

This project builds an interpretable, pathology-inspired pipeline to classify **renal cell carcinoma (RCC) subtypes** using:
1) **Morphological features first** (coarse diagnostic rules)
2) **IHC features second** (fine-grained confirmation/refinement)
3) To DO: **patient-level confidence scores** and **diagnostic entropy** (uncertainty)

Each row represents a **patient**, and each feature column is **binary** (e.g., present/absent, positive/negative).

In real pathology workflows:
- **Morphology** usually narrows the differential diagnosis first
- **IHC panels** are used next to confirm or resolve ambiguity
This project implements that logic explicitly rather than letting the model freely split on any feature at any time.


Author 
Ryan Tso
2026
