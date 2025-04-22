🧠 Hypertension Risk Prediction with Dynamic Binary Classifier
Overview
This project implements a modular, extensible machine learning pipeline to classify whether a patient developed hypertension over a 24-month follow-up period. The pipeline leverages:

🔬 Deep neural networks (PyTorch)

📈 Logistic regression benchmarking (scikit-learn + statsmodels)

🧪 Statistical comparison (McNemar's Test)

📊 Full HTML and PDF reporting with plots, metrics, and model summaries

The workflow is designed for scientific reproducibility, clinical relevance, and clean documentation of model performance.

💻 Key Features
Dynamic MLP architecture with grid search over:

Number of layers

Hidden units

Dropout rates

Batch sizes

He (Kaiming) initialization, LeakyReLU activations, batch normalization, dropout regularization

Early stopping based on validation loss

Comprehensive performance metrics (Accuracy, Precision, Recall, F1, Specificity, ROC AUC, PR AUC)

Logistic Regression comparison (ROC/PR curves, coefficient summary)

Statistical evaluation with McNemar's test

Auto-generated HTML and PDF report with embedded plots and interpretation

Clean modular structure with py_utils.py helper library

📂 Project Structure
bash
Copy
Edit
.
├── data/
│   └── hypertension_study_MAIN_20250408.csv
├── py_utils.py                  # Utility functions for preprocessing, metrics, plots, etc.
├── main_pipeline.py            # Main training/testing/reporting script (your script)
├── outputs/
│   ├── plots/
│   │   ├── best_training_models/
│   │   └── tested_models/
│   ├── tables/
│   └── Report_for_DynamicBinaryClassifier_MLP_<timestamp>.html
│   └── Report_for_DynamicBinaryClassifier_MLP_<timestamp>.pdf
🧪 Data Description
Subjects: 60,000 patients

Target: Ever_had_HT — whether the patient ever had hypertension during the 24-month period

Features:

Demographics: Race, Age Group, Sex

Socioeconomic: Income, Education

Lifestyle: BMI, Alcohol, Smoking

Location: Home city

Treatment: Hypertensive drug use

Data is split using stratified sampling:

37,500 (Training)

11,250 (Validation)

11,250 (Test)

📄 Outputs
📊 Plots: Per-epoch metrics (loss, ROC AUC, PR AUC, etc.)

📈 Tables: Final model summaries (CSV, JSON)

🧪 Statistical Test: McNemar's test results (JSON)

📋 Reports: Human-readable HTML + PDF report of experiment

📦 Dependencies
torch, scikit-learn, statsmodels

matplotlib, seaborn, jinja2, weasyprint, pandas, numpy

📚 References
PyTorch Docs: https://pytorch.org/docs/

McNemar's Test: https://en.wikipedia.org/wiki/McNemar%27s_test

WeasyPrint (HTML → PDF): https://weasyprint.org/
