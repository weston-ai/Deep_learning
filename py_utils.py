# ============================
# 📦 py_utils.py
# General-purpose ML utilities
# ============================

# ============================
# 🔁 DATA TRANSFORM UTILITIES
# ============================

##### Preprocesses my training, validation, and test datasets
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def preprocess_train_valid_test(X_train, X_valid, X_test, features_num, features_cat):
    """
    Applies independent preprocessing (Z-score normalization & One-Hot Encoding)
    to training and validation datasets, while extracting exact transformed feature names.

    Parameters:
        X_train (pd.DataFrame): Training dataset.
        X_valid (pd.DataFrame): Validation dataset.
        X_test (pd.DataFrame): Test dataset
        features_num (list): List of numerical feature names.
        features_cat (list): List of categorical feature names.

    Returns:
        X_train_transformed (np.ndarray): Transformed training data (array format).
        X_valid_transformed (np.ndarray): Transformed validation data (array format).
        feature_names (list): Ordered list of feature names after preprocessing.
    """

    # Independent numerical transformations for training and validation data
    preprocessor_train = make_column_transformer(
        (StandardScaler(with_mean=True, with_std=True), features_num),
        (OneHotEncoder(handle_unknown='ignore'), features_cat)
    )
    preprocessor_valid = make_column_transformer(
        (StandardScaler(with_mean=True, with_std=True), features_num),
        (OneHotEncoder(handle_unknown='ignore'), features_cat)
    )

    preprocessor_test = make_column_transformer(
        (StandardScaler(with_mean=True, with_std=True), features_num),
        (OneHotEncoder(handle_unknown='ignore'), features_cat)
    )

    # Fit and transform training data
    X_train_transformed = preprocessor_train.fit_transform(X_train)

    # Fit and transform validation data
    X_valid_transformed = preprocessor_valid.fit_transform(X_valid)

    # Fit and transform test data
    X_test_transformed = preprocessor_test.fit_transform(X_test)

    # Extract numerical feature names (somewhat redundant here)
    num_feature_names = features_num

    # Extract OneHotEncoded feature names
    ohe = preprocessor_train.named_transformers_['onehotencoder']
    cat_feature_names = ohe.get_feature_names_out(features_cat).tolist()

    # Combine all feature names
    final_feature_names = num_feature_names + cat_feature_names

    return X_train_transformed, X_valid_transformed, X_test_transformed, final_feature_names

# ============================
# 📦 CUSTOM DATASET WRAPPER
# ============================

##### Create a class that can convert training, validation, and testing datasets to tensors, while accounting for index problems and sparse matricies due to OneHotEncoding
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class TensorAsTabular(Dataset):
    def __init__(self, X, y):
        """
        A PyTorch Dataset that takes in preprocessed tabular data (e.g., one-hot encoded + scaled)
        and returns X and y tensors suitable for binary classification.

        Parameters:
            X (scipy sparse matrix or NumPy array): Transformed features.
            y (pandas Series, NumPy array, or list): Labels.
        """
        # Reset index defensively (this resets the index to 0, 1, 2, 3, etc, if it detects pd.Series type, which is what was created in the test_train_split)
        if isinstance(y, pd.Series):
            y = y.reset_index(drop=True)

        # Convert X to dense matrix if it's sparse (i.e. from OneHotEncoder)
        if hasattr(X, "toarray"):
            X = X.toarray()

        self.X_tensor = torch.from_numpy(X.astype(np.float32)) #faster than torch.tensor()
        self.y_tensor = torch.from_numpy(np.array(y).astype(np.float32)).unsqueeze(1) #faster than torch.tensor()

    def __len__(self):
        return len(self.y_tensor) # returns length of self.y_tensor (mandatory for DataLoader)

    def __getitem__(self, idx):
        return self.X_tensor[idx], self.y_tensor[idx] # Fetches a single sample (X, y pair) at a given index, and will do this for the batch size defined in DataLoader()

    # Note: the returned tuple from TensorAsTabular() is (feature_values_for_this_sample, target_value_for_this_sample)


##### Function to convert metrics to numpy array if ndarray (supporting JSON)
import numpy as np

def convert_metrics(metrics):
    return [list(np.asarray(m)) if isinstance(m, (list, np.ndarray)) else m for m in metrics]


# ============================
# 📊 MODEL PERFORMANCE
# ============================

##### Function that measures spead of dataloader with different number of workers
import time
from torch.utils.data import DataLoader

def benchmark_dataloader(dataset, batch_size=64, num_workers_list=[0, 2, 4, 6, 8]):
    """
    Benchmarks the time it takes to iterate through one full pass (epoch)
    over the given dataset using different numbers of DataLoader workers.

    Parameters:
        dataset (torch.utils.data.Dataset): Dataset to benchmark
        batch_size (int): Batch size for DataLoader
        num_workers_list (list): List of num_workers values to test

    Returns:
        dict: Mapping of num_workers → time taken (in seconds)
    """
    results = {}

    print(f"\nDataset size: {len(dataset)} samples")

    for nw in num_workers_list:
        print(f"  ➤ Testing DataLoader with num_workers = {nw}")
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=nw, shuffle=True, pin_memory=True)

        start = time.time()

        # Just iterate through all batches (simulate one epoch)
        for batch in loader:
            pass

        duration = time.time() - start
        results[nw] = duration
        print(f"     ⏱️  Time taken: {duration:.4f} seconds")

    return results

### the benchmark_dataloader() fxn is immediately followed by:
# """
# # Dictionary of datasets for looping (datasets were defined with TensorAsTabular() above)
# datasets = {
#     "Train": train_dataset,
#     "Valid": valid_dataset,
#     "Test": test_dataset
# }
#
# # Check number of available logical worker threads
# import os
# print(os.cpu_count())  # shows me 8
#
# # Run benchmark on each dataset
# for name, ds in datasets.items():
#     print(f"\n==============================")
#     print(f"📊 Benchmarking {name} Dataset")
#     print(f"==============================")
#     benchmark_dataloader(ds, batch_size=64, num_workers_list=[0, 2, 4, 6, 8]) # adding zero runs without multithreading
# """


# ============================
# 🧹 FILE MANAGEMENT
# ============================

##### Function to delete best model when better model detected: used in run_experiments() fxn
import os
import glob

def delete_best_model_plots(tag: str, directory: str):
    """
    Delete old best model plots tagged with `tag` (e.g., "val_loss", "roc_auc") in the given directory.
    Now handles filenames that include a timestamp (e.g., *_best_val_loss_20250408_143015.png),
    and standardizes metric names (e.g., "F1 Score" → "f1_score").
    """
    plot_names = ["Loss", "Accuracy", "Precision", "Recall", "F1 Score", "Specificity", "ROC AUC", "PR AUC"]

    for name in plot_names:
        normalized_name = name.lower().replace(" ", "_")  # make it match saved filename
        pattern = os.path.join(directory, f"*_{normalized_name}_best_{tag}_*.png")
        for file in glob.glob(pattern):
            try:
                os.remove(file)
                print(f"🗑️ Deleted old plot: {file}")
            except FileNotFoundError:
                pass
            except Exception as e:
                print(f"⚠️ Warning deleting plot '{file}': {e}")


### Function to delete files post-analysis (e.g. [".png", ".csv", ".json", ".txt", ".pt"] before we reiterate the model, or if we don't need them anymore
import os
import glob

def delete_files_in_directories(base_path, subdirs, extensions):
    """
    Deletes files with specified extensions in given subdirectories under the base path.

    Parameters:
        base_path (str): The root directory path.
        subdirs (list): List of relative subdirectory paths.
        extensions (list): List of file extensions to delete (e.g., ['.png', '.csv']).
    """
    for subdir in subdirs:
        full_dir = os.path.join(base_path, subdir)
        if not os.path.exists(full_dir):
            print(f"⚠️ Directory does not exist: {full_dir}")
            continue

        for ext in extensions:
            pattern = os.path.join(full_dir, f"*{ext}")
            files = glob.glob(pattern)

            for file_path in files:
                try:
                    os.remove(file_path)
                    print(f"🗑️ Deleted: {file_path}")
                except Exception as e:
                    print(f"❌ Error deleting {file_path}: {e}")


# ============================
# 🖼️ IMAGE ENCODING
# ============================

##### Function to convert standard images to base64 (required to embed images in html
import base64
import os

# Function to convert standard images to base64 (required to create embedded images in the html)
def image_to_base64_str(image_path):
    assert os.path.exists(image_path), f"❌ Image not found: {image_path}"
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# ============================
# 📊 STATISTICAL TESTS
# ============================

##### McNemar's test to compare predictive performance of two models
import numpy as np
# Optional: import inside function (helps modularity)
from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar

def run_mcnemars_test(nn_preds, logreg_preds, true_labels):
    """
    Runs McNemar's test to assess whether two classifiers differ significantly in prediction error.

    Parameters:
        nn_preds (np.array): Binary predictions from neural network.
        logreg_preds (np.array): Binary predictions from logistic regression.
        true_labels (np.array): Ground truth binary labels.

    Returns:
        dict: Dictionary with McNemar test summary.
    """
    # Ensure arrays are 1D NumPy arrays
    nn_preds = np.asarray(nn_preds).flatten()
    logreg_preds = np.asarray(logreg_preds).flatten()
    true_labels = np.asarray(true_labels).flatten()

    # Compute b and c
    b = np.sum((nn_preds == true_labels) & (logreg_preds != true_labels))  # NN correct, LogReg wrong
    c = np.sum((logreg_preds == true_labels) & (nn_preds != true_labels))  # LogReg correct, NN wrong

    # McNemar's 2x2 contingency table
    table = [[0, b],
             [c, 0]]

    # Run McNemar's test with continuity correction
    result = sm_mcnemar(table, exact=False, correction=True)

    return {
        "b": int(b),
        "c": int(c),
        "test_statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "significant": result.pvalue < 0.05 # force native Python bool
    }


# ============================
# ✨ EXPORT CONTROL (optional)
# ============================

__all__ = [
    # Preprocessing
    "preprocess_train_valid_test",

    # Dataset class
    "TensorAsTabular",

    # Utilities
    "convert_metrics",
    "benchmark_dataloader",

    # File management
    "delete_best_model_plots",

    # Image encoding
    "image_to_base64_str",

    # Statistical test
    "run_mcnemars_test"
]

