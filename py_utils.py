######### My custom python functions

### preprocess_train_valid_test: this preprocesses my training, validation, and test datasets
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

### benchmark_dataloader(): this evaluates the time required to feed my tensor into dataloader, based on the utilization of a different number of worker threads
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

### This is immediately followed by:
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



