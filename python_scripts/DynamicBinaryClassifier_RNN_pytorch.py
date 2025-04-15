#### PROJECT DESCRIPTION: This project developed a deep feed-forward binary classifier in PyTorch to predict whether a patient (simulated patient) developed hypertension within a 24-month observation window. Our objective was to permute the network architecture across different combinations of layer depth, layer capacity, batch size, and dropout percentage, to identify the optimal performing architecture (according to ROC AUC and PR AUC). The total dataset contained 60,000 simulated human patients that were matched with appropriate individual and community confounding variables. The pre-training split used class-stratified sampling to assign 37,500 patients to the training dataset; 11,250 patients to the validation dataset; and 11,250 patients to the testing dataset. Each input vector was cross-sectional, with no temporal sequence. All data underwent a preprocessing pipeline using internal and external python modules. The neural net was then trained on a validation dataset, and the best performant training model was tested on a holdout test dataset. Those testing results were contrasted with the performance of a logistic regression model that was trained and tested with the same training and holdout test datasets. All results were summarized via HTML and saved to PDF.

import pandas as pd
pd.set_option('display.max_columns', None) # see all columns when we call head()
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split

#Import dataset
df=pd.read_csv("/home/cweston1/miniconda3/envs/PythonProject/datasets/my_custom_datasets/hypertension_study_MAIN_20250408.csv", na_values=[""])
# df.shape (445061; 27)

###########################################
############# PREPROCESS DATA #############
###########################################

# Update 'Smoked_one_pack_10_years_or_more' to be a string data type
df['Smoked_one_pack_10_years_or_more'] = df['Smoked_one_pack_10_years_or_more'].astype(str)
# df.iloc[:,10].value_counts()
# 0    11835
# 1     5290

# Create binary variable to indicate whether a person ever encountered hypertension
df['Ever_had_HT'] = df.groupby('Patient_ID')['Hypertension'].transform('max')
df['Ever_had_HT'].value_counts()
# 1 (302544); 0 (177456)

# Select the first observation for every patient (We only need one)
df = df.groupby('Patient_ID').first().reset_index(drop=True)
df.shape # (20000, 26)

# Convert -9999 values to NaN so that torch can ignore NaN values in the network model
df.replace(-9999, np.nan, inplace=True) # inplace=True updates the working dataframe

# Select only patients who remained in study for 24 months
df = df[df['Dropout_Month'] == 24]
# df.shape (17125, 27)

### Select features and target for each type of target analysis
# "Ever_had_hypertension"
EverHT = df.filter(items=['Race', 'Sex', 'Age_Group', 'Education',
       'Income', 'Treatment', 'City', 'Smoked_one_pack_10_years_or_more', 'Alcohol_daily', 'BMI', 'Ever_had_HT'])

### Create feature and target datasets
X = EverHT.copy()
y = X.pop('Ever_had_HT').astype('int8')

# Create training*validation and testing datasets
X_temp, X_test, y_temp, y_test = \
    train_test_split(X, y, stratify=y, test_size=0.20, random_state=33) # Use stratification especially when dealing with classification tasks that have imbalanced classes.

# Create training*validation and testing datasets
X_train, X_valid, y_train, y_valid = \
    train_test_split(X_temp, y_temp, stratify=y_temp, test_size=0.30, random_state=33) # Use stratification especially when dealing with classification tasks that have imbalanced classes.

# Export X_train, y_train, X_test, and y_test (for statistical analysis / log regression)
output_dir = "/home/cweston1/miniconda3/envs/PythonProject/datasets/Datasets_for_statistical_comparison/"
os.makedirs(output_dir, exist_ok=True)

X_train.to_csv(os.path.join(output_dir, "X_training_EverHadHT.csv"), index=False)
y_train.to_csv(os.path.join(output_dir, "y_training_EverHadHT.csv"), index=False)
X_test.to_csv(os.path.join(output_dir, "X_test_EverHadHT.csv"), index=False)
y_test.to_csv(os.path.join(output_dir, "y_test_EverHadHT.csv"), index=False)

# Convert y variable data to dataframes
y_train = pd.DataFrame(y_train)
y_test = pd.DataFrame(y_test)

### Preprocess numerical and categorical columns in the feature datasets
# Obtain a list of the column names for numerical features
features_num = list(X_train.select_dtypes(include=['number']).columns)

# Obtain a list of the column names for categorical features
features_cat = list(X_train.select_dtypes(exclude=['number']).columns)

## Preprocess X_train, X_valid, and X_test
# Import my custom fxn (preprocess_train_valid_test)
sys.path.append("/home/cweston1/miniconda3/envs/PythonProject/py_scripts/deep_learning_scripts/pytorch_scripts/vanilla_RNN/hypertension/")
from py_utils import preprocess_train_valid_test

# Apply fxn
X_train_transformed, X_valid_transformed, X_test_transformed, feature_names_from_network = preprocess_train_valid_test(X_train, X_valid, X_test, features_num, features_cat)

# Get the input shape
input_shape = [X_train_transformed.shape[1]]

#######################################################
################## NEURAL NET MODEL ###################
#######################################################
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, \
    average_precision_score, roc_curve, precision_recall_curve
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg') # necessary for PyCharm interactive mode
import seaborn as sns
import pickle
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML
import time
from datetime import datetime
from uuid import uuid4
import glob
import copy


# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

### CREATE TENSORS
# Create a class that can convert training, validation, and testing datasets to tensors, while accounting for index problems and sparse matricies due to OneHotEncoding
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

# Create tensors in tabular form
train_dataset = TensorAsTabular(X_train_transformed, y_train)
valid_dataset = TensorAsTabular(X_valid_transformed, y_valid)
test_dataset = TensorAsTabular(X_test_transformed, y_test)

#############################################################
##### DEFINE AND TRAIN PERMUTED NEURAL NETWORK MODELS #######
#############################################################

################################
##### SCRAPPY THINGS FIRST #####
# Function to support saving JSON
def convert_metrics(metrics):
    return [list(np.asarray(m)) if isinstance(m, (list, np.ndarray)) else m for m in metrics]

# Define global timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Define global output directory
base_output_dir = "/home/cweston1/miniconda3/envs/PythonProject/outputs/hypertension"

# Subdirectories for training outputs
best_plot_dir = os.path.join(base_output_dir, "plots/EverHadHT/best_training_models")
table_dir = os.path.join(base_output_dir, "tables", "EverHadHT")

# Create output directories (if they don't exist)...don't worry, it won't overwrite them
os.makedirs(table_dir, exist_ok=True)
os.makedirs(best_plot_dir, exist_ok=True)
###### END SCRAPPY THINGS ######
################################

### Create a dynamic binary classifier that will allow me to optimize the depth and capacity of the network, while also permute the batch size and dropout percentage
class DynamicBinaryClassifier(nn.Module):
    """
    A flexible binary classification model with a variable number of hidden layers
    and units per layer. Includes dropout, batch normalization, and LeakyReLU.
    """
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate=0.3):
        super(DynamicBinaryClassifier, self).__init__()
        self.layers = nn.ModuleList()

        # Build hidden layers dynamically
        for i in range(num_layers):
            in_size = input_size if i == 0 else hidden_size  # First layer uses input_size

            # Create and initialize layer
            linear = nn.Linear(in_size, hidden_size)
            init.kaiming_uniform_(linear.weight, nonlinearity='leaky_relu')
            init.zeros_(linear.bias)

            # Append the actual initialized layer
            self.layers.append(linear)
            self.layers.append(nn.BatchNorm1d(hidden_size))
            self.layers.append(nn.LeakyReLU(0.01))
            self.layers.append(nn.Dropout(dropout_rate))

        # Final output layer
        self.out = nn.Linear(hidden_size, 1)
        init.kaiming_uniform_(self.out.weight, nonlinearity='leaky_relu')
        init.zeros_(self.out.bias)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.out(x)  # raw logits — not passed through sigmoid

### Function to trains a PyTorch binary classification model using early stopping, tracks comprehensive training and validation metrics across epochs, and returns performance summaries for model evaluation and comparison.
def train_binary_classifier(model, train_data, valid_data,
                epochs: int = 100, patience: int = 7, learning_rate: float = 0.001):
    """
    Train a PyTorch binary classification model with early stopping and full metric tracking.

    Parameters:
        model (nn.Module): The PyTorch model to train.
        train_data (DataLoader): DataLoader for training data.
        valid_data (DataLoader): DataLoader for validation data.
        epochs (int): Maximum number of training epochs.
        patience (int): Number of epochs to wait for improvement before early stopping.
        learning_rate (float): Learning rate for optimizer.

    Returns:
        tuple: Metrics tracked across epochs.
    """
    # Determin GPU or CPU processing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    # Initialize optimizer and binary cross-entropy loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-7)
    criterion = nn.BCEWithLogitsLoss()

    # Set up variables for early stopping and best weights
    best_val_loss = float('inf')
    early_stopping_counter = 0
    best_model_weights = None

    # Initialize metric tracking lists
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    precisions, recalls, f1s, specificities = [], [], [], []
    roc_aucs, pr_aucs = [], []

    # Begin training loop
    for epoch in range(epochs):
        model.train()
        running_loss, running_acc = 0.0, 0.0

        # Training phase
        for X_batch, y_batch in train_data:
            X_batch = X_batch.to(device)
            y_batch = y_batch.contiguous().to(device=device, dtype=torch.float32).view(-1)

            optimizer.zero_grad()
            outputs = model(X_batch).view(-1) # ensures outputs is a 1D vector (required for BCE)

            loss = criterion(outputs, y_batch) # ensures y_batch is 1D vector
            loss.backward()
            optimizer.step()

            # Accumulate loss and accuracy
            probs = torch.sigmoid(outputs)  # converts logits to probabilities
            preds = (probs.detach() > 0.5).float() # if > 0.5, convert to 1, else 0
            running_acc += (preds == y_batch).float().sum().item()
            running_loss += loss.item() * len(X_batch)

        train_losses.append(running_loss / len(train_data.dataset))
        train_accs.append(running_acc / len(train_data.dataset))

        # Validation phase
        model.eval()
        val_loss, val_acc = 0.0, 0.0
        val_preds, val_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in valid_data:
                X_batch = X_batch.to(device)
                y_batch = y_batch.contiguous().to(device=device, dtype=torch.float32).view(-1)
                outputs = model(X_batch).view(-1)  # ensures outputs is a 1D vector

                loss = criterion(outputs, y_batch) # ensures y_batch is 1D vector
                val_loss += loss.item() * len(X_batch)

                probs = torch.sigmoid(outputs) # converts logits to probabilities
                preds = (probs > 0.5).float() # if > 0.5, convert to 1, else 0
                val_acc += (preds == y_batch).float().sum().item()

                val_preds.extend(probs.cpu().numpy()) # numpy only works on cpu
                val_labels.extend(y_batch.cpu().numpy()) # numpy only works on cpu

        # Compute binary classification metrics
        val_preds_bin = (np.array(val_preds) > 0.5).astype(int)
        val_labels_bin = np.array(val_labels).astype(int)

        precision = precision_score(val_labels_bin, val_preds_bin, zero_division=0)
        recall = recall_score(val_labels_bin, val_preds_bin, zero_division=0)
        f1 = f1_score(val_labels_bin, val_preds_bin, zero_division=0)
        cm = confusion_matrix(val_labels_bin, val_preds_bin, labels=[0, 1])

        # Specificity calculation
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        else:
            specificity = 0.0

        # ROC_AUC calculation
        try:
            roc_auc = roc_auc_score(val_labels_bin, val_preds)
        except ValueError:
            roc_auc = 0.0

        # PR_AUC calculation
        try:
            pr_auc = average_precision_score(val_labels_bin, val_preds)
        except ValueError:
            pr_auc = 0.0

        # Store validation metrics
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        specificities.append(specificity)
        roc_aucs.append(roc_auc)
        pr_aucs.append(pr_auc)

        val_losses.append(val_loss / len(valid_data.dataset))
        val_accs.append(val_acc / len(valid_data.dataset))

        # Print epoch summary
        print(
            f"Epoch {epoch + 1:03} | "
            f"Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_losses[-1]:.4f} | "
            f"Val Acc: {val_accs[-1]:.4f} | "
            f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | "
            f"Specificity: {specificity:.4f} | "
            f"ROC AUC: {roc_auc:.4f} | PR AUC: {pr_auc:.4f}"
        )

        # Early stopping logic
        if val_losses[-1] < best_val_loss - 0.001:
            best_val_loss = val_losses[-1]
            best_model_weights = copy.deepcopy(model.state_dict()) # defensible copy of weights; prevents overwrite
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print("Early stopping triggered.")
                break

    # Restore the best weights seen during training
    if best_model_weights:
        model.load_state_dict(best_model_weights)
        print("Restored best model weights from early stopping.")

    early_stopping_epoch = len(val_losses)

    # Return all tracked metrics
    return (train_losses, val_losses, train_accs, val_accs,
            precisions, recalls, f1s, specificities, roc_aucs, pr_aucs,
            early_stopping_epoch)

### Plot comparisons of metrics and save
def plot_and_save(metric1, metric2, name, filename, model_id, timestamp, label1="Metric 1", label2="Metric 2"):
    """
    Plot one or two metric curves and save the plot to a file.
    Falls back to legend_map if default labels are used.
    """
    # Legend fallback map
    legend_map = {
        "Loss": ("Training Loss", "Validation Loss"),
        "Accuracy": ("Training Accuracy", "Validation Accuracy"),
        "Precision": ("Validation Precision", "N/A"),
        "Recall": ("Validation Recall", "N/A"),
        "F1": ("Validation F1 Score", "N/A"),
        "Specificity": ("Validation Specificity", "N/A"),
        "ROC AUC": ("Validation ROC AUC", "N/A"),
        "PR AUC": ("Validation PR AUC", "N/A")
    }

    # Use fallback from legend_map only if labels are defaults
    if label1 == "Metric 1" and label2 == "Metric 2":
        label1, label2 = legend_map.get(name, (label1, label2))

    # Plotting
    plt.figure()
    plt.plot(metric1, label=label1)
    if metric2 is not None and label2 != "N/A":
        plt.plot(metric2, label=label2)

    plt.title(f"{name} over Epochs\nModel: {model_id} | Time: {timestamp}")
    plt.xlabel("Epoch")
    plt.ylabel(name)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.close()

def delete_best_model_plots(tag: str, directory: str):
    """
    Delete old best model plots tagged with `tag` (e.g., "val_loss", "roc_auc") in the given directory.
    Now handles filenames that include a timestamp (e.g., *_best_val_loss_20250408_143015.png).
    """
    plot_names = ["loss", "acc", "precision", "recall", "roc_auc"]
    for name in plot_names:
        pattern = os.path.join(directory, f"*_{name}_best_{tag}_*.png")  # updated to include wildcard for timestamp
        for file in glob.glob(pattern):
            try:
                os.remove(file)
            except FileNotFoundError:
                pass
            except Exception as e:
                print(f"⚠️ Warning deleting plot '{file}': {e}")

### Function to systematically train and evaluate binary classification models across combinations of hidden layers, hidden unit sizes, and batch sizes, collecting performance metrics for each configuration
def run_experiments(train_dataset, valid_dataset, input_size, timestamp, best_plot_dir, table_dir):
    """
    Trains multiple binary classification models across different hyperparameter settings:
    - Varying number of hidden layers
    - Varying number of hidden units
    - Varying batch sizes
    - Varying dropout rates

    Parameters:
        train_dataset (Dataset): The training dataset
        valid_dataset (Dataset): The validation dataset
        input_size (int): Number of input features

    Returns:
        results (list): A list of dictionaries containing hyperparameter settings and training metrics
    """
    import json
    # Define the hyperparameter grid to explore
    hidden_layer_options = [2, 3, 4]           # Number of hidden layers
    hidden_unit_sizes = [64, 128, 256, 512]    # Number of units per hidden layer
    batch_sizes = [128, 256, 512, 1024, 2048]  # Batch sizes for training
    dropout_options = [0.2, 0.3]               # Dropout rates

    results = []  # Store results for each model
    global_best_loss = float('inf')  # Tracks best final val_loss across ALL models
    global_best_roc_auc = float('-inf') # Tracks best final ROC AUC across ALL models
    best_model_filename = None
    best_model_auc_filename = None
    json_best_auc_path = None
    json_best_loss_path = None

    # Loop over all combinations
    for num_layers in hidden_layer_options:
        for hidden_size in hidden_unit_sizes:
            for batch_size in batch_sizes:
                for dropout_rate in dropout_options:
                    # Display current setting
                    print(f"\n🔧 Training model: Layers={num_layers}, Units={hidden_size}, "
                          f"Batch Size={batch_size}, Dropout={dropout_rate}")

                    # Create DataLoaders using current batch size
                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
                    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

                    # Initialize model with current architecture
                    model = DynamicBinaryClassifier(
                        input_size=input_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        dropout_rate=dropout_rate
                    )

                    # Train model and collect metrics
                    metrics = train_binary_classifier(
                        model,
                        train_loader,
                        valid_loader,
                        epochs=100,
                        patience=15,
                        learning_rate=0.001
                    )

                    # Define unique model ID
                    model_id = f"L{num_layers}_H{hidden_size}_B{batch_size}_D{int(dropout_rate * 100)}"

                    ## Save the model weights for the models with the best val_loss and roc_auc
                    final_val_loss = np.mean(metrics[1][-5:])  # mean of last 5 validation losses
                    final_roc_auc = np.mean(metrics[8][-5:])  # mean of last 5 ROC AUCs

                    # Best model according to Validation Loss
                    if final_val_loss < global_best_loss:

                        # Delete previous best-by-loss model file
                        if best_model_filename is not None:
                            prev_path = os.path.join(best_plot_dir, best_model_filename)
                            prev_info_path = os.path.join(best_plot_dir, "best_val_loss_model_info.json")

                            # Protects from File Not Found, Permission Errors, Logging Weirdness
                            for f in [prev_path, prev_info_path]:
                                try:
                                    os.remove(f)
                                except FileNotFoundError:
                                    pass
                                except Exception as e:
                                    print(f"⚠️ Warning deleting {f}: {e}")

                            # Delete old best val loss plots
                            delete_best_model_plots("val_loss", best_plot_dir)

                        # Update best tracking
                        global_best_loss = final_val_loss
                        best_model_filename = f"best_val_loss_model_{model_id}.pt"

                        # Save model + metadata
                        torch.save(model.state_dict(), os.path.join(best_plot_dir, best_model_filename))
                        print(f"🌟 Best model so far (by val_loss) saved as '{best_model_filename}' with val_loss: {global_best_loss:.4f}")

                        # Define the model metadata
                        model_info = {
                            "model_id": model_id,
                            "num_layers": num_layers,
                            "hidden_size": hidden_size,
                            "batch_size": batch_size,
                            "dropout_rate": dropout_rate,
                            "final_val_loss_avg_last_5_epochs": global_best_loss,
                            "early_stopping_epoch": metrics[10] if len(metrics) > 10 else None,
                            "timestamp": timestamp
                        }

                        # Save the metadata as a JSON file
                        json_best_loss_path = os.path.join(best_plot_dir, "best_val_loss_model_info.json")
                        with open(json_best_loss_path, "w") as f:
                            json.dump(model_info, f, indent=2)

                        # Plot metrics for the model with best val_loss
                        PLOT_METRICS = [
                            ("Loss", 0, 1, "Training Loss", "Validation Loss"),
                            ("Accuracy", 2, 3, "Training Accuracy", "Validation Accuracy"),
                            ("Precision", 4, None, "Validation Precision", None),
                            ("Recall", 5, None, "Validation Recall", None),
                            ("F1 Score", 6, None, "Validation F1 Score", None),
                            ("Specificity", 7, None, "Validation Specificity", None),
                            ("ROC AUC", 8, None, "Validation ROC AUC", None),
                            ("PR AUC", 9, None, "Validation PR AUC", None)
                        ]

                        for name, i1, i2, label1, label2 in PLOT_METRICS:
                            plot_and_save(
                                metric1=metrics[i1],
                                metric2=metrics[i2] if i2 is not None else None,
                                name=name,
                                filename=os.path.join(best_plot_dir,
                                                      f"{model_id}_{name.lower().replace(' ', '_')}_best_val_loss_{timestamp}.png"),
                                model_id=model_id,
                                timestamp=timestamp,
                                label1=label1,
                                label2=label2
                            )

                    # Best model according to ROC AUC
                    if final_roc_auc > global_best_roc_auc:

                        # Delete previous best-by-AUC model file
                        if best_model_auc_filename is not None:
                            prev_auc_path = os.path.join(best_plot_dir, best_model_auc_filename)
                            prev_auc_info_path = os.path.join(best_plot_dir, "best_roc_auc_model_info.json")

                            # Protects from File Not Found, Permission Errors, Logging Weirdness
                            for f in [prev_auc_path, prev_auc_info_path]:
                                try:
                                    os.remove(f)
                                except FileNotFoundError:
                                    pass
                                except Exception as e:
                                    print(f"⚠️ Warning deleting {f}: {e}")

                            # Delete old best ROC AUC plots
                            delete_best_model_plots("roc_auc", best_plot_dir)

                        # Update best tracking
                        global_best_roc_auc = final_roc_auc
                        best_model_auc_filename = f"best_roc_auc_model_{model_id}.pt"

                        # Save model + metadata
                        torch.save(model.state_dict(), os.path.join(best_plot_dir, best_model_auc_filename))
                        print(f"🌟 Best model so far (by ROC AUC) saved as '{best_model_auc_filename}' with ROC AUC: {global_best_roc_auc:.4f}"
                        )

                        model_info = {
                            "model_id": model_id,
                            "num_layers": num_layers,
                            "hidden_size": hidden_size,
                            "batch_size": batch_size,
                            "dropout_rate": dropout_rate,
                            "final_val_loss_avg_last_5_epochs": global_best_loss,
                            "early_stopping_epoch": metrics[10] if len(metrics) > 10 else None,
                            "timestamp": timestamp
                        }

                        # Save the metadata as a JSON file
                        json_best_auc_path = os.path.join(best_plot_dir, "best_roc_auc_model_info.json")
                        with open(json_best_auc_path, "w") as f:
                            json.dump(model_info, f, indent=2)

                        # Plot metrics for the model with best roc_auc
                        PLOT_METRICS = [
                            ("Loss", 0, 1, "Training Loss", "Validation Loss"),
                            ("Accuracy", 2, 3, "Training Accuracy", "Validation Accuracy"),
                            ("Precision", 4, None, "Validation Precision", None),
                            ("Recall", 5, None, "Validation Recall", None),
                            ("F1 Score", 6, None, "Validation F1 Score", None),
                            ("Specificity", 7, None, "Validation Specificity", None),
                            ("ROC AUC", 8, None, "Validation ROC AUC", None),
                            ("PR AUC", 9, None, "Validation PR AUC", None)
                        ]

                        for name, i1, i2, label1, label2 in PLOT_METRICS:
                            plot_and_save(
                                metric1=metrics[i1],
                                metric2=metrics[i2] if i2 is not None else None,
                                name=name,
                                filename=os.path.join(best_plot_dir,
                                                      f"{model_id}_{name.lower().replace(' ', '_')}_best_roc_auc_{timestamp}.png"),
                                model_id=model_id,
                                timestamp=timestamp,
                                label1=label1,
                                label2=label2
                            )

                    # Store result with hyperparameters and metrics
                    results.append({
                        'model_id': model_id,
                        'timestamp': timestamp,
                        'dataset_name': 'EverHadHT_v1',
                        'num_layers': num_layers,
                        'hidden_size': hidden_size,
                        'batch_size': batch_size,
                        'dropout_rate': dropout_rate,
                        'metrics': metrics # this is the tuple containing all epoch-level lists
                    })

    ## JSON: Save cleaned results with lists
    json_ready_results = []
    for r in results:
        r_copy = r.copy()
        r_copy['metrics'] = convert_metrics(r['metrics'])
        json_ready_results.append(r_copy)

    json_path = os.path.join(table_dir, f"model_metrics_summary_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(json_ready_results, f, indent=2, default=str)

    ## Save summary CSV for final metrics only (accounting for the mean and std of each metric across the final five epochs of the model)
    # Fxn to calculate mean and std
    def mean_std(values):
        return np.mean(values[-5:]), np.std(values[-5:])
    summary_rows = []
    for r in results:
        metrics = r["metrics"]

        # Compute mean and std for each metric over the last 5 epochs
        train_loss_mean, train_loss_std = mean_std(metrics[0])
        val_loss_mean, val_loss_std = mean_std(metrics[1])
        train_acc_mean, train_acc_std = mean_std(metrics[2])
        val_acc_mean, val_acc_std = mean_std(metrics[3])
        precision_mean, precision_std = mean_std(metrics[4])
        recall_mean, recall_std = mean_std(metrics[5])
        f1_mean, f1_std = mean_std(metrics[6])
        specificity_mean, specificity_std = mean_std(metrics[7])
        roc_auc_mean, roc_auc_std = mean_std(metrics[8])
        pr_auc_mean, pr_auc_std = mean_std(metrics[9])

        summary_rows.append({
            "model_id": r["model_id"],
            "timestamp": r["timestamp"],
            "num_layers": r["num_layers"],
            "hidden_size": r["hidden_size"],
            "batch_size": r["batch_size"],
            "dropout_rate": r["dropout_rate"],
            "final_train_loss": train_loss_mean,
            "final_train_loss_std": train_loss_std,
            "final_val_loss": val_loss_mean,
            "final_val_loss_std": val_loss_std,
            "final_train_acc": train_acc_mean,
            "final_train_acc_std": train_acc_std,
            "final_val_acc": val_acc_mean,
            "final_val_acc_std": val_acc_std,
            "final_precision": precision_mean,
            "final_precision_std": precision_std,
            "final_recall": recall_mean,
            "final_recall_std": recall_std,
            "final_f1": f1_mean,
            "final_f1_std": f1_std,
            "final_specificity": specificity_mean,
            "final_specificity_std": specificity_std,
            "final_roc_auc": roc_auc_mean,
            "final_roc_auc_std": roc_auc_std,
            "final_pr_auc": pr_auc_mean,
            "final_pr_auc_std": pr_auc_std,
            "early_stopping_epoch": metrics[10]
        })

    df_summary = pd.DataFrame(summary_rows)
    csv_path = os.path.join(table_dir, "model_metrics_summary.csv")
    df_summary.to_csv(csv_path, index=False)

    return results, best_model_auc_filename, json_best_auc_path

### Run the training model
if __name__ == "__main__": # prevents autoexecution when fxns are loaded externally
    input_size = train_dataset[0][0].shape[0] # Obtains the number of input features
    results, best_model_auc_filename, json_best_auc_path = run_experiments(train_dataset, valid_dataset, input_size, timestamp, best_plot_dir, table_dir)

# Function to reconstruct the model with assigned architecture (i.e. best model architecture)
def reconstruct_model(input_size, metadata):
    return DynamicBinaryClassifier(
        input_size=input_size,
        hidden_size=int(metadata["hidden_size"]),
        num_layers=int(metadata["num_layers"]),
        dropout_rate=float(metadata["dropout_rate"])
    )

### Test the best performing trained model
# Function to test the best performing roc_auc model against the test data
def evaluate_best_model_on_test_data(test_dataset, input_size, json_best_auc_path, best_model_weights_path, plot_output_dir, table_output_dir):
    """
    Evaluates the best ROC AUC model on the test dataset and saves metrics, plots, and summaries.

    Parameters:
        test_dataset (Dataset): Test dataset
        input_size (int): Number of input features
        json_best_auc_path (str): Path to the JSON file containing best model metadata
        best_model_weights_path (str): Path to the best model's weights (.pt file)
        plot_output_dir (str): Directory to save plots
        table_output_dir (str): Directory to save evaluation summaries
    """
    import json

    # Create output directories (if they don't exist)...don't worry, it won't overwrite them
    os.makedirs(plot_output_dir, exist_ok=True)
    os.makedirs(table_output_dir, exist_ok=True)

    # Check to make sure file path exists
    if not os.path.exists(json_best_auc_path):
        raise FileNotFoundError(f"Metadata file not found: {json_best_auc_path}")

    # Load metadata (so that we can add hyperparameters from the best model)
    with open(json_best_auc_path, "r") as f:
        metadata = json.load(f)

    # Validate required keys
    required_keys = ["hidden_size", "num_layers", "batch_size", "dropout_rate"]
    missing = [k for k in required_keys if k not in metadata]
    if missing:
        raise KeyError(f"❌ Missing required metadata fields: {missing}")

    model_id = metadata["model_id"]
    timestamp = str(metadata["timestamp"])

    model = reconstruct_model(input_size, metadata) # reconstruct model with architecture in json

    model.load_state_dict(torch.load(best_model_weights_path)) # load weights from best model
    model.eval()

    # Create test loader
    test_loader = DataLoader(test_dataset, batch_size=int(metadata["batch_size"]), shuffle=False, num_workers=0)

    # Evaluate
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    loss_total, correct = 0.0, 0
    preds, labels, probs = [], [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.contiguous().to(device=device, dtype=torch.float32).view(-1)
            outputs = model(X_batch).view(-1)  # ensures outputs is a 1D vector

            loss = criterion(outputs, y_batch)
            loss_total += loss.item() * len(X_batch)

            p = torch.sigmoid(outputs)
            pred = (p > 0.5).float()

            correct += (pred == y_batch).float().sum().item()
            probs.extend(p.cpu().numpy())
            preds.extend(pred.cpu().numpy())
            labels.extend(y_batch.cpu().numpy())

    ## Compute metrics
    preds = np.array(preds)
    probs = np.array(probs)
    labels = np.array(labels)

    # Loss, accuracy, precision, recall, and f1-score
    loss_avg = loss_total / len(test_dataset)
    acc = np.mean(preds == labels)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)

    # Specificity calculation
    cm = confusion_matrix(labels, preds, labels=[0, 1])  # make sure both classes are accounted for
    if cm.shape == (2, 2):  # double-check in case of weird label distributions
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp + 1e-8)
    else:
        specificity = 0.0  # safe fallback

    # ROC AUC and PR AUC
    roc_auc = roc_auc_score(labels, probs)
    pr_auc = average_precision_score(labels, probs)

    # Save metrics to file
    results_txt = os.path.join(plot_output_dir, f"test_eval_{model_id}.txt")
    with open(results_txt, "w") as f:
        f.write(f"Model ID: {model_id}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Loss: {loss_avg:.4f}\nAccuracy: {acc:.4f}\nPrecision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\nF1 Score: {f1:.4f}\nSpecificity: {specificity:.4f}\n")
        f.write(f"ROC AUC: {roc_auc:.4f}\nPR AUC: {pr_auc:.4f}\n")

    # Save JSON
    summary_json = {
        "model_id": model_id,
        "timestamp": timestamp,
        "loss": loss_avg,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc
    }
    with open(os.path.join(table_output_dir, f"test_eval_{model_id}.json"), "w") as f:
        json.dump(summary_json, f, indent=2)

    # Save single-row CSV
    pd.DataFrame([summary_json]).to_csv(os.path.join(table_output_dir, f"test_eval_{model_id}.csv"), index=False)

    # Save ROC + PR curves
    fpr, tpr, _ = roc_curve(labels, probs)
    precision_curve, recall_curve, _ = precision_recall_curve(labels, probs)

    plt.figure()
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Random Chance")
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve ({model_id})")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_output_dir, f"roc_curve_{model_id}_{timestamp}.png"))
    plt.close()

    plt.figure()
    plt.plot(recall_curve, precision_curve, label=f"PR AUC = {pr_auc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR Curve ({model_id})")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_output_dir, f"pr_curve_{model_id}_{timestamp}.png"))
    plt.close()

    print(f"✅ Test evaluation complete. Results saved in: {plot_output_dir}")

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "model_id": model_id
    }

### Run the testing model, using the best trained roc_auc model as the architecture
# Subdirectories for testing outputs
tested_plot_dir = os.path.join(base_output_dir, "plots/EverHadHT/tested_models")
tested_table_dir = os.path.join(base_output_dir, "tables/EverHadHT/tested_models")

# Get best model weights file safely
best_model_weights_path = os.path.join(best_plot_dir, best_model_auc_filename)

if __name__ == "__main__": # prevents autoexecution when fxns are loaded externally
    # Run the model
    nn_results = evaluate_best_model_on_test_data(
        test_dataset=test_dataset,
        input_size=input_size,
        json_best_auc_path=json_best_auc_path,
        best_model_weights_path=best_model_weights_path,
        plot_output_dir=tested_plot_dir,
        table_output_dir=tested_table_dir
    )

### Run Logistic Regression of training dataset, to predict on test dataset
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, precision_recall_curve, average_precision_score
)
import statsmodels.api as sm

# Import X_train, y_train, X_test, and y_test (for statistical analysis / log regression)
dir = "/home/cweston1/miniconda3/envs/PythonProject/datasets/Datasets_for_statistical_comparison/"

X_train = pd.read_csv(os.path.join(dir, "X_training_EverHadHT.csv"), na_values=[""])
X_test = pd.read_csv(os.path.join(dir, "X_test_EverHadHT.csv"), na_values=[""])
y_train = pd.read_csv(os.path.join(dir, "y_training_EverHadHT.csv"), na_values=[""])
y_test = pd.read_csv(os.path.join(dir, "y_test_EverHadHT.csv"), na_values=[""])

# --------------------------------------------
# Define paths
# --------------------------------------------
base_dir = "/home/cweston1/miniconda3/envs/PythonProject/outputs/hypertension"
csv_dir = os.path.join(base_dir, "tables/EverHadHT/logistic_regression")
plot_dir = os.path.join(base_dir, "plots/EverHadHT/logistic_regression")

os.makedirs(csv_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

# --------------------------------------------
# Define your variables
# --------------------------------------------
binary_vars = ['Sex', 'Treatment', 'Smoked_one_pack_10_years_or_more']
multiclass_vars = ['Race', 'Age_Group', 'Education', 'Income', 'City']
numerical_vars = ['Alcohol_daily', 'BMI']
categorical_vars = binary_vars + multiclass_vars

# --------------------------------------------
# Preprocessing (fit on train, transform both sets)
# --------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_vars),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_vars)
    ],
    remainder='drop'
)

X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# Ensure targets are 1D arrays
y_train = y_train.values.ravel() #ravel converts to 1D array
y_test = y_test.values.ravel() #ravel converts to 1D array

# Get transformed feature names
ohe = preprocessor.named_transformers_['cat']
cat_feature_names = ohe.get_feature_names_out(categorical_vars)
feature_names = numerical_vars + list(cat_feature_names)

# --------------------------------------------
# Train scikit-learn LogisticRegression for prediction
# --------------------------------------------
log_reg = LogisticRegression(max_iter=1000, solver='liblinear')
log_reg.fit(X_train_transformed, y_train)
y_pred = log_reg.predict(X_test_transformed)
y_proba = log_reg.predict_proba(X_test_transformed)[:, 1]

# --------------------------------------------
# Evaluate performance
# --------------------------------------------
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
specificity = tn / (tn + fp)
precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba)
pr_auc = average_precision_score(y_test, y_proba)

# Save updated metrics with PR AUC
metrics_df = pd.DataFrame([{
    'Accuracy': accuracy,
    'ROC_AUC': roc_auc,
    'PR_AUC': pr_auc,
    'Precision': precision,
    'Recall': recall,
    'Specificity': specificity,
    'F1_Score': f1
}])

metrics_csv_path = os.path.join(csv_dir, "logistic_regression_metrics.csv")
metrics_df.to_csv(metrics_csv_path, index=False)
print(f"Saved metrics to {metrics_csv_path}")

# --------------------------------------------
# ROC Curve and save as PNG
# --------------------------------------------
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Logistic Regression')
plt.legend()
plt.grid(True)
plt.tight_layout()

roc_path = os.path.join(plot_dir, "roc_curve_logistic_regression.png")
plt.savefig(roc_path)
plt.close()
print(f"Saved ROC curve plot to {roc_path}")

# Save PR curve plot
plt.figure(figsize=(6, 5))
plt.plot(recall_curve, precision_curve, label=f'PR Curve (AUC = {pr_auc:.4f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve - Logistic Regression')
plt.legend()
plt.grid(True)
plt.tight_layout()

pr_path = os.path.join(plot_dir, "pr_curve_logistic_regression.png")
plt.savefig(pr_path)
plt.close()
print(f"Saved PR curve plot to {pr_path}")

log_reg_results = {
    "roc_auc": roc_auc,
    "pr_auc": pr_auc,
    "roc_plot_path": "plots/EverHadHT/logistic_regression/roc_curve_logistic_regression.png",
    "pr_plot_path": "plots/EverHadHT/logistic_regression/pr_curve_logistic_regression.png"
}

# --------------------------------------------
# Inference with Statsmodels
# --------------------------------------------
X_train_sm = sm.add_constant(X_train_transformed)
model_sm = sm.Logit(y_train, X_train_sm)
result = model_sm.fit(disp=False)

summary_frame = result.summary2().tables[1].rename(columns={
    'Coef.': 'Coefficient',
    'Std.Err.': 'Std_Error',
    'z': 'Z_score',
    'P>|z|': 'P_value',
    '[0.025': 'CI_lower',
    '0.975]': 'CI_upper'
})
summary_frame['Feature'] = ['Intercept'] + feature_names

summary_df = summary_frame[['Feature', 'Coefficient', 'Std_Error', 'Z_score', 'P_value', 'CI_lower', 'CI_upper']]
summary_df = summary_df.sort_values(by='P_value')

# Save model summary as CSV
coef_csv_path = os.path.join(csv_dir, "logistic_regression_coefficients.csv")
summary_df.to_csv(coef_csv_path, index=False)
print(f"Saved coefficient summary to {coef_csv_path}")

### Prepare HTML template (this is for rendering and saving the HTML)
def generate_html_report(report_path, model_id, timestamp, nn_results, log_reg_results):
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hypertension Risk Model Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; padding: 20px; max-width: 900px; margin: auto; }}
        h1 {{ color: #2c3e50; }}
        h2 {{ color: #34495e; }}
        img {{ max-width: 100%; height: auto; border: 1px solid #ccc; margin: 10px 0; }}
        hr {{ border: none; height: 1px; background-color: #ddd; margin: 30px 0; }}
    </style>
</head>
<body>

<h1>🧠 Hypertension Risk Classification — Deep Neural Network Summary</h1>

<h2>📌 Objective</h2>
<p>This project developed a deep feed-forward binary classifier in PyTorch to predict whether a patient developed hypertension within a 24-month observation window. The total dataset contained 60,000 simulated human patients that were matched with appropriate measures for individual and community confounding. The pre-training split used class-stratified random sampling with a 70/30 split between training and validation. The breakdown of sampling assignment was as follows: 37,500 patients in training, 11,250 patients in validation, and 11,250 in testing. All input vectors were temporally stationary (i.e. cross-sectional). </p>

<h2>🧱 Model Architecture</h2>
<p>The model architecture is defined by a dynamic PyTorch class <code>DynamicBinaryClassifier</code>, which flexibly builds a stack of hidden layers using:</p>
<ul>
    <li>He (Kaiming) initialization</li>
    <li>LeakyReLU activations</li>
    <li>Batch normalization</li>
    <li>Dropout regularization</li>
    <li>Output layer: raw logits</li>
    <li>Loss function: Binary crossentropy with logits loss</li>
    <li>Early Stopping: patience 15 | MinDelta (val loss) 0.001 | Reset Weights</li>
</ul>

<h2>🔧 Architecture Calibration</h2>
<p>We conducted a full grid search over combinations of:</p>
<ul>
    <li>Hidden layers: 2, 3, 4</li>
    <li>Hidden units per layer: 64, 128, 256, 512, 1024</li>
    <li>Dropout rates: 0.2, 0.3</li>
    <li>Batch sizes: 128, 256, 512, 1024, 2048</li>
</ul>

<p>Best performant model was determined according to:</p>
<ul>
    <li>🌟 Best ROC AUC Model (defined by the mean ROC AUC value of the five epochs prior to early stop)</li>
    
<h3>Optimal Architecture: <strong>4 Dense Layers | 64 Units | Batch Size 265 | Dropout 0.30</strong></h3>
    
</ul>

<hr>

<h2>📊 Training Curves</h2>
"""
    display_names = {
        "loss": "Loss",
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1 Score",
        "specificity": "Specificity",
        "roc_auc": "ROC AUC",
        "pr_auc": "PR AUC"
    }

    # Embed best model training plots
    metrics = ["loss", "accuracy", "precision", "roc_auc"]
    for metric in metrics:
        relative_best_plot_path = f"plots/EverHadHT/best_training_models/{model_id}_{metric}_best_roc_auc_{timestamp}.png"
        html += f"""
<h3>{display_names.get(metric, metric.title())}</h3>
<img src="{relative_best_plot_path}" alt="{display_names.get(metric, metric.title())} plot">
"""

    html += """
<hr>
<h2>🧪 Network Test Evaluation — Best ROC AUC Model</h2>
<p>Final performance of the optimized neural net model was evaluated on a held-out test dataset (n=11,250 observations). Below are the evaluation curves:</p>
"""

    roc_curve_path = f"plots/EverHadHT/tested_models/roc_curve_{model_id}_{timestamp}.png"
    pr_curve_path = f"plots/EverHadHT/tested_models/pr_curve_{model_id}_{timestamp}.png"

    # Embed test evaluation plots
    html += f"""
<h3>ROC AUC of Optimized Model</h3>
<img src="{roc_curve_path}" alt="ROC Curve">

<h3>PR AUC of Optimized Model</h3>
<img src="{pr_curve_path}" alt="PR Curve">
"""

    html += f"""
    <hr>
    <h2>🤖 Neural Net vs. Logistic Regression</h2>

    <p>The final ROC AUC score for the neural network model was <strong>{nn_results['roc_auc']:.4f}</strong>, compared to <strong>{log_reg_results['roc_auc']:.4f}</strong> for logistic regression. 
    Similarly, the PR AUC was <strong>{nn_results['pr_auc']:.4f}</strong> for the neural net and <strong>{log_reg_results['pr_auc']:.4f}</strong> for logistic regression.</p>

    <h3>ROC Curve Comparison</h3>
    <h4>Optimized Neural Net</h4>
    <img src="{roc_curve_path}" alt="NN ROC Curve">
    <h4>Logistic Regression</h4>
    <img src="{log_reg_results['roc_plot_path']}" alt="LogReg ROC Curve">

    <h3>Precision-Recall Curve Comparison</h3>
    <h4>Optimized Neural Net</h4>
    <img src="{pr_curve_path}" alt="NN PR Curve">
    <h4>Logistic Regression</h4>
    <img src="{log_reg_results['pr_plot_path']}" alt="LogReg PR Curve">
    """

    html += f"""
<p><strong>Report Timestamp:</strong> {timestamp}</p>

<footer>
  <hr>
  <p><em>Report generated automatically using PyTorch experiment logs.</em></p>
</footer>
</body>
</html>
"""

    # Save the report
    with open(report_path, "w") as f:
        f.write(html)

    print(f"✅ HTML report saved at: {report_path}")

### Run generate_html_report fxn
# Open and load the JSON file into a Python dictionary
import json
with open(json_best_auc_path, "r") as f:
    metadata = json.load(f)

# Access the 'model_id' from the metadata
model_id = metadata["model_id"]

report_path = os.path.join(base_output_dir, "DynamicBinaryClassifier_RNN_report.html")

# Run HTML function
generate_html_report(
    report_path,
    model_id,
    timestamp,
    nn_results=nn_results,
    log_reg_results=log_reg_results
)

### Create PDF of HTML with weasyprint
from weasyprint import HTML

html_path = os.path.join(base_output_dir, "DynamicBinaryClassifier_RNN_report.html")
pdf_path = os.path.join(base_output_dir, "DynamicBinaryClassifier_RNN_report.pdf")

HTML(html_path).write_pdf(pdf_path)
print(f"📄 PDF saved at: {pdf_path}")

################################
######## END OF SCRIPT #########
################################
