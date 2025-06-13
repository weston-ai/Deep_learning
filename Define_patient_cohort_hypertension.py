# SCRIPT OBJECTIVE: This script simulates a realistic longitudinal cohort of 60,000 synthetic patients observed monthly over a 24-month period for hypertension development. Each patient is assigned demographic features (race, sex, age, education, income), a treatment group (Drug A or Drug B), a dropout month, and a city with hypertension risk weighting based on race. Lifestyle behaviors such as BMI, smoking status, and alcohol use are stochastically generated and influence hypertension risk. Monthly hypertension outcomes are drawn from a Bernoulli distribution using a risk function that integrates treatment, demographics, and behavior modifiers. The dataset includes event flag variables to denote the first through eighth hypertension occurrences, and calculates time-to-event values for each patient. Patients’ BMI is categorized as Low, Overweight, or Obese using WHO standards. After a patient’s dropout month, all non-essential variables are set to missing (NaN) to simulate incomplete follow-up. The final dataset is exported as a CSV file and is structured for downstream analysis including machine learning, causal inference, or survival modeling.

import numpy as np
import pandas as pd
import random
from scipy.stats import bernoulli
import pickle

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Constants
N_PATIENTS = 60000
N_MONTHS = 24
DROP_OUT_RATE = 0.14

# Updated demographic distributions:
# White: 59%, Black: 13%, Hispanic: 18%, East Asian: 5%, Other: 5%
race_distribution = {
    "White": 0.59, "Black": 0.13, "Hispanic": 0.18, "East Asian": 0.05, "Other": 0.05
}
sex_distribution = {"Male": 0.49, "Female": 0.51}
age_distribution = {"18-44": 0.3, "45-64": 0.5, "65-85": 0.2}
education_levels = ["No High School", "High School", "2-Year College", "4-Year University", "Graduate School"]
income_levels = ["<25K", "25K-50K", "50K-75K", "75K-100K", "100K+"]

# Major US cities and associated risks
cities = ["New York City", "Los Angeles", "Phoenix", "San Diego",
          "Dallas", "San Francisco", "Seattle", "Denver", "Tampa Bay"]
medium_risk_cities = ["Houston", "St Louis", "Detroit", "Cleveland", "Philadelphia", "Chicago", "San Antonio"]
high_risk_cities = ["Memphis", "New Orleans", "Atlanta", "Birmingham", "Jackson"]

# Combine all cities into a dictionary for easier access
city_risk_categories = {
    "low": cities,
    "medium": medium_risk_cities,
    "high": high_risk_cities
}

# Function to assign city with weighted randomness based on risk level
def assign_city(race):
    # Default weights for each risk tier
    weights = {"low": 0.6, "medium": 0.3, "high": 0.1}

    # Set the city assignment weights for Black race
    if race == "Black":
        weights = {"low": 0.1, "medium": 0.3, "high": 0.6}
    # Set the city assignment weights for Hispanic race
    elif race == "Hispanic":
        weights = {"low": 0.3, "medium": 0.4, "high": 0.3}
    # Set the city assignment weights for White race
    elif race == "White":
        weights = {"low": 0.5, "medium": 0.3, "high": 0.2}

    # Normalize weights and choose a risk tier
    risk_tiers = list(weights.keys())
    tier_probs = list(weights.values())
    chosen_tier = np.random.choice(risk_tiers, p=tier_probs)

    # Select a random city from the chosen risk tier
    return np.random.choice(city_risk_categories[chosen_tier]), chosen_tier

# Generate patients
def generate_patients(n):
    patients = []
    # Assign treatment
    treatments = ["Drug A"] * (n // 2) + ["Drug B"] * (n - n // 2) # Makes a list with 1/2*n Drug A and 1/2*n Drug B
    random.shuffle(treatments)

    for i in range(n):
        treatment = treatments[i]
        # Select race using the updated distribution.
        race = np.random.choice(list(race_distribution.keys()), p=list(race_distribution.values()))
        # Select sex.
        sex = np.random.choice(list(sex_distribution.keys()), p=list(sex_distribution.values()))
        # Select age group.
        age_group = np.random.choice(list(age_distribution.keys()), p=list(age_distribution.values()))
        # Assign education and income based on age group.
        if age_group == "18-44":
            education_probs = [0.1, 0.4, 0.3, 0.15, 0.05]
            income_probs = [0.3, 0.3, 0.2, 0.15, 0.05]
        elif age_group == "45-64":
            education_probs = [0.05, 0.3, 0.3, 0.25, 0.1]
            income_probs = [0.15, 0.3, 0.3, 0.15, 0.1]
        else:
            education_probs = [0.05, 0.25, 0.3, 0.25, 0.15]
            income_probs = [0.1, 0.2, 0.3, 0.2, 0.2]
        education = np.random.choice(education_levels, p=education_probs)
        income = np.random.choice(income_levels, p=income_probs)

        # Determine dropout: if dropout occurs, select a random month between 1 and 23; else complete 24 months.
        dropout = bernoulli.rvs(DROP_OUT_RATE)
        dropout_month = np.random.randint(1, N_MONTHS) if dropout else N_MONTHS
        # City assignment
        city, city_risk = assign_city(race)
        # Generate BMI from a normal distribution, with upward adjustment if over threshold.
        bmi = np.random.normal(27, 4)
        if (sex == "Male" and bmi > 28) or (sex == "Female" and bmi > 34):
            bmi += np.random.uniform(2, 5)
        # 30% of patients are long-term smokers.
        smoked = bernoulli.rvs(0.3)

        # Generate Alcohol_daily:
        # Draw a raw value from a gamma distribution (shape=0.3, scale=1/0.3 gives mean ~1).
        raw_alcohol = np.random.gamma(shape=0.3, scale=(1 / 0.3))
        # Round to the nearest 0.5 and clip between 0 and 8.
        alcohol = np.round(raw_alcohol * 2) / 2.0
        alcohol = np.clip(alcohol, 0, 8)
        # Increase alcohol consumption by 15% for males.
        if sex == "Male":
            alcohol = np.round(alcohol * 1.15 * 2) / 2.0
            alcohol = np.clip(alcohol, 0, 8)

        patients.append(
            [i, race, sex, age_group, education, income, treatment, dropout_month, city, city_risk, bmi, smoked, alcohol])
    return pd.DataFrame(patients, columns=["Patient_ID", "Race", "Sex", "Age_Group", "Education", "Income", "Treatment", "Dropout_Month", "City", "City_Risk_Tier", "BMI", "Smoked_one_pack_10_years_or_more", "Alcohol_daily"])

# Generate observations for each patient over 24 months.
def generate_observations(patients):
    observations = []
    for _, row in patients.iterrows():
        for month in range(1, N_MONTHS + 1):
            if month <= row["Dropout_Month"]:
                # Set baseline HT risk based on Drug Treatment
                hypertension_risk = 0.02 if row["Treatment"] == "Drug A" else 0.04
                # For patients on Drug B who are Hispanic (any sex) or Black males,
                # significantly increase the risk by 0.02 if Hispanic Male or Black Either Sex.
                if row["Treatment"] == "Drug B" and (
                        (row["Race"] == "Hispanic" and row["Sex"] == "Male") or (row["Race"] == "Black")):
                    hypertension_risk += 0.02
                # Increase risk due to BMI: add 0.01 if male BMI > 28 or female BMI > 34.
                if (row["Sex"] == "Male" and row["BMI"] > 28) or (row["Sex"] == "Female" and row["BMI"] > 34):
                    hypertension_risk += 0.01
                # Increase risk for smokers: add 0.02.
                if row["Smoked_one_pack_10_years_or_more"] == 1:
                    hypertension_risk += 0.02
                # Increase risk if Alcohol_daily > 3.0: add 0.01.
                if row["Alcohol_daily"] > 3.0:
                    hypertension_risk += 0.01

                hypertension = bernoulli.rvs(hypertension_risk)
            else:
                # For months after dropout, set observation to np.nan
                hypertension = np.nan
            observations.append([row["Patient_ID"], month, hypertension])
    return pd.DataFrame(observations, columns=["Patient_ID", "Month", "Hypertension"])


# Generate patient and observation data.
patients_df = generate_patients(N_PATIENTS)
observations_df = generate_observations(patients_df)
# Merge patient-level data into the monthly observations.
dataset = observations_df.merge(patients_df, on="Patient_ID")
# Instead of computing Hypertension_Proportion, we compute Hypertension_Count:
# For each patient, count the number of months with a 1 for Hypertension.
dataset["Hypertension_Count"] = dataset.groupby("Patient_ID")["Hypertension"].transform(lambda x: (x == 1).sum())


###########################################
# Add Event Flag Variables
###########################################
def add_event_flags(group):
    # Ensure the group is sorted by Month.
    group = group.sort_values("Month").copy()
    event_names = ["first_event", "second_event", "third_event", "fourth_event",
                   "fifth_event", "sixth_event", "seventh_event", "eighth_event"]
    # Identify months when a hypertension event occurred.
    event_months = group.loc[group["Hypertension"] == 1, "Month"].tolist()
    # For each event flag, mark rows: if Month < event_time: 0; if Month == event_time: 1; if Month > event_time: np.nan.
    for idx, event_name in enumerate(event_names):
        if idx < len(event_months):
            event_time = event_months[idx]
            group[event_name] = group["Month"].apply(
                lambda m: 0 if m < event_time else (1 if m == event_time else np.nan))
        else:
            group[event_name] = 0
    return group

dataset = dataset.groupby("Patient_ID").apply(add_event_flags).reset_index(drop=True)

### Function to calculate time to HT events
def calculate_time_to_events(group):
    events = group[group['Hypertension'] == 1]['Month'].tolist() # make list of months where the patient encountered hypertension
    first_event = events[0] if len(events) > 0 else np.nan # assign 1st month
    second_event = events[1] if len(events) > 1 else np.nan # assign 2nd month
    third_event = events[2] if len(events) > 2 else np.nan # assign 3rd month
    fourth_event = events[3] if len(events) > 3 else np.nan # assign 4th month
    fifth_event = events[4] if len(events) > 4 else np.nan # assign 5th month
    sixth_event = events[5] if len(events) > 5 else np.nan  # assign 6th month
    seventh_event = events[6] if len(events) > 6 else np.nan  # assign 7th month
    eighth_event = events[7] if len(events) > 7 else np.nan  # assign 8th month

    group['Time_to_first_event'] = first_event
    group['Time_to_second_event'] = second_event
    group['Time_to_third_event'] = third_event
    group['Time_to_fourth_event'] = fourth_event
    group['Time_to_fifth_event'] = fifth_event
    group['Time_to_sixth_event'] = sixth_event
    group['Time_to_seventh_event'] = seventh_event
    group['Time_to_eighth_event'] = eighth_event

    return group

# Run the "calculate_time_to_events" fxn on the dataset
dataset = dataset.groupby('Patient_ID').apply(calculate_time_to_events).reset_index(drop=True) # apply the function to grouped patient

# Define function to categorize BMI (using World Health Organization standards)
def categorize_BMI(BMI):
    if BMI < 25:
        return 'Low BMI'
    elif 25 <= BMI < 30:
        return 'Overweight'
    else:
        return 'Obese'

# Apply the categorize_BMI() function to the BMI column
dataset['BMI_category'] = dataset['BMI'].apply(categorize_BMI).reset_index(drop=True)

## Move the Hypertension column to the last column
Hypertension = 'Hypertension'
dataset[Hypertension] = dataset.pop(Hypertension)

###########################################
# Mark All Variables (except Patient_ID, Month, and Dropout_Month) as np.nan for months after dropout.
###########################################
def mark_dropout_rows(df):
    # Columns to preserve: Patient_ID, Month, Dropout_Month.
    cols_to_preserve = ["Patient_ID", "Month", "Dropout_Month"]
    cols_to_replace = [col for col in df.columns if col not in cols_to_preserve]
    # For each row where Month > Dropout_Month, set all other columns to np.nan
    mask = df["Month"] > df["Dropout_Month"]
    df.loc[mask, cols_to_replace] = np.nan
    return df

dataset = mark_dropout_rows(dataset)

# (Optional) Display a sample patient's data.
sample_patient = dataset[dataset["Patient_ID"] == 0]
print("\nSample patient (Patient_ID=0) with event flags and dropout rows marked:")
print(sample_patient.head(24))

dataset = pd.DataFrame(dataset)

dataset['Smoked_one_pack_10_years_or_more'] = dataset['Smoked_one_pack_10_years_or_more'].astype(str)

dataset.to_csv("/home/cweston1/miniconda3/envs/PythonProject/datasets/my_custom_datasets/hypertension_study_MAIN_20250408.csv", index=False)
print("Dataset generated and saved as hypertension_study_MAIN_20250408.csv")

#########################################
#### SUMMARY OF SCRIPT FUNCTIONALITY ####
#########################################

# 🔹 1. Setup and Configuration
# Imports: Common libraries like NumPy, Pandas, SciPy, and pickle (although pickle isn’t used here).
#
# Random seeds are set for reproducibility.
#
# Constants:
#
# N_PATIENTS = 60,000
#
# N_MONTHS = 24
#
# DROP_OUT_RATE = 14%
#
# Demographics: Custom distributions for race, sex, age, education, and income.
#
# City risk tiers: Cities are classified into low, medium, and high hypertension risk zones, which affect city assignment logic by race.
#
# 🔹 2. Patient Generation: generate_patients(n)
# For each patient:
#
# Assigns:
#
# Treatment group (Drug A or Drug B, 50/50 split)
#
# Demographics (race, sex, age, education, income)
#
# Dropout month (early dropout or complete)
#
# City and city risk tier (weighted by race)
#
# BMI (with boosted values if in a risk category)
#
# Smoking status (30% long-term smokers)
#
# Daily alcohol use (gamma-distributed, scaled by sex)
#
# Returns a DataFrame with one row per patient.
#
# 🔹 3. Observation Generation: generate_observations(patients)
# For each patient-month up to the dropout point:
#
# Calculates a probability of hypertension, based on:
#
# Treatment (Drug B has higher risk)
#
# Race and sex (Black or Hispanic males on Drug B get +0.02 risk)
#
# BMI (male > 28 or female > 34 adds +0.01)
#
# Smoking (+0.02)
#
# Alcohol (>3.0 units adds +0.01)
#
# Generates a hypertension event with bernoulli.rvs(prob)
#
# After dropout, sets hypertension to NaN
#
# 🔹 4. Dataset Assembly
# Merges the patient info into monthly observation rows.
#
# Computes Hypertension_Count per patient across all observed months (sum of 1s).
#
# 🔹 5. Adds Event Flags
# For each patient:
#
# Flags the first through eighth hypertension events with columns like first_event, second_event, etc.
#
# Marks only the month of the event with 1, earlier months as 0, later as NaN.
#
# 🔹 6. Adds Time-to-Event Columns
# Computes months at which each hypertension event occurred:
# Time_to_first_event, ..., Time_to_eighth_event.
#
# 🔹 7. Categorizes BMI
# Labels each patient as:
#
# 'Low BMI' (<25),
#
# 'Overweight' (25–29.9),
#
# 'Obese' (30+)
#
# 🔹 8. Dropout Row Cleanup
# For each patient:
#
# After their dropout month, sets all columns except Patient_ID, Month, and Dropout_Month to NaN — mimicking missing data post-dropout.
#
# 🔹 9. Final Steps
# Converts smoking status to str for compatibility.
#
# Exports the full dataset as a CSV file:
# hypertension_study_MAIN_20250408.csv
#
# Displays a preview of one sample patient's monthly records (Patient_ID=0).
#
# 📊 Result:
# You now have a rich, realistic, longitudinal dataset for 60,000 synthetic patients with:
#
# Time-dependent hypertension outcomes
#
# Demographic and behavioral covariates
#
# Dropout simulation
#
# Event timing and flags for modeling survival or recurrence
#
# Clean structure for ML, regression, or deep learning