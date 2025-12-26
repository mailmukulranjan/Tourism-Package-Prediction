# for data manipulation
import pandas as pd
import sklearn
import numpy as np
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder, StandardScaler
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/mailmukulranjan/tourism-package-prediction/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")
# Data Cleaning
print("Initial shape:", df.shape)
# Drop unique identifier column (not useful for modeling)
df.drop(columns=['CustomerID'], inplace=True)
# Handle missing values
# For numerical columns, fill with median
numerical_cols = df.select_dtypes(include=[np.number]).columns
for col in numerical_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

# For categorical columns, fill with mode
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)

# Handle typo in Gender column ('Fe Male' should be 'Female')
if 'Gender' in df.columns:
    df['Gender'] = df['Gender'].replace({'Fe Male': 'Female'})

    # Remove duplicates
df.drop_duplicates(inplace=True)

print("Shape after cleaning:", df.shape)

# Encode categorical variables
label_encoders = {}
categorical_features = ['TypeofContact', 'Occupation', 'Gender', 'ProductPitched',
       'MaritalStatus', 'Designation']

for col in categorical_features:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

# Separate features and target
X = df.drop(['ProdTaken', 'CustomerID'], axis=1, errors='ignore')
y = df['ProdTaken']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale numerical features
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame
# X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
# X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Save processed data
X_train.to_csv('tourism_project/model_building/X_train.csv', index=False)
X_test.to_csv('tourism_project/model_building/X_test.csv', index=False)
y_train.to_csv('tourism_project/model_building/y_train.csv', index=False)
y_test.to_csv('tourism_project/model_building/y_test.csv', index=False)

files = ["X_train.csv","X_test.csv","y_train.csv","y_test.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="mailmukulranjan/tourism-package-prediction",
        repo_type="dataset",
    )
