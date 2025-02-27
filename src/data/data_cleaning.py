import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def clean_kidney_data(df):
    """
    Clean the kidney disease dataset.
    """
    saved_data_path = r'data/processed/kidney_cleanned.csv'
    cols_names={"bp":"blood_pressure",
          "sg":"specific_gravity",
          "al":"albumin",
          "su":"sugar",
          "rbc":"red_blood_cells",
          "pc":"pus_cell",
          "pcc":"pus_cell_clumps",
          "ba":"bacteria",
          "bgr":"blood_glucose_random",
          "bu":"blood_urea",
          "sc":"serum_creatinine",
          "sod":"sodium",
          "pot":"potassium",
          "hemo":"haemoglobin",
          "pcv":"packed_cell_volume",
          "wc":"white_blood_cell_count",
          "rc":"red_blood_cell_count",
          "htn":"hypertension",
          "dm":"diabetes_mellitus",
          "cad":"coronary_artery_disease",
          "appet":"appetite",
          "pe":"pedal_edema",
          "ane":"anemia"}
    df.rename(columns=cols_names, inplace=True)
    df.drop(['id'], axis=1, inplace=True)
    df['diabetes_mellitus'] = df['diabetes_mellitus'].replace(to_replace = {'no':'no','yes':'yes',' yes':'yes'})
    df['coronary_artery_disease'] = df['coronary_artery_disease'].replace(to_replace = 'no', value='no')
    df['classification'] = df['classification'].replace(to_replace = 'ckd', value = 'ckd')
    df.to_csv(saved_data_path, index=False)
    
    # Save cleaned data
    df.to_csv(saved_data_path, index=False)
    print(f"Kidney disease data cleaned and saved to {saved_data_path}")

def clean_liver_data(df):
    # Get the directory of the current script
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the path to the 'data/processed' directory
    processed_dir = os.path.join(base_dir, '../../data/processed')

    # Ensure the directory exists
    os.makedirs(processed_dir, exist_ok=True)

    # Define the full path to save the processed data
    saved_data_path = os.path.join(processed_dir, 'processed_liver_data.csv')

    # Drop specified columns
    df.drop(columns=['Albumin_and_Globulin_Ratio', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin'], inplace=True)
    
    # Drop rows with missing values
    df.dropna(inplace=True)
    
    # Drop duplicate rows
    df.drop_duplicates(inplace=True)
    
    # Rename the 'Dataset' column to 'Status'
    df.rename(columns={'Dataset': 'Status'}, inplace=True)

    # Encode 'Gender' using LabelEncoder
    if 'Gender' in df.columns:
        gender_encoder = LabelEncoder()
        df['Gender'] = gender_encoder.fit_transform(df['Gender'])
    # Transform 'Dataset' column: 1 -> 0, 2 -> 1
    if 'Status' in df.columns:
        df['Status'] = df['Status'].apply(lambda x: 0 if x == 1 else 1)

    # Save the processed data
    df.to_csv(saved_data_path, index=False)
    print(f"Processed data saved to {saved_data_path}")

def clean_diabetes_data(df):
    """
    Clean the diabetes dataset.
    """
    saved_data_path = r'data/processed/diabetes_cleanned.csv'
    
    # Drop duplicates and missing values
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    
    # Save the cleaned data
    df.to_csv(saved_data_path, index=False)
    print(f"Diabetes data cleaned and saved to {saved_data_path}")

def clean_cancer_data(df):
    """
    Clean the cancer dataset.
    """
    saved_data_path = r'data/processed/cancer_cleanned.csv'
    
    # List of selected features
    selected_features = [
        'perimeter_mean', 'area_mean', 'concavity_mean', 'concave points_mean',
        'radius_mean', 'diagnosis']
    
    df.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    df.drop_duplicates(inplace=True)
    cleaned_data = df[selected_features]
    cleaned_data.to_csv(saved_data_path, index=False)
    print(f"Cancer data cleaned and saved to {saved_data_path}")


def clean_heart_data(df):
      # Define column renaming
    rename_columns = {
        "age": "Age",
        "sex": "Gender",
        "cp": "Chest_Pain_Type",
        "trestbps": "Resting_Blood_Pressure",
        "chol": "Serum_Cholesterol",
        "fbs": "Fasting_Blood_Sugar",
        "restecg": "Resting_ECG_Results",
        "thalach": "Max_Heart_Rate_Achieved",
        "exang": "Exercise_Induced_Angina",
        "oldpeak": "ST_Depression_Exercise_vs_Rest",
        "slope": "Slope_of_ST_Segment",
        "ca": "Major_Vessels_Fluoroscopy",
        "thal": "Thalassemia_Type",
        "target": "Heart_Disease"
    }

    # Rename the columns in the dataset
    df.rename(columns=rename_columns, inplace=True)
    # Get the directory of the current script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the path to the 'data/processed' directory
    processed_dir = os.path.join(base_dir, '../../data/processed')
    
    # Ensure the directory exists
    os.makedirs(processed_dir, exist_ok=True)

    
    # Define the full path to save the processed data
    saved_data_path = os.path.join(processed_dir, 'processed_heart_disease.csv')
    
    
    # Save the processed data
    df.to_csv(saved_data_path, index=False)


if __name__ == "__main__":
    # Path to the raw datasets
    kidney_data_path = r"data/raw/kidney_disease.csv"
    liver_data_path = r"data/raw/indian_liver_patient.csv"
    diabetes_data_path = r"data/raw/diabetes.csv"
    cancer_data_path = r"data/raw/cancer.csv"
    heart_data_path = r"data/raw/heart.csv"


    kidney_data_path = r"https://raw.githubusercontent.com/Adrshp806/data_file_for-projects/refs/heads/main/kidney_disease.csv"
    liver_data_path = r"https://raw.githubusercontent.com/Adrshp806/data_file_for-projects/refs/heads/main/indian_liver_patient.csv"
    diabetes_data_path = r"https://raw.githubusercontent.com/Adrshp806/data_file_for-projects/refs/heads/main/diabetes.csv"
    cancer_data_path = r"https://raw.githubusercontent.com/Adrshp806/data_file_for-projects/refs/heads/main/cancer.csv"
    heart_data_path = r"https://raw.githubusercontent.com/Adrshp806/data_file_for-projects/refs/heads/main/heart.csv"
    
    # Load and clean the kidney dataset
    kidney_df = pd.read_csv(kidney_data_path)
    clean_kidney_data(kidney_df)
    
    # Load and clean the liver dataset
    liver_df = pd.read_csv(liver_data_path)
    clean_liver_data(liver_df)
    
    # Load and clean the diabetes dataset
    diabetes_df = pd.read_csv(diabetes_data_path)
    clean_diabetes_data(diabetes_df)
    
    # Load and clean the cancer dataset
    cancer_df = pd.read_csv(cancer_data_path)
    clean_cancer_data(cancer_df)
    

    #load and clean the heart dataset
    heart_df = pd.read_csv(heart_data_path)
    clean_heart_data(heart_df)
