import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def clean_kidney_data(df):
    """
    Clean the kidney disease dataset.
    """
    saved_data_path = r'data/processed/kidney_cleanned.csv'
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
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

    # Fill missing values for 'Albumin_and_Globulin_Ratio'
    if 'Albumin_and_Globulin_Ratio' in df.columns:
        mean_value = round(df['Albumin_and_Globulin_Ratio'].mean(), 2)
        df['Albumin_and_Globulin_Ratio'] = df['Albumin_and_Globulin_Ratio'].fillna(mean_value)

    # Filter data based on conditions
    df_filtered = df[
        (df['Total_Bilirubin'] <= 25) &
        (df['Direct_Bilirubin'] <= 15) &
        (df['Alkaline_Phosphotase'] <= 1000) &
        (df['Alamine_Aminotransferase'] <= 500) &
        (df['Aspartate_Aminotransferase'] <= 1000)
]

    # Copy the filtered dataset
    df_processed = df_filtered.copy()

    # Encode 'Gender' using LabelEncoder
    if 'Gender' in df_processed.columns:
        gender_encoder = LabelEncoder()
        df_processed['Gender'] = gender_encoder.fit_transform(df_processed['Gender'])

    # Drop 'Direct_Bilirubin' column
    if 'Direct_Bilirubin' in df_processed.columns:
        df_processed.drop('Direct_Bilirubin', axis=1, inplace=True)

    # Transform 'Dataset' column: 1 -> 0, 2 -> 1
    if 'Dataset' in df_processed.columns:
        df_processed['Dataset'] = df_processed['Dataset'].apply(lambda x: 0 if x == 1 else 1)

    # Save the processed data
    df_processed.to_csv(saved_data_path, index=False)
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
