import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def clean_data(df):
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

if __name__ == "__main__":
    # Path to the input dataset
    DATA_PATH = r"data/raw/indian_liver_patient.csv"

    # Load the dataset
    df = pd.read_csv(DATA_PATH)

    # Process and clean the data
    clean_data(df)
