import os
import pandas as pd

def clean_data(df):
    """
    Cleans the input DataFrame by:
    - Removing columns that are not important.
    - Converting data types where necessary.
    """
    # Drop columns that are not important
    columns_to_drop = ['id']  # Replace 'id' with actual columns you deem unnecessary
    df = df.drop(columns=columns_to_drop, errors='ignore')
    
    # Convert categorical columns to lowercase strings for consistency
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.lower().str.strip()
    
    return df

def save_data(df):
    """
    Saves the cleaned DataFrame to the processed data directory.
    """
    # Get the directory of the current script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the path to the 'data/processed' directory
    processed_dir = os.path.join(base_dir, '../../data/processed')
    
    # Ensure the directory exists
    os.makedirs(processed_dir, exist_ok=True)
    
    # Define the full path to save the data
    saved_data_path = os.path.join(processed_dir, 'processed_kidney_disease.csv')

    # Save the data
    df.to_csv(saved_data_path, index=False)

if __name__ == "__main__":
    # Path to the raw data file
    DATA_PATH = r"data/raw/kidney_disease.csv"
    
    try:
        # Read the raw dataset
        df = pd.read_csv(DATA_PATH)
        
        # Clean the data
        cleaned_df = clean_data(df)
        
        # Save the cleaned data
        save_data(cleaned_df)
        print("Data cleaning and saving completed successfully.")
    
    except FileNotFoundError:
        print(f"File not found: {DATA_PATH}")
    except pd.errors.EmptyDataError:
        print(f"No data in file: {DATA_PATH}")
    except Exception as e:
        print(f"An error occurred: {e}")
