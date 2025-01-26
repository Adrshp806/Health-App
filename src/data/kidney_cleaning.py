import os
import pandas as pd

def save_data(df):
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
    DATA_PATH = r"data/raw/kidney_disease.csv"
    df = pd.read_csv(DATA_PATH)
    save_data(df)
  