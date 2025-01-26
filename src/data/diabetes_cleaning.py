import pandas as pd




def clean_data(df):
    saved_data_path = r'data/processed/diabetes_cleanned.csv'
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    df.to_csv(saved_data_path, index=False)

if __name__ == "__main__":
    DATA_PATH = r"data/raw/diabetes.csv"
    df = pd.read_csv(DATA_PATH)
    clean_data(df)
