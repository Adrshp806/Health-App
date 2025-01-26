import pandas as pd

def clean_data(df):
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

if __name__ == "__main__":
    DATA_PATH = r"data/raw/cancer.csv"
    df = pd.read_csv(DATA_PATH)
    clean_data(df)
