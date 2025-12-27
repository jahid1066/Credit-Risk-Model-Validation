import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(path):
    df = pd.read_csv(path)
    return df

def clean_data(df):
    df['person_emp_length'] = df['person_emp_length'].clip(upper=50)
    df = df.dropna()
    return df

def encode_features(df):
    categorical_cols = df.select_dtypes(include='object').columns
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df_encoded

def split_features_target(df):
    X = df.drop('loan_status', axis=1)
    y = df['loan_status']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y
