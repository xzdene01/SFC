import pandas as pd
import numpy as np

def load_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)

def load_data_split(filepath: str, seed: int = None) -> pd.DataFrame:
    df = load_data(filepath)
    split = int(0.8 * len(df))
    train = df.iloc[:split]
    test = df.iloc[split:].reset_index(drop=True)

    return train, test