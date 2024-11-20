import pandas as pd
import numpy as np

def load_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)

def load_data_split(filepath: str, seed: int = None) -> pd.DataFrame:
    if seed is not None:
        np.random.seed(seed)

    df = load_data(filepath)

    # partition dataset into train and test (80% - 20%)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    split = int(0.8 * len(df))
    train = df.iloc[:split]
    test = df.iloc[split:].reset_index(drop=True)

    return train, test