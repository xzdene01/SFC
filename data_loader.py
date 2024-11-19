import pandas as pd

def load_data(filepath: str = 'data/wine.data') -> pd.DataFrame:
    df = pd.read_csv(filepath, header=None)

    # just for wine dataset - other are not yet supported
    df.columns = ['class', 'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols',
                  'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue',
                  'OD280/OD315_of_diluted_wines', 'proline']
    return df