import pandas as pd

def load_data():
    # load wine dataset
    df = pd.read_csv('data/wine.data', header=None)
    df.columns = ['class', 'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols',
                  'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue',
                  'OD280/OD315_of_diluted_wines', 'proline']
    return df