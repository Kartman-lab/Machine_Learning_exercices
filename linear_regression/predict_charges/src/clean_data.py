import pandas as pd
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('data/insurance.csv')
data = data.dropna()
le = LabelEncoder()

def clean_data(data):
    data = data.dropna()
    data = pd.get_dummies(data, drop_first=True)
    return data







