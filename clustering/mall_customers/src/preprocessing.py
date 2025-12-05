import pandas as pd
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('data/Mall_Customers.csv.xls')
print(data.head)

def scale_features(data):
    X = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled

