import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def clean_data(data):
    data = data.dropna()
    data = pd.get_dummies(data, drop_first=True)
    return data

def get_best_corr(data):
    corr = data.corr()['Survived'].abs()
    best_features = corr[corr > 0.2].index.tolist() #converti best_features en liste
    best_features.remove('Survived')
    return best_features

def annalyse_features(data):
    features = data.drop(columns=['Survived'])
    all_surv_rates = {}
    for col in features:
        surv_rate = data.groupby(col)['Survived'].mean()
        all_surv_rates[col] = surv_rate
    return all_surv_rates

