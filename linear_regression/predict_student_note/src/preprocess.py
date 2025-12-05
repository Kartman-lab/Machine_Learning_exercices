import pandas as pd

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Lasso

def clean_data(data):
    data = data.dropna()
    data = pd.get_dummies(data, drop_first=True)  # transforme toutes les colonnes catégorielles
    return data

def build_features_matix(X):
    poly = PolynomialFeatures(interaction_only=True, include_bias=False)
    X_interactions = poly.fit_transform(X)
    features_names = poly.get_feature_names_out(X.columns)

    return X_interactions, features_names, poly

def scale_features(X_ineractions):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_ineractions)

    return scaler, X_scaled

def select_features_lasso(X_scaled, y, features_names, alpha=0.001):
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_scaled, y)

    coeff_dt = pd.DataFrame({
        "features": features_names,
        "coeff": lasso.coef_
    }).sort_values(by='coeff', key=abs, ascending=False)

    return coeff_dt, lasso

def preprocess(data):
    X = data.drop(columns=['G3'])
    y = data['G3']

    X_interactions, features_names, poly = build_features_matix(X)
    scaler, X_scaled = scale_features(X_interactions)
    coeff_df, lasso = select_features_lasso(X_scaled, y, features_names)
    selected_mask = lasso.coef_ != 0
    X_selected = X_scaled[:, selected_mask]
    selected_features = features_names[selected_mask]

    return {
        'X_scaled': X_scaled,
        'X_selected': X_selected,
        'y': y,
        'poly': poly,
        'scaler': scaler, 
        'lasso': lasso,
        'features_names': features_names,
        'selected_features': selected_features,
        'coeff_df': coeff_df
     }


    



