import pandas as pd

from src.clean_data import clean_data, data
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Lasso

data = clean_data(data)

def build_feature_matrix(X):
    '''Créer les intéractions entre variables'''
    poly = PolynomialFeatures(interaction_only=True, include_bias=False)
    X_interactions = poly.fit_transform(X)
    features_names = poly.get_feature_names_out(X.columns)
    return X_interactions, features_names, poly

def scale_features(X_interactions):
    '''Standardise les variables pour stabiliser Lasso'''
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_interactions)
    return X_scaled, scaler

def select_features_lasso(X_scaled, y, features_names, alpha=0.01):
    '''Utilise Lasso pour sélectionner les meilleurs features'''
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_scaled, y)

    coeff_df = pd.DataFrame({
        "features": features_names,
        "coef": lasso.coef_
    }).sort_values(by='coef', key=abs, ascending=False)

    return coeff_df, lasso

def preprocess():
    X = data.drop(columns=['charges'])  
    y = data['charges']

    X_interactions, features_names, poly = build_feature_matrix(X) 
    X_scaled, scaler = scale_features(X_interactions)
    coeff_df, lasso = select_features_lasso(X_scaled, y, features_names)

    print("Meilleures features selon Lasso :")
    print(coeff_df.head(10))

    selected_mask = lasso.coef_ != 0
    X_selected = X_scaled[:, selected_mask]
    selected_features = features_names[selected_mask]
    
    return {
        "X_scaled": X_scaled,
        "X_selected": X_selected,
        "y": y,
        "poly": poly,
        "scaler": scaler,
        "lasso": lasso,
        "feature_names": features_names,
        "selected_features": selected_features,
        "coeff_df": coeff_df,
    }