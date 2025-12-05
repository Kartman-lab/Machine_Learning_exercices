import pandas as pd 
from src.preprocess import preprocess, clean_data
from src.model import train_model

math_data = pd.read_csv('data/student/student-mat.csv', sep=';')
math_data = clean_data(math_data)

math_df_coeff = preprocess(math_data)
X_math = math_df_coeff['X_selected']
y_math = math_df_coeff['y']

rmse_math, mape_math, y_pred_math, y_test_math = train_model(X_math, y_math)

print('Math prédictions')
print('rmse:', rmse_math)
print('mape:', mape_math)

port_data = math_data = pd.read_csv('data/student/student-por.csv', sep=';')
port_data = clean_data(port_data)
port_df_coeff = preprocess(port_data)

X_port = port_df_coeff['X_selected']
y_port = port_df_coeff['y']

rmse_port, mape_port, y_pred_port, y_test_port = train_model(X_port, y_port)

print('Portuguese predictions')
print('rmse:', rmse_port)
print('mape:', mape_port)



