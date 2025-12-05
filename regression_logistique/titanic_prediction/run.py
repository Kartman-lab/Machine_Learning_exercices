import pandas as pd

from src.preprocess import clean_data, get_best_corr, annalyse_features
from src.model import train_model
from src.visualise import visualise_confusion_matrix, visualize_roc_curve

titanic_data = pd.read_csv('data/train.csv')
data = clean_data(titanic_data)

best_features = get_best_corr(data)
surv_rate = annalyse_features(data)

X = data[best_features]
y = data['Survived']

acc_score, conf_matrix, class_report, y_pred, y_test, X_test, log = train_model(X, y)

print("Accruacy:", acc_score)
print("Confusion Matrix:", conf_matrix)
print(class_report)

visualise_confusion_matrix(conf_matrix, 'figures/conf_matrix.png')
visualize_roc_curve(log, X_test, y_test, 'figures/roc_curve.png')