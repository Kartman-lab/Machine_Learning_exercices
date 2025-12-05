from src.preprocess import preprocess
from src.model import split_data, train_model
from src.visualize import visualize_residu, vizualise_pred_y_vs_test_y

data = preprocess()

X = data["X_selected"]
y = data["y"]

X_train, X_test, y_train, y_test = split_data(X, y)
rmse, mape, y_pred = train_model(X_train, y_train, X_test, y_test)

print("rmse:", rmse)
print("mape:", mape)
