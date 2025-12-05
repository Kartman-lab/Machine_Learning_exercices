import matplotlib.pyplot as plt 
import os

def visualise_y_test_vs_y_pred(y_pred, y_test, title, save_path):
    plt.figure(figsize=(12, 8))
    plt.scatter(y_test, y_pred, alpha=0.5)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='prediction')
    plt.xlabel("y_test (valeurs réelles)")
    plt.ylabel("y_pred (valeurs prédites)")
    plt.title(f"Comparaison y_test vs y_pred - {title}")
    plt.legend()
    plt.grid(True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.show()

    print(f"Graphique sauvegardé dans {save_path}")


def visualize_residu(y_pred, y_test, title, save_path):
    residuals = y_test - y_pred

    plt.figure(figsize=(12, 8))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--', linewidth=2)
    plt.xlabel("Valeurs prédites (y_pred)")
    plt.ylabel("Résidus (y_test - y_pred)")
    plt.title(f"Graphique des résidus - {title}")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.show()

    print(f"Graphique sauvegardé dans {save_path}")

