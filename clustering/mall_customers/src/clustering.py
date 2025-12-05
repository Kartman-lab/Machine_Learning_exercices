from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def create_kmeans(X_scaled, k=3):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    labels = kmeans.labels_
    return labels

def elbow_method(X_scaled, max_k=10):
    inertias = []
    for k in range(1, max_k +1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)

    plt.plot(range(1, max_k + 1), inertias, marker='o')
    plt.xlabel('Nombre de clusters k')
    plt.ylabel('Inertie (Within-Cluster Sum of Squares)')
    plt.title('Méthode du coude')
    plt.show()

def show_clusters(labels, data):
    # Ajouter les labels dans le dataset
    data['Cluster'] = labels

    print("\n=== Moyennes par cluster ===")
    print(data.groupby('Cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean())

    print("\n=== Répartition du genre dans chaque cluster ===")
    print(data.groupby('Cluster')['Gender'].value_counts(normalize=True))

    print("\n=== Nombre d'individus par cluster ===")
    print(data['Cluster'].value_counts())