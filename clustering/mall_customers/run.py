from src.preprocessing import scale_features, data
from src.clustering import create_kmeans, elbow_method, show_clusters
from src.visualization import visualize_2D, visualize_clusters_3D

X = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].values

X_scaled = scale_features(data)
elbow_method(X_scaled)

labels = create_kmeans(X_scaled)
visualize_2D(data, labels)
show_clusters(labels, data)
visualize_clusters_3D(X, labels)
