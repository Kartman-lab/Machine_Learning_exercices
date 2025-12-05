import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_2D(data, labels):
    plt.figure(figsize=(8, 6))
    plt.scatter(
        data['Annual Income (k$)'],
        data['Spending Score (1-100)'],
        c=labels,
        cmap='viridis'
    )

    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.title('Clusters de clients (2D)')
    plt.colorbar(label='Cluster')
    plt.show()

def visualize_clusters_3D(X, labels):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(
        X[:, 0],
        X[:, 1],
        X[:, 2],
        c=labels,
        cmap='viridis',
        s=50
    )
    
    ax.set_xlabel('Age')
    ax.set_ylabel('Annual Income (k$)')
    ax.set_zlabel('Spending Score (1-100)')
    ax.set_title('Clusters de clients (3D)')

    fig.colorbar(scatter, ax=ax, label='Cluster')

    plt.show()

   