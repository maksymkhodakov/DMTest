import numpy as np
import matplotlib.pyplot as plt
import random

# -----------------------------
# 1. Вхідні дані Варіант 3
# -----------------------------
data = np.array([
    [1.0, 2.0, 1.5],
    [1.5, 1.8, 2.0],
    [2.0, 2.2, 1.7],
    [8.0, 8.5, 7.8],
    [8.3, 8.0, 8.5],
    [7.8, 8.2, 7.9],
    [0.5, 1.0, 1.2],
    [0.8, 0.5, 0.7],
    [1.1, 1.2, 1.0],
    [5.0, 5.5, 5.2],
    [5.5, 5.0, 5.5],
    [5.2, 5.2, 5.0]
])


# -----------------------------
# 2. Ініціалізація методом "найдальших точок"
# -----------------------------
def initialize_centroids_far_points(data, k):
    centroids = []
    idx = random.randint(0, len(data) - 1)
    centroids.append(data[idx])

    for _ in range(1, k):
        distances = np.min(
            [np.linalg.norm(data - c, axis=1) for c in centroids],
            axis=0
        )
        next_idx = np.argmax(distances)
        centroids.append(data[next_idx])

    return np.array(centroids)


# -----------------------------
# 3. Власна реалізація K-Means
# -----------------------------
def kmeans_custom(data, k, max_iter=100):
    centroids = initialize_centroids_far_points(data, k)

    for _ in range(max_iter):
        labels = np.argmin([np.linalg.norm(data - c, axis=1) for c in centroids], axis=0)

        new_centroids = np.array([
            data[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i]
            for i in range(k)
        ])

        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids

    distances = [np.linalg.norm(data[i] - centroids[labels[i]]) ** 2 for i in range(len(data))]
    inertia = np.sum(distances)

    return labels, centroids, inertia


# -----------------------------
# 4. Elbow-графік
# -----------------------------
inertias = []
K = range(1, 8)

for k in K:
    _, _, inertia = kmeans_custom(data, k)
    inertias.append(inertia)

plt.figure(figsize=(8, 5))
plt.plot(K, inertias, 'bo-')
plt.xlabel('Кількість кластерів (k)')
plt.ylabel('Сума квадратів помилок (Inertia)')
plt.title('Elbow-графік для підбору оптимального k')
plt.grid(True)
plt.show()

# -----------------------------
# 5. Візуалізація кластерів у 3D (наприклад, для k = 4)
# -----------------------------
k = 4  # Можеш змінити вручну, якщо бачиш з графіка, що краще 3 або 5
labels, centroids, _ = kmeans_custom(data, k)

fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection='3d')

colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
for i in range(k):
    cluster_points = data[labels == i]
    ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2],
               label=f'Кластер {i + 1}', color=colors[i % len(colors)], s=50)

# Центроїди
ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2],
           color='black', s=200, marker='X', label='Центроїди')

ax.set_title(f'Візуалізація кластерів (k = {k})')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.tight_layout()
plt.show()
