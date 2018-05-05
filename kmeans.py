from sklearn.datasets.samples_generator import make_blobs
import numpy as np
import math
import matplotlib.pyplot as plt

def kmeans(points, k, eps=1e-10):
    points = np.asarray(points)

    labels = np.zeros(points.shape[0], dtype=int)

    means = initial_means(points, k)
    means_new = means

    max_change = eps + 1
    while max_change > eps:
        for i, p in enumerate(points):
            # find index of closest cluster mean
            cluster = np.argmin([dist(p, m) for m in means])
            labels[i] = cluster
        for cluster in range(k):
            means_new[cluster,:] = points[labels==cluster].mean(axis=0)

        max_change = max([dist(m, m_new) for m, m_new in zip(means, means_new)])
        means = np.array(means_new)
    return labels

def initial_means(points, k):
    num_points = points.shape[0]
    probs = np.zeros(num_points)
 
    idx = np.random.choice(num_points, 1)[0]
    means = [points[idx,:]]

    for _ in range(k-1):
        # choose next mean proportional to the squared distance from the closest already chosen mean
        for i, p in enumerate(points):
            closest = min(means, key=lambda m: dist(p, m))
            probs[i] = dist(p, closest)**2
        
        idx = np.random.choice(num_points, 1, p=probs/sum(probs))[0]
        means.append(points[idx,:])

    return np.asarray(means)

def dist(p1, p2):
    s = sum((x-y)**2 for x, y in zip(p1, p2))
    return math.sqrt(s)


def unique_items(array):
    _, idx = np.unique(array, return_index=True)
    return array[np.sort(idx)]

def check_equal_clusters(labels1, labels2):
    labels1 = np.array(labels1)
    unique1 = unique_items(labels1)
    labels2 = np.array(labels2)
    unique2 = unique_items(labels2)

    def translate(l):
        return unique2[unique1 == l][0]
    labels1 = list(map(translate, labels1))
    np.testing.assert_equal(labels1, labels2)


def test_kmeans():
    centers = [(1, 1), (-1, -1), (1, -1), (10,0), (10, 5)]
    ps, labels_true = make_blobs(n_samples=100, centers=centers, cluster_std=0.6, random_state=0)
    k = len(centers)
    labels = kmeans(ps, k=k)
    # check_equal_clusters(labels, labels_true)

    for cluster in range(k):
        c = ps[labels==cluster]
        plt.scatter(c[:,0], c[:,1])
    plt.show()

if __name__ == "__main__":
    test_kmeans()
