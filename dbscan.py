import numpy as np
from scipy.spatial import KDTree
from sklearn.datasets.samples_generator import make_blobs
from contexttimer import Timer
import matplotlib.pyplot as plt


def dbscan(points, min_points, epsilon):
    """
    Return an array of labels for each point indicating which cluster it belongs to.
    Cluster labels start with index 0. Ouliers have -1.
    """
    OUTLIER = -1
    UNDEFINED = -2
    if not len(points):
        return []
    kdtree = KDTree(points)
    labels = [UNDEFINED] * len(points)

    cluster = 0
    for i, p in enumerate(points):
        if labels[i] != UNDEFINED:
            continue

        neighbors = kdtree.query_ball_point(p, epsilon)
        neighbors = set(neighbors)
        if len(neighbors) < min_points:
            labels[i] = OUTLIER
            continue

        neighbors.remove(i)
        labels[i] = cluster
        while neighbors:
            n = neighbors.pop()
            if labels[n] == OUTLIER:
                labels[n] = cluster
                continue
            if labels[n] != UNDEFINED:
                continue
            labels[n] = cluster
            neighbors_inner = kdtree.query_ball_point(points[n], epsilon)
            if len(neighbors_inner) >= min_points:
                neighbors.update(neighbors_inner)
        cluster += 1
    return labels


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


def test_dbscan_simple():
    ps = [(0, 0), (0, 1), (1, 0), (5, 5), (5, 6)]
    labels = dbscan(ps, 3, 2)
    np.testing.assert_equal(labels, (0, 0, 0, -1, -1))

def test_dbscan():
    centers = [(1, 1), (-1, -1), (1, -1)]
    ps, labels_true = make_blobs(n_samples=100, centers=centers, cluster_std=0.1, random_state=0)
    labels = np.array(dbscan(ps, 10, 0.3))

    check_equal_clusters(labels, labels_true)


def profile():
    N = 10000
    scale = N**(1/2)
    ps = np.random.uniform(-1*scale, 1*scale, (N, 2))
    with Timer() as t:
        labels = dbscan(ps, 3, 0.5)
    print("elapesed: {:.4f}".format(t.elapsed), "num labels:", len(set(labels)))

if __name__ == "__main__":
    np.random.seed(0)
    profile()
    # import cProfile
    # cProfile.run('profile()')
    # Ns = np.array([10000])#range(10, 10000, 2000))
    # ts = []
    # for N in Ns:
    #     scale = N**(1/2)
    #     ps = np.random.uniform(-1*scale, 1*scale, (N, 2))
    #     with Timer() as t:
    #         labels = dbscan(ps, 3, 0.5)
    #     ts.append(t.elapsed)
    #     print("elapesed: {:.4f}".format(t.elapsed), "num labels:", len(set(labels)))
    
    # plt.plot(Ns, ts)
    # plt.plot(Ns, np.log(Ns)*Ns / (np.log(Ns[-1])*Ns[-1]) * ts[-1])
    # plt.show()