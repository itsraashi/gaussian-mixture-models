import numpy as np
# import matplotlib.pyplot as plt

def load_data():
    """ Load randomly permuted MNIST dataset

        Returns an 30000-by-784 numpy ndarray X_train
    """

    X = np.load('data/mnist.npy').astype(np.float32)
    X_train = np.reshape(X[:30000][:][:], (30000, 784))
    return X_train

def kmeans_loss(X, C, z):
    """ Compute the K-means loss.

        Input:
        X: a numpy ndarray with shape (N,M), where each row is a data point
        C: a numpy ndarray with shape (K,M), where each row is a cluster center
        z: a numpy ndarray with shape (N,) where the i-th entry is an int from {0..K-1}
            representing the cluster index for the i-th point in X

        Returns mean squared distance from each point to the center for its assigned cluster
    """
    N = X.shape[0]
    loss = 0
    for i in range(N):
        diff = X[i] - C[z[i]]
        loss += diff @ diff / N

    return loss


# Feel free to add any helper functions you need here
### YOUR CODE HERE



def kmeans(X, K):
    """ Cluster data X into K converged clusters.
    
        X: an N-by-M numpy ndarray, where we want to assign each
            of the N data points to a cluster.

        K: an integer denoting the number of clusters.

        Returns a tuple of length two containing (C, z):
            C: a numpy ndarray with shape (K,M), where each row is a cluster center
            z: a numpy ndarray with shape (N,) where the i-th entry is an int from {0..K-1}
                representing the cluster index for the i-th point in X
    """
    N = X.shape[0]

    # Initialize cluster centers to the first K points of X
    C = np.copy(X[:K])

    # Initialize z temporarily to all -1 values
    z = -1*np.ones(N, dtype=np.int)

    (K, M) = C.shape
    avgs = C

    var = True
    while(var):
        for n in range(N):
            min_dist = float("inf")
            for k in range(K):
                dist = np.linalg.norm(C[k] - X[n])
                if dist < min_dist:
                    min_dist = dist
                    z[n] = k

        points = [[] for k in range(K)]
        for i in range(N):
            k = z[i]
            points[k].append(X[i])

        new_avgs = [np.mean(points[k], axis=0) for k in range(K)]
        if np.array_equal(new_avgs, avgs):
            var = False

        avgs = new_avgs
        C = np.asarray(avgs)

    return (C, z)

