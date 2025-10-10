from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import ClusterCentroids
from sklearn.cluster import KMeans
from collections import Counter
import sys

def nearMissSUS(X, y):
    print("Original class distribution:", Counter(y))

    # Initialize NearMiss (selective under-sampling)
    nm = NearMiss(sampling_strategy=float(sys.argv[2]), version=1)  # version=1 selects majority samples closest to minority
    X_resampled, y_resampled = nm.fit_resample(X, y)

    print("Resampled class (after near miss) distribution:", Counter(y_resampled))

    return X_resampled, y_resampled


def tomekLinksSUS(X, y):
    print("Original class distribution:", Counter(y))

    tl = TomekLinks(sampling_strategy='auto')  # 'auto' removes majority samples only
    X_resampled_tl, y_resampled_tl = tl.fit_resample(X, y)

    print("Resampled class (after tomelink) distribution:", Counter(y_resampled_tl))

    return X_resampled_tl, y_resampled_tl

def clusterCentriodSUS(X, y):
    print("Original class distribution:", Counter(y))
    kmeans = KMeans(n_clusters=None, random_state=42)

    # Cluster-based under-sampling
    cc = ClusterCentroids(sampling_strategy="auto", estimator=kmeans)
    X_resampled_cc, y_resampled_cc = cc.fit_resample(X, y)

    print("Resampled class (after cluster centroids) distribution:", Counter(y_resampled_cc))

    return X_resampled_cc, y_resampled_cc

def chooseSUS(typeSUS, X, y):
    if typeSUS == "NearMiss":
        return nearMissSUS(X, y)
    elif typeSUS == "TomekLink":
        return nearMissSUS(X, y)
    elif typeSUS == "Cluster":
        return nearMissSUS(X, y)
    else:
        print(f"No implementation for {typeSUS}")
        return X, y