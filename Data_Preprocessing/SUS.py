from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import ClusterCentroids
from sklearn.cluster import KMeans

def nearMissSUS(X, y, sampling_strategy):

    # Initialize NearMiss (selective under-sampling)
    nm = NearMiss(sampling_strategy=sampling_strategy, version=1)
    X_resampled, y_resampled = nm.fit_resample(X, y)
    
    return X_resampled, y_resampled


def tomekLinksSUS(X, y, sampling_strategy):

    tl = TomekLinks(sampling_strategy=sampling_strategy)  # 'auto' removes majority samples only
    X_resampled_tl, y_resampled_tl = tl.fit_resample(X, y)

    return X_resampled_tl, y_resampled_tl

def clusterCentriodSUS(X, y, sampling_strategy):
    kmeans = KMeans(n_clusters=None, random_state=42)

    # Cluster-based under-sampling
    cc = ClusterCentroids(sampling_strategy=sampling_strategy, estimator=kmeans)
    X_resampled_cc, y_resampled_cc = cc.fit_resample(X, y)

    return X_resampled_cc, y_resampled_cc

def applySUS(X, y, sampStrat = "auto", typeSUS = "NearMiss"):
    try:
        if typeSUS == "NearMiss":
            return nearMissSUS(X, y, sampStrat)
        elif typeSUS == "TomekLink":
            return tomekLinksSUS(X, y, sampStrat)
        elif typeSUS == "clusterCentriodSUS":
            return nearMissSUS(X, y, sampStrat)
        else:
            print(f"No implementation for {typeSUS}")
            return X, y
    except Exception as e:
        print(f"Passing on SUS: {str(e)[:100]}...")
        return X, y
