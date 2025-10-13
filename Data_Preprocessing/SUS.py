from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import ClusterCentroids
from sklearn.cluster import KMeans
from collections import Counter
import sys
import InputLineManagement as ilm

GLOBAL_VERB = 0

def nearMissSUS(X, y, sampling_strategy):

    if GLOBAL_VERB >= 1: print(">Original class distribution:", Counter(y))

    # Initialize NearMiss (selective under-sampling)
    changed = True
    changedSamStrat = False
    while changed:
        try:
            nm = NearMiss(sampling_strategy=sampling_strategy, version=1)  # version=1 selects majority samples closest to minority
            X_resampled, y_resampled = nm.fit_resample(X, y)
            changed = False
        except:
            changedSamStrat = True
            if type(sampling_strategy) == float:
                sampling_strategy += 0.01
                if sampling_strategy > 1:
                    return X, y
            else:
                return X, y

    if GLOBAL_VERB >= 1 and changedSamStrat: print(">Changed SUS sample strategy:", sampling_strategy)

    if GLOBAL_VERB >= 1: print(">Resampled class (after near miss) distribution:", Counter(y_resampled))

    return X_resampled, y_resampled


def tomekLinksSUS(X, y, sampling_strategy):
    if GLOBAL_VERB >= 1: print(">Original class distribution:", Counter(y))

    tl = TomekLinks(sampling_strategy=sampling_strategy)  # 'auto' removes majority samples only
    X_resampled_tl, y_resampled_tl = tl.fit_resample(X, y)

    if GLOBAL_VERB >= 1: print(">Resampled class (after tomelink) distribution:", Counter(y_resampled_tl))

    return X_resampled_tl, y_resampled_tl

def clusterCentriodSUS(X, y, sampling_strategy):
    if GLOBAL_VERB >= 1: print(">Original class distribution:", Counter(y))
    kmeans = KMeans(n_clusters=None, random_state=42)

    # Cluster-based under-sampling
    cc = ClusterCentroids(sampling_strategy=sampling_strategy, estimator=kmeans)
    X_resampled_cc, y_resampled_cc = cc.fit_resample(X, y)

    if GLOBAL_VERB >= 1: print(">Resampled class (after cluster centroids) distribution:", Counter(y_resampled_cc))

    return X_resampled_cc, y_resampled_cc

def applySUS(X, y, sampStrat = None):
    global GLOBAL_VERB
    GLOBAL_VERB = int(ilm.getArg("VERBOSE"))
    typeSUS = ilm.getArg("SUS-TYPE")
    if GLOBAL_VERB >= 2: 
        print(">>SUS type:", typeSUS)
        print(">>SUS sample strategy:", ilm.getArg("SUS-SAMP-STRAT"))
    if sampStrat is None:
        if ilm.getArg("SUS-SAMP-STRAT") == "pass":
            return X, y
        elif is_float(ilm.getArg("SUS-SAMP-STRAT")):
            sampStrat = float(ilm.getArg("SUS-SAMP-STRAT"))
        else:
            sampStrat = ilm.getArg("SUS-SAMP-STRAT")
    if typeSUS == "NearMiss":
        return nearMissSUS(X, y, sampStrat)
    elif typeSUS == "TomekLink":
        return tomekLinksSUS(X, y, sampStrat)
    elif typeSUS == "clusterCentriodSUS":
        return nearMissSUS(X, y, sampStrat)
    else:
        print(f"No implementation for {typeSUS}")
        return X, y

def is_float(value):
    try:
        float(value)  # Attempt to convert the string to a float
        return True
    except ValueError:
        return False