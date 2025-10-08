from KNN import *

prefix = "WLGNZ"

file_path = f"./{prefix}_processed.csv"

if __name__ == "__main__": 

    param_grid = {
        "n_neighbors": list(range(1, 10)),    # test k from 1 to 10
        "weights": ["uniform", "distance"],   # voting weight
        "metric": ["euclidean", "manhattan", "minkowski"]  # distance metrics
    }

    # Default: random sample of 100000 rows
    run_knn_pipeline(file_path, prefix, param_grid)