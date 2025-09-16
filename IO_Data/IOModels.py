import pickle
import joblib
import sys
import os


def saveModel(model):

    if len(sys.argv) < 4:
        modelName = model.getModelInfo()
        path = f'Saved_Models/trained_model_{modelName}'
    else:
        if sys.argv[3] == "_":
            modelName = model.getModelInfo()
            path = f'Saved_Models/trained_model_{modelName}'
        else:
            path = sys.argv[3]

    try:
        filename = f'{path}.joblib'
        joblib.dump(model, filename)
    except Exception as ex:
        print("joblib library failed to save")
        print("Error message", ex)
        try:
            filename = f'{path}.pkl'
            with open(filename, 'wb') as file:
                pickle.dump(model, file)
        except Exception as ex:
            print("pickle library failed to save")
            print("Error message", ex)
            

def loadModel(file=None):

    if len(sys.argv) < 5:
        print("If you want to save models to a specific location run this next time:")
        print("python MainPowerOutagePrediction.py <data_file_path> <Models_save_Location> <Models_to_load_Location: folder_path> ...")
        sys.exit(1)

    if file is not None:
        path = file
    else:
        path = sys.argv[4]
        isFile = False
        isDirectory = False

    if os.path.isfile(path):
        print(f"'{path}' is a file.")
        isFile = True
    elif os.path.isdir(path):
        print(f"'{path}' is a directory.")
        isDirectory = True
    else:
        print(f"'{path}' is neither a file nor a directory or does not exist.")
        sys.exit(1)
    
    if isFile:
        if path[path.rfind("."):] == ".joblib":
            loaded_model = joblib.load(path)
        elif path[path.rfind("."):] == ".pkl":
            with open(path, 'rb') as file:
                loaded_model = pickle.load(file)
        else:
            indexDot = path.rfind(".")
            print(f"{path[indexDot:]} is not a supported file type")
            sys.exit(1)
        
    elif isDirectory:
        loaded_model = []
        files = os.listdir(path)
        model_files = [f for f in files if f.endswith('.joblib') or f.endswith('.pkl')]
        if not model_files:
            print("No model files found in the directory.")
            sys.exit(1)

        for file in model_files:
            full_path = os.path.join(path, file)
            loaded_model.append(loadModel(full_path))

    return loaded_model
    