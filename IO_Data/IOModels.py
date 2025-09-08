import pickle
import joblib
import sys
import os


def saveModel(model):

    if len(sys.argv) < 4:
        print("If you want to save models to a specific location run this next time:")
        print("python MainPowerOutagePrediction.py <data_file_path> <Optional: Models_save_Location> ...")
        print("Now saving to current directory")
        modelName = model.__class__.__name__()
        path = f'trained_model_{modelName}'
    else:
        path = sys.argv[3]

    try:
        filename = f'{path}.joblib'
        joblib.dump(model, file)
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
            

def loadModel():

    if len(sys.argv) < 5:
        print("If you want to save models to a specific location run this next time:")
        print("python MainPowerOutagePrediction.py <data_file_path> <Models_save_Location> <Models_to_load_Location: folder_path_or_file_path> ...")
        sys.exit(1)

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
            print(f"{path[path.rfind("."):]} is not a supported file type")
            sys.exit(1)
        
    elif isDirectory:
        #Add function later
        return
    