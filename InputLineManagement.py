import sys

GLOBAL_DICTIONARY = {
    "SMOTE-SAMP-STRAT" : ["pass", "SMOTE(sample....) - type float"],
    "SUS-TYPE" : ["NearMiss", "NearMiss, TomekLink, Cluster"],
    "SUS-SAMP-STRAT" : ["pass", "<SUS-TYPE>(sampling_strategy)"],
    "PRED-SET" : [None, "For Predicting final models"],
    "INPUT" : [0, "Input data"],
    "SECONDARY-INPUT" : [0, "Secondary Input date to add onto the original data"],
    "ENSEMBLE-INPUT" : [None, ""],
    "VERBOSE" : [0, "Debugging information messages"],
    "RM-FOLDS" : [1, "Remake the folders, most cases this should be true"],
    "SAVE-RM-FOLDS" : [1, "Saving folds"],
    "MODEL-SAVE-LABEL" : ["", "Tag to differenciate models saved"],
    "MODELS-FILENAME-WRITE" : [None, 'Text filename to write Trained_Models/<model_filename>,tag e.g. Trained_Models/rf_model.joblib,rf\\n...'],
    "MODELS-FILENAME-READ" : [None, 'Reading a file of the same format of MODELS-FILENAME-WRITE'],
    "SMOTE-KNN" : [5, "k_neighbors for smote"],
    "TEST-SET" : [None, "For prediction at the end"]
}

def intialiseGLOBAL_DICTIONARY():
    global GLOBAL_DICTIONARY
    
    if "help" in sys.argv[1]:
        printHelp()
        sys.exit(0)

    for argument in sys.argv[1:]:
        
        if "=" in argument:
            inputKey, value = argument.split("=")
            inputKey = inputKey.upper()
            if inputKey in GLOBAL_DICTIONARY:
                GLOBAL_DICTIONARY[inputKey][0] = value
            else:
                print(f"ERROR IN INPUTLINE: {inputKey} is not a key in the GLOBAL_DICTIONARY")
                printHelp()
                sys.exit(1)
        elif GLOBAL_DICTIONARY["INPUT"][0] == 0:
            GLOBAL_DICTIONARY["INPUT"][0] = argument
        else:
            print(f"Trying to set multiple sets of data as Input")
            printHelp()
            sys.exit(1)

def getArg(keyword):
    if keyword in GLOBAL_DICTIONARY:
        return GLOBAL_DICTIONARY[keyword][0]
    else:
        print(f"ERROR IN CODE: {keyword} is not a key in the GLOBAL_DICTIONARY")
        printHelp()
        sys.exit(1)

def setArg(keyword):
    if keyword in GLOBAL_DICTIONARY:
        return GLOBAL_DICTIONARY[keyword][0]
    else:
        print(f"ERROR IN CODE: {keyword} is not a key in the GLOBAL_DICTIONARY")
        printHelp()
        sys.exit(1)

def takeArg(keyword, default=None):

    for arg in sys.argv[1:]:
        if arg.startswith(f"{keyword}="):
            return arg.split("=", 1)[1]
    return default

def printHelp():
    print("Brief help for running code in this Repo")
    print()
    for aKey in GLOBAL_DICTIONARY:
        keyString = f'{aKey}=<Value>'
        print(f'{keyString:<30}:{GLOBAL_DICTIONARY[aKey][1]} | DEFAULT="{GLOBAL_DICTIONARY[aKey][0]}"')
        print()

