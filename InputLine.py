import sys

GLOBAL_DICTIONARY = {
    "SMOTE-SAMP-STRAT" : ["pass", "SMOTE(sample....)"],
    "SUS-TYPE" : ["NearMiss", ""],
    "SUS-SAMP-STRAT" : ["pass", "<SUS-TYPE>(sampling_strategy)"],
    "PRED-SET" : [None, "For Predicting final models"],
    "INPUT" : [0, "Input data"],
    "VERBOSE" : [0, "Debugging information messages"],
    "RM-FOLDS" : [True, ""]
}

def getArg(keyword):
    
    for argument in sys.argv:
        if 
