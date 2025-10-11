from imblearn.over_sampling import SMOTE
from collections import Counter
import sys
import InputLineManagement as ilm

GLOBAL_VERB = ilm.getArg("VERBOSE")

def apply_smote(x, y):
    """
    Input:
        x: data column (excluding flag)
        y: flag column
    returns:
        x: SMOTE data column
        y: SMOTE flag column
    """
    if GLOBAL_VERB == 1: print("Distribution before SMOTE:", Counter(y))
    if is_float(ilm.getArg("SMOTE-SAMP-STRAT")):
        smote = SMOTE(sampling_strategy=float(ilm.getArg("SMOTE-SAMP-STRAT")), random_state=42)
    elif ilm.getArg("SMOTE-SAMP-STRAT") != "pass":
        smote = SMOTE(sampling_strategy=ilm.getArg("SMOTE-SAMP-STRAT"), random_state=42)
    else:
        return x,y
    x,y = smote.fit_resample(x,y)
    if GLOBAL_VERB == 1: print("Distribution after SMOTE:", Counter(y))
    return x,y

def is_float(value):
    try:
        float(value)  # Attempt to convert the string to a float
        return True
    except ValueError:
        return False






