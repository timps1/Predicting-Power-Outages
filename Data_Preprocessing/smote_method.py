from imblearn.over_sampling import SMOTE
from collections import Counter
import sys

def apply_smote(x, y):
    """
    Input:
        x: data column (excluding flag)
        y: flag column
    returns:
        x: SMOTE data column
        y: SMOTE flag column
    """
    print("Distribution before SMOTE:", Counter(y))
    smote = SMOTE(sampling_strategy=float(sys.argv[3]), random_state=42)
    x,y = smote.fit_resample(x,y)
    print("Distribution after SMOTE:", Counter(y))
    return x,y








