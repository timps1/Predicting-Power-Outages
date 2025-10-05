from imblearn.over_sampling import SMOTE

def apply_smote(x, y):
    """
    Input:
        x: data column (excluding flag)
        y: flag column
    returns:
        x: SMOTE data column
        y: SMOTE flag column
    """
    
    smote = SMOTE(sampling_strategy='minority')
    x,y = smote.fit_resample(x,y)
    return x,y








