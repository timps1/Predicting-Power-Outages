from imblearn.over_sampling import SMOTE

def apply_smote(x, y, sampling_strategy="auto", k_neighbors=5):
    """
    Input:
        x: data column (excluding flag)
        y: flag column
    returns:
        x: SMOTE data column
        y: SMOTE flag column
    """

    try:
        smote = SMOTE(k_neighbors=k_neighbors, sampling_strategy=sampling_strategy, random_state=42)
        x,y = smote.fit_resample(x,y)
    except:
        print("SMOTE passed on")
    return x,y






