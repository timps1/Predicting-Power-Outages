import os
import subprocess

input_folder = "../Weather_Data/USA_training_data"
files = [f for f in os.listdir(input_folder) if f.endswith(".csv")]

for filename in files[0:1]:
    wfo_code = filename[:3]
    input_path = "../CHCNZ_Training_Data_normalised_baselearner_train_set.csv" # os.path.join(input_folder, filename)
    for sus_val in [0.13, 0.2, 0.4]:
        for smote_val in [0.8, 1]:
            sus_str = f"{int(sus_val * 10):02d}"  # e.g., 0.2 -> "02"
            smote_str = f"{int(smote_val * 10)}"
            subprocess.run([
                "python3",
                "MainPowerOutagePrediction.py",
                "MODELS-FILENAME-WRITE=Trained_Models/Models_List_ALLUSASOMENZMODELS.txt",
                "MODELS-FILENAME-READ=Trained_Models/Models_List_ALLUSASOMENZMODELS.txt",
                f"MODEL-SAVE-LABEL={wfo_code}_NZ_SUS{sus_str}_SM{smote_str}",
                "SMOTE-KNN=5",
                f"SUS-SAMP-STRAT={sus_val}", 
                f"SMOTE-SAMP-STRAT={smote_val}",
                input_path
            ])

"""
python3 
MainPowerOutagePrediction.py 
MODELS-FILENAME-WRITE=Trained_Models/Models_List_ALLUSAMODELS.txt 
MODELS-FILENAME-READ=Trained_Models/Models_List_ALLUSAMODELS.txt 
MODEL-SAVE-LABEL=ABQ_SUS02_SM1 
SMOTE-KNN=5 
SUS-SAMP-STRAT=0.2 
SMOTE-SAMP-STRAT=1 
../Weather_Data/USA_training_data/ABQ_Training_Data_normalised.csv 
"""