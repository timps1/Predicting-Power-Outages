import sys
from Models import KNNModel, SVMModel, XGBoostModel
from IO_Data import IOData, IOModels
from Model_Evaluation import CrossValidation
from Data_Preprocessing import NormaliseData
from Display import Graphing
import pandas as pd



def main():
    """
    Main function to run the power outage prediction project.
    """

    # Get input data
    data = IOData.getInputData()

    if False: #Just for normalizing data/preprocessing
        # Normalize/Preprocess data --- If normalization is needed
        print("Preprocess data...")
        methodLabel='zero_to_one'
        normalizer = NormaliseData.DataNormalizer()
        normalizer.normalizeDataByColumn(data, methodLabel)
        
        # Save normalized data
        print("Save normalized data...", end=" ")
        if sys.argv[1].rfind("/") != -1:
            print(f'{sys.argv[1][:sys.argv[1].rfind("/")+1]}normalized_{sys.argv[1][sys.argv[1].rfind("/")+1:]}')
            IOData.SaveInputData(data, f'{sys.argv[1][:sys.argv[1].rfind("/")+1]}normalized_{sys.argv[1][sys.argv[1].rfind("/")+1:]}')
        else:
            print(f'normalized_{sys.argv[1]}')
            IOData.SaveInputData(data, f'normalized_{sys.argv[1]}')
        return
    
    print(len(sys.argv))
    if len(sys.argv) > 4:
        ############## Not sure if it works ##############
        # Load pre-trained model
        modelsRetrainList = IOModels.loadModel()
        print(f"Loaded model: {model.__class__.__name__}")
        sys.exit(0)
    
    else:

        modelClassParameterList =[]

        for i in range(11, 50, 5):
            modelClassParameterList.append((SVMModel.SVM, ('rbf',i)))
        

    # Split data into features and target
    targetLabel = 'outage_flag'  # Assuming 'outage' is the target column
    if targetLabel not in data.columns:
        print(f"Column '{targetLabel}' does not exist in the dataset.")
        print("Available columns:", data.columns)
        sys.exit(1)
    
    X = data
    y = data[targetLabel]
    X = data.drop(columns=[targetLabel])  # features only

    if not check_for_non_numeric_values(X):
        print("Data contains non-numeric values. Please preprocess the data to convert all features to numeric types.")
        sys.exit(1)

    # Cross-validation
    cross_validator = CrossValidation.CrossValidator(n_splits=10, storeResults=True)

    modelsList = []
    current_model_type = None

    for i, (model_class, parameters) in enumerate(modelClassParameterList):

        model = model_class(parameters)  # Re-initialize model with parameters

        if current_model_type is None:
            current_model_type = model.getType()

        elif current_model_type != model.getType():
            current_model_type = model.getType()
            cross_validator.reset()  # Reset cross-validator for new model type

        print(f"------------------------------------------------------------")
        print(f"Model {i}: {model.getModelInfo()}")

        cross_validator.storeResults = (i == len(modelsList) - 1)

        cross_validator.setModel(model)
        print(f"Cross-validating model: {model.__class__.__name__}")
        cross_validator.crossValidate(X.values, y.values)

        modelsList.append(model)
        print(f"------------------------------------------------------------")
    
    cross_validator.printBestModel()
    # Display results
    # display = Graphing.Graphs()
    # display.barPlot(cross_validator.scores, 
    #                 title='Bar Plot', 
    #                 xlabel='Categories', 
    #                 ylabel='Values', 
    #                 legend=[model.getModelInfo() for model in modelsList])
    
    # Save models
    # for model in modelsList:
    #     IOModels.saveModel(model)


def check_for_non_numeric_values(df):
    """
    Checks for any values in the DataFrame that are not real numbers (int or float).
    """
    non_numeric_rows = {}
    
    for col in df.columns:
        mask = ~df[col].apply(lambda x: isinstance(x, (int, float)) and not pd.isna(x))
        if mask.any():
            non_numeric_rows[col] = df[col][mask].unique()
    
    if non_numeric_rows:
        print("Found non-numeric values in columns:")
        for col, vals in non_numeric_rows.items():
            print(f"{col}: {vals}")
        return False
    else:
        print("All values are numeric.")
        return True

if __name__ == "__main__":
    print("Starting power outage prediction project...")
    print("--------------------------------------------------------------------")
    main()
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("Power outage prediction project completed successfully.")