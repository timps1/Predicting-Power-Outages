import sys
from Models import KNNModel, SVMModel, XGBoostModel
from IO_Data import IOData, IOModels
from Model_Evaluation import CrossValidation
from Data_Preprocessing import NormaliseData
from Display import Graphing

def main():
    """
    Main function to run the power outage prediction project.
    """

    # Get input data
    data = IOData.getInputData()

    # Normalize/Preprocess data --- If normalization is needed
    print("Preprocess data...")
    methodLabel='zero_to_one'
    normalizer = NormaliseData.DataNormalizer()
    normalizer.normalizeDataByColumn(data, methodLabel)
    
    # Save normalized data
    IOData.SaveInputData(data, 'normalized_data.csv')

    # Initialize model
    modelKNN = KNNModel.KNN()
    # modelSVM = SVMModel.SVM()
    # modelXGBoost = XGBoostModel.XGBoost()

    modelsList =[modelKNN] #, modelSVM, modelXGBoost]

    # Split data into features and target
    targetLabel = 'outage'  # Assuming 'outage' is the target column
    if targetLabel not in data.columns:
        print(f"Column '{targetLabel}' does not exist in the dataset.")
        print("Available columns:", data.columns)
        sys.exit(1)
    
    X = data
    y = data[targetLabel]

    # Cross-validation
    cross_validator = CrossValidation.CrossValidator(n_splits=5, storeResults=True)
    resultsOfCrossValidation = []

    for model in modelsList:
        cross_validator.setModel(model)
        print(f"Cross-validating model: {model.__class__.__name__}")
        cross_validator.cross_validate(X.values, y.values)
        resultsOfCrossValidation.append(tuple(cross_validator.scores))

    # Display results
    display = Graphing.Graphs()
    display.barPlot(resultsOfCrossValidation, 
                    title='Bar Plot', 
                    xlabel='Categories', 
                    ylabel='Values', 
                    legend=[model.__class__.__name__ for model in modelsList])
    
if __name__ == "__main__":
    print("Starting power outage prediction project...")
    print("--------------------------------------------------------------------")
    main()
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("Power outage prediction project completed successfully.")