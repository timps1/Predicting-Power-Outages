import sys

def getInputData():
    """
    Function to get input data for the power outage prediction project.
    This function is designed to be called from the command line.
    """
    if len(sys.argv) < 2:
        print("Usage: python MainPowerOutagePrediction.py <data_file_path>")
        sys.exit(1)
    
    # Data Location
    data_file_path = sys.argv[1]
    
    try:
        import pandas as pd
        data = pd.read_csv(data_file_path)
        return data
    except Exception as e:
        print(f"Error reading data file: {e}")
        sys.exit(1)

def SaveInputData(data, output_file_path):
    """
    Function to save input data to a specified file path.
    """
    try:
        data.to_csv(output_file_path, index=False)
        print(f"Data saved to {output_file_path}")
    except Exception as e:
        print(f"Error saving data file: {e}")