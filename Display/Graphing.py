import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys


def basicPlotData(data, title='Basic Data Plot', xlabel='X-axis', ylabel='Y-axis'):
    """
    Plot the given data.
    
    Parameters:
    - data: DataFrame or array-like structure containing the data to plot.
    - title: Title of the plot.
    - xlabel: Label for the X-axis.
    - ylabel: Label for the Y-axis.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()

def barPlot(series_list, title="", xlabel="", ylabel="", legend=None):
    """
    series_list: list of series, each series = [folds_list, values_list]
    legend: list of labels for each series
    """
    num_series = len(series_list)
    folds = series_list[0][0]  # assume all series have same folds
    num_folds = len(folds)
    
    x = np.arange(num_folds)
    bar_width = 0.8 / num_series  # leave space between groups
    
    plt.figure(figsize=(10, 6))
    
    for i, (folds_i, values_i) in enumerate(series_list):
        plt.bar(x + i*bar_width, values_i, width=bar_width,
                label=legend[i] if legend else f'Series {i}')
    
    plt.xticks(x + bar_width*(num_series-1)/2, folds)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(axis='y')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    df = pd.read_csv(sys.argv[1])
    series_list = [df[["Fold", "Average_Accuracy"]].values.T.tolist()]
    barPlot(series_list, title="", xlabel="", ylabel="", legend=None)