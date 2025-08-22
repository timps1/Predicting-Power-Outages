import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Graphs:

    def __init__(self):
        """
        Initialize the Graphs class.
        """
        pass

    def basicPlotData(self, data, title='Basic Data Plot', xlabel='X-axis', ylabel='Y-axis'):
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

    def barPlot(self, data, title='Bar Plot', xlabel='Categories', ylabel='Values', legend=[]):
        """
        Create a bar plot for the given data.
        
        Parameters:
        - data: DataFrame or Series containing the data to plot.
        - title: Title of the plot.
        - xlabel: Label for the X-axis.
        - ylabel: Label for the Y-axis.
        """
        
        plt.figure(figsize=(10, 6))
        for i, (fold, accuracy) in enumerate(data):
            plt.bar(fold, accuracy, label=legend[i] if legend else f'Legend Label {i}')
        plt.legend()
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(axis='y')
        plt.show()