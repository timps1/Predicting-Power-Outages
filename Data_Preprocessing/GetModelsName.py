"""
This is a TEMPERARY file and only works if files follow specific format in models folder and excludeList is updated correctly
"""

import os

exculdeList = ["SklearnModels.py"]

modelsList = os.listdir("./Models")
print("Files in Models:", modelsList)

for filenameToExclude in exculdeList:
    modelsList.remove(filenameToExclude)

newModelsList = "["

for i in range(len(modelsList)):
    modelName = modelsList[i][:modelsList[i].find("Model.py")]
    print(f"model{modelName} = {modelName}Model.{modelName}()")
    newModelsList += f"model{modelName}, "

print("modelsList =" + newModelsList[:-2] + "]")