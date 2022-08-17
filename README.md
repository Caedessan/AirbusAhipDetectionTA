This is an Airbus ship detection project for Kaggle competition

DataAnalysis.ipynb consists of data as analysis well as its preprocessing

ShipDetection.py contains all needed functions for the project as well as model structure 

DataGenerator.py contains DataGenerator class for generating training data for model fitting

ModelFitting.ipynb contains model creation and fitting

ModelFitting.py is the same as ModelFitting.ipynb

ModelInference.py contains methods to use the trained model

model1.h5 is a trained model


ModelInference can be called from terminal with parameters:
    <p>python ModelInference  [Path to chosen model][Path to chosen image][Path for output image]</p>