from keras.models import load_model
from matplotlib.image import imread
import ShipDetection
import sys
'''Model inference file
    parameters are:
    <Path to chosen model> <Path to chosen image> <Path for output image>
'''

def OpenModel(path):
    '''Function for opening a saved model'''
    model = load_model(path)
    return model

def getPicMask(model, picPath):
    '''Function for predicting a pics mask'''
    pic = imread(picPath)
    return model.predict(pic)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        try:
            model = OpenModel(sys.argv[1])
            mask = getPicMask(model,sys.argv[2])
            newPic = ShipDetection.applyMask(sys.argv[2],mask)
            if len(sys.argv) >= 4:
                ShipDetection.saveOutput(sys.argv[3],newPic)
            else:
                ShipDetection.saveOutput("output/outPic.jpg",newPic)
        except:
            print("Wrong arguments")
