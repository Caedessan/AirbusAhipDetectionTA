import numpy as np
import tensorflow as tf
from keras.models import load_model
from matplotlib.image import imread
import ShipDetection
import sys
'''Model inference file
    parameters are:
    <Path to chosen model> <Path to chosen image> <Path for output image>
'''

if __name__ == '__main__':
    if len(sys.argv) > 1:
            try:
                model = load_model(sys.argv[1], custom_objects = {"dice_coef_loss":ShipDetection.dice_coef_loss})
                pic = imread(sys.argv[2])
                with tf.device("/CPU:0"):
                    res = model.predict(np.array([pic]))
                mask = np.rint(np.array(res[0])).reshape((256, 256))
                newPic = ShipDetection.applyMask(sys.argv[2], mask)
                if len(sys.argv) >= 4:
                    ShipDetection.saveOutput(sys.argv[3],newPic)
                else:
                    ShipDetection.saveOutput("output/outPic.jpg",newPic)
            except:
                print("Wrong arguments")
# model = load_model("modelFinal1", custom_objects = {"dice_coef_loss":ShipDetection.dice_coef_loss})
#
# pic = imread("data/train_v2/fcfc12d8d.jpg")
# with tf.device("/CPU:0"):
#     res = model.predict(np.array([pic]))
# mask = np.rint(np.array(res[0])).reshape((256,256))
# newPic = ShipDetection.applyMask("data/train_v2/fcfc12d8d.jpg", mask)
# ShipDetection.saveOutput("output/outPic.jpg", newPic)