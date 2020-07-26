from keras.models import Model, Input
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Activation, Dropout, Dense, Flatten, BatchNormalization, AveragePooling2D
from keras.layers import Concatenate, Add, Reshape, Multiply
from keras.utils import to_categorical
from keras.callbacks import Callback, ModelCheckpoint, CSVLogger
from keras.optimizers import RMSprop
from keras.datasets import cifar10                  # ----------------------------??

import keras.backend as K
import numpy as np

from tqdm.auto import tqdm
import cv2

from model import Inception_ResNet_SENet


classes = 10
smoothing_param = 0.1

def Upscaling_Data(data_list, reshape_dim):         # ----------------------------??
    resized_imgs = []
    for img in tqdm(data_list):
        resized_imgs.append( cv2.resize(img, (reshape_dim[0], reshape_dim[1])) )
    # return np.stack(resized_imgs)
    return resized_imgs


def load_data():
    (_, _), (x_train, y_train) = cifar10.load_data()
    x_train = x_train[:100]
    y_train = y_train[:100]
    print(x_train.shape)

    data_upscaled = np.zeros((100, 3, 299, 299))

    for i, img in enumerate(x_train):
        im = img.transpose((1, 2, 0))
        large_img = cv2.resize(
            im, dsize=(299, 299), interpolation=cv2.INTER_CUBIC)
        data_upscaled[i] = large_img.transpose((2, 0, 1))

    y_train = to_categorical(y_train, 10)

    return data_upscaled, y_train


classes = 10
smoothing_param = 0.1

def smoothed_categorical_crossentropy(y_true, y_pred): 
    if smoothing_param > 0:
        smooth_positives = 1.0 - smoothing_param 
        smooth_negatives = smoothing_param / classes 
        y_true = y_true * smooth_positives + smooth_negatives 

    return K.categorical_crossentropy(y_true, y_pred)


class LearningRateSchedule(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch+1)%2 == 0:
            lr = K.get_value(self.model.optimizer.lr)
            K.set_value(self.model.optimizer.lr, lr*0.94)


input_shape = (299, 299, 3)
# input_shape = (224, 224, 3)
# (x_train, y_train), (x_test, y_test) = cifar10.load_data()      # ----------------------------??

# x_train = Upscaling_Data(x_train, input_shape)      # ----------------------------??
# x_test = Upscaling_Data(x_test, input_shape)        # ----------------------------??

# print ("Training data:")
# print ("Number of examples: ", x_train.shape[0])
# print ("Number of channels:",x_train.shape[3]) 
# print ("Image size:", x_train.shape[1], x_train.shape[2])
# print ("Test data:")
# print ("Number of examples:", x_test.shape[0])
# print ("Number of channels:", x_test.shape[3])
# print ("Image size:", x_test.shape[1], x_test.shape[2]) 

# x_train = x_train.astype('float32')/255.
# x_test = x_test.astype('float32')/255.

# y_train = to_categorical(y_train, num_classes=classes)
# y_test = to_categorical(y_test, num_classes=classes)

x_train, y_train = load_data()

model_input = Input( shape=input_shape )

# model = Inception_v3(model_input)
model = Inception_ResNet_SENet(model_input, classes=classes)
# model = Inception_ResNet_SENet(model_input)

optimizer = RMSprop(lr=0.045, epsilon=1.0, decay=0.9)
filepath = 'weights/' + model.name + '.h5'
callbacks_list = [ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_weights_only=True, save_best_only=True, mode='auto', period=1),
                  CSVLogger(model.name + '.log'),
                  LearningRateSchedule()]

# model.compile(optimizer, 
#         	loss = { 'main_classifier' : smoothed_categorical_crossentropy, 'auxiliary_classifier' : smoothed_categorical_crossentropy},
#             loss_weights={'main_classifier' : 1.0, 'auxiliary_classifier' : 0.4},
#             metrics=['acc'])

model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, [y_train, y_train], batch_size=32, epochs=100, validation_split=0.2, callbacks=callbacks_list)
