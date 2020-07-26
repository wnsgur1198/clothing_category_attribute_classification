# Inception Resnet V2 + SENet

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.models import Model, Input
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Activation, Dropout, Dense, Flatten, BatchNormalization, AveragePooling2D
from keras.layers import Concatenate, Add, Reshape, Multiply, Lambda
import keras.backend as K


# 소스코드 및 네트워크 구조는 아래 링크 참조 ----------------------------
# https://sike6054.github.io/blog/paper/third-post/

def conv2d_bn(x, filters, kernel_size, padding='same', strides=1, activation='relu'):
    x = Conv2D(filters, (kernel_size[0], kernel_size[1]), padding=padding, strides=strides)(x)    
    x = BatchNormalization()(x)
    
    if activation:
        x = Activation(activation)(x)
    
    return x


def Scaling_Residual(Inception, scale):
    x = Lambda(lambda Inception, scale: Inception * scale, arguments={'scale': scale})(Inception)
    x = Activation(activation='relu')(x)
    
    return x


# 소스코드 및 네트워크 구조는 아래 링크 참조 ----------------------------
# https://sike6054.github.io/blog/paper/fourth-post/

# Inception V4와 Inception Resnet V2는 동일한 Stem 구조를 가짐
def Stem(input_tensor, version=None, name=None):
    if version == 'Inception-v4' or version == 'Inception-ResNet-v2':

        x = conv2d_bn(input_tensor, 32, (3, 3), padding='valid', strides=2) # 299x299x3 -> 149x149x32
        x = conv2d_bn(x, 32, (3, 3), padding='valid') # 149x149x32 -> 147x147x32
        x = conv2d_bn(x, 64, (3, 3)) # 147x147x32 -> 147x147x64
        
        branch_1 = MaxPooling2D((3, 3), padding='valid', strides=2)(x)
        branch_2 = conv2d_bn(x, 96, (3, 3), padding='valid', strides=2)
        x = Concatenate()([branch_1, branch_2]) # 73x73x160
        
        branch_1 = conv2d_bn(x, 64, (1, 1))
        branch_1 = conv2d_bn(branch_1, 96, (3, 3), padding='valid')
        branch_2 = conv2d_bn(x, 64, (1, 1))
        branch_2 = conv2d_bn(branch_2, 64, (7, 1))
        branch_2 = conv2d_bn(branch_2, 64, (1, 7))
        branch_2 = conv2d_bn(branch_2, 96, (3, 3), padding='valid')
        x = Concatenate()([branch_1, branch_2]) # 71x71x192
        
        branch_1 = conv2d_bn(x, 192, (3, 3), padding='valid', strides=2) # Fig.4 is wrong
        branch_2 = MaxPooling2D((3, 3), padding='valid', strides=2)(x)
        x = Concatenate(name=name)([branch_1, branch_2]) if name else Concatenate()([branch_1, branch_2]) # 35x35x384
        
    elif version == 'Inception-ResNet-v1':
        x = conv2d_bn(input_tensor, 32, (3, 3), padding='valid', strides=2) # 299x299x3 -> 149x149x32
        x = conv2d_bn(x, 32, (3, 3), padding='valid') # 149x149x32 -> 147x147x32
        x = conv2d_bn(x, 64, (3, 3)) # 147x147x32 -> 147x147x64
        
        x = MaxPooling2D((3, 3), strides=2, padding='valid')(x) # 147x147x64 -> 73x73x64
        
        x = conv2d_bn(x, 80, (1, 1)) # 73x73x64 -> 73x73x80
        x = conv2d_bn(x, 192, (3, 3), padding='valid') # 73x73x80 -> 71x71x192U
        x = conv2d_bn(x, 256, (3, 3), padding='valid', strides=2, name=name) # 71x71x192 -> 35x35x256
        
    else:
        return None # Kill ^^
    
    return x


reduction_table = { 'Inception-ResNet-v2' : [256, 256, 384, 384] }

def Reduction_A(input_tensor, version=None, name=None):
    k, l, m, n = reduction_table[version]

    branch_1 = MaxPooling2D((3, 3), padding='valid', strides=2)(input_tensor)

    branch_2 = conv2d_bn(input_tensor, n, (3, 3), padding='valid', strides=2)

    branch_3 = conv2d_bn(input_tensor, k, (1, 1))
    branch_3 = conv2d_bn(branch_3, l, (3, 3))
    branch_3 = conv2d_bn(branch_3, m, (3, 3), padding='valid', strides=2)

    filter_concat = Concatenate(name=name)([branch_1, branch_2, branch_3]) if name else Concatenate()([branch_1, branch_2, branch_3])

    return filter_concat


def Reduction_B(input_tensor, version=None, name=None):
    if version == 'Inception-v4':
        branch_1 = MaxPooling2D((3, 3), padding='valid', strides=2)(input_tensor)
    
        branch_2 = conv2d_bn(input_tensor, 192, (1, 1))
        branch_2 = conv2d_bn(branch_2, 192, (3, 3), padding='valid', strides=2)
    
        branch_3 = conv2d_bn(input_tensor, 256, (1, 1))
        branch_3 = conv2d_bn(branch_3, 256, (1, 7))
        branch_3 = conv2d_bn(branch_3, 320, (7, 1))
        branch_3 = conv2d_bn(branch_3, 320, (3, 3), padding='valid', strides=2)
    
        filter_concat = Concatenate(name=name)([branch_1, branch_2, branch_3]) if name else Concatenate()([branch_1, branch_2, branch_3])

    elif version == 'Inception-ResNet-v1':
        branch_1 = MaxPooling2D((3, 3), padding='valid', strides=2)(input_tensor)
    
        branch_2 = conv2d_bn(input_tensor, 256, (1, 1))
        branch_2 = conv2d_bn(branch_2, 384, (3, 3), padding='valid', strides=2)
    
        branch_3 = conv2d_bn(input_tensor, 256, (1, 1))
        branch_3 = conv2d_bn(branch_3, 256, (3, 3), padding='valid', strides=2)
        
        branch_4 = conv2d_bn(input_tensor, 256, (1, 1))
        branch_4 = conv2d_bn(branch_4, 256, (3, 3))
        branch_4 = conv2d_bn(branch_4, 256, (3, 3), padding='valid', strides=2)
    
        filter_concat = Concatenate(name=name)([branch_1, branch_2, branch_3, branch_4]) if name else Concatenate()([branch_1, branch_2, branch_3, branch_4])

    elif version == 'Inception-ResNet-v2':
        branch_1 = MaxPooling2D((3, 3), padding='valid', strides=2)(input_tensor)
    
        branch_2 = conv2d_bn(input_tensor, 256, (1, 1))
        branch_2 = conv2d_bn(branch_2, 384, (3, 3), padding='valid', strides=2)
    
        branch_3 = conv2d_bn(input_tensor, 256, (1, 1))
        branch_3 = conv2d_bn(branch_3, 288, (3, 3), padding='valid', strides=2)
        
        branch_4 = conv2d_bn(input_tensor, 256, (1, 1))
        branch_4 = conv2d_bn(branch_4, 288, (3, 3))
        branch_4 = conv2d_bn(branch_4, 320, (3, 3), padding='valid', strides=2)
    
        filter_concat = Concatenate(name=name)([branch_1, branch_2, branch_3, branch_4]) if name else Concatenate()([branch_1, branch_2, branch_3, branch_4])
    
    else:
        return None # Kill ^^
    
    return filter_concat


def Inception_ResNet_A(input_tensor, scale=0.1, version=None, name=None):   
    if version == 'Inception-ResNet-v1':
        branch_1 = conv2d_bn(input_tensor, 32, (1, 1))
    
        branch_2 = conv2d_bn(input_tensor, 32, (1, 1))
        branch_2 = conv2d_bn(branch_2, 32, (3, 3))
        
        branch_3 = conv2d_bn(input_tensor, 32, (1, 1))
        branch_3 = conv2d_bn(branch_3, 32, (3, 3))
        branch_3 = conv2d_bn(branch_3, 32, (3, 3))
        
        branches = Concatenate()([branch_1, branch_2, branch_3])
        Inception = conv2d_bn(branches, 256, (1, 1), activation=None)
    
    elif version == 'Inception-ResNet-v2':
        branch_1 = conv2d_bn(input_tensor, 32, (1, 1))
    
        branch_2 = conv2d_bn(input_tensor, 32, (1, 1))
        branch_2 = conv2d_bn(branch_2, 32, (3, 3))
        
        branch_3 = conv2d_bn(input_tensor, 32, (1, 1))
        branch_3 = conv2d_bn(branch_3, 48, (3, 3))
        branch_3 = conv2d_bn(branch_3, 64, (3, 3))
        
        branches = Concatenate()([branch_1, branch_2, branch_3])
        Inception = conv2d_bn(branches, 384, (1, 1), activation=None)
    
    else:
        return None # Kill ^^
    
    scaled_activation = Scaling_Residual(Inception, scale=scale)
    
    residual_connection = Add(name=name)([input_tensor, scaled_activation]) if name else Add()([input_tensor, scaled_activation])
    
    return residual_connection


def Inception_ResNet_B(input_tensor, scale=0.1, version=None, name=None):
    if version == 'Inception-ResNet-v1':
        branch_1 = conv2d_bn(input_tensor, 128, (1, 1))
        
        branch_2 = conv2d_bn(input_tensor, 128, (1, 1))
        branch_2 = conv2d_bn(branch_2, 128, (1, 7))
        branch_2 = conv2d_bn(branch_2, 128, (7, 1))
        
        branches = Concatenate()([branch_1, branch_2])
        Inception = conv2d_bn(branches, 896, (1, 1), activation=None)
    
    elif version == 'Inception-ResNet-v2':
        branch_1 = conv2d_bn(input_tensor, 192, (1, 1))
        
        branch_2 = conv2d_bn(input_tensor, 128, (1, 1))
        branch_2 = conv2d_bn(branch_2, 160, (1, 7))
        branch_2 = conv2d_bn(branch_2, 192, (7, 1))
        
        branches = Concatenate()([branch_1, branch_2])
        Inception = conv2d_bn(branches, 1152, (1, 1), activation=None) # Fig.17 is wrong
    
    else:
        return None # Kill ^^
    
    scaled_activation = Scaling_Residual(Inception, scale=scale)
    
    residual_connection = Add(name=name)([input_tensor, scaled_activation]) if name else Add()([input_tensor, scaled_activation])
    
    return residual_connection


def Inception_ResNet_C(input_tensor, scale=0.1, version=None, name=None):    
    if version == 'Inception-ResNet-v1':
        branch_1 = conv2d_bn(input_tensor, 192, (1, 1))
        
        branch_2 = conv2d_bn(input_tensor, 192, (1, 1))
        branch_2 = conv2d_bn(branch_2, 192, (1, 3))
        branch_2 = conv2d_bn(branch_2, 192, (3, 1))
        
        branches = Concatenate()([branch_1, branch_2])
        Inception = conv2d_bn(branches, 1792, (1, 1), activation=None)
    
    elif version == 'Inception-ResNet-v2':
        branch_1 = conv2d_bn(input_tensor, 192, (1, 1))
        
        branch_2 = conv2d_bn(input_tensor, 192, (1, 1))
        branch_2 = conv2d_bn(branch_2, 224, (1, 3))
        branch_2 = conv2d_bn(branch_2, 256, (3, 1))
        
        branches = Concatenate()([branch_1, branch_2])
        Inception = conv2d_bn(branches, 2144, (1, 1), activation=None) # Fig.19 is wrong
    
    else:
        return None # Kill ^^
    
    scaled_activation = Scaling_Residual(Inception, scale=scale)
    
    residual_connection = Add(name=name)([input_tensor, scaled_activation]) if name else Add()([input_tensor, scaled_activation])
    
    return residual_connection


# SENet
def SE_block(input_tensor, reduction_ratio=16):
    ch_input = K.int_shape(input_tensor)[-1]
    ch_reduced = ch_input//reduction_ratio
    
    # Squeeze
    x = GlobalAveragePooling2D()(input_tensor) # Eqn.2
    
    # Excitation
    x = Dense(ch_reduced, kernel_initializer='he_normal', activation='relu', use_bias=False)(x) # Eqn.3
    x = Dense(ch_input, kernel_initializer='he_normal', activation='sigmoid', use_bias=False)(x) # Eqn.3
    
    x = Reshape( (1, 1, ch_input) )(x)
    x = Multiply()([input_tensor, x]) # Eqn.4
    
    return x


reduction_ratio = 4

# Inception_ResNet + SENet 네트워크 구성
def Inception_ResNet_SENet(model_input, version='Inception-ResNet-v2', classes=1000):    

    x = Stem(model_input, version=version, name='Stem')
    # Inception-ResNet-v1 : (299, 299, 3) -> (35, 35, 256)
    # Inception-ResNet-v2 : (299, 299, 3) -> (35, 35, 384)
    
    for i in range(5):
        x = Inception_ResNet_A(x, scale=0.17, version=version, name='Inception-ResNet-A-'+str(i+1))
        # Inception-ResNet-v1 : (35, 35, 256)
        # Inception-ResNet-v2 : (35, 35, 384)
        x = SE_block(x, reduction_ratio)
        
    x = Reduction_A(x, version=version, name='Reduction-A')
    # Inception-ResNet-v1 : (35, 35, 256) -> (17, 17, 896)
    # Inception-ResNet-v2 : (35, 35, 384) -> (17, 17, 1152)
    
    for i in range(10):
        x = Inception_ResNet_B(x, scale=0.1, version=version, name='Inception-ResNet-B-'+str(i+1))
        # Inception-ResNet-v1 : (17, 17, 896)
        # Inception-ResNet-v2 : (17, 17, 1152)
        x = SE_block(x, reduction_ratio)

    x = Reduction_B(x, version=version, name='Reduction-B')
    # Inception-ResNet-v1 : (17, 17, 896) -> (8, 8, 1792)
    # Inception-ResNet-v2 : (17, 17, 1152) -> (8, 8, 2144)

    x = SE_block(x, reduction_ratio)
    
    for i in range(5):
        x = Inception_ResNet_C(x, scale=0.2, version=version, name='Inception-ResNet-C-'+str(i+1))
        # Inception-ResNet-v1 : (8, 8, 1792)
        # Inception-ResNet-v2 : (8, 8, 2144)
        x = SE_block(x, reduction_ratio)
    
    x = GlobalAveragePooling2D()(x)
    # Inception-ResNet-v1 : (1792)
    # Inception-ResNet-v2 : (2144)
    
    # x = Dropout(0.8)(x)
    x = Dropout(0.2)(x)
    
    model_output = Dense(classes, activation='softmax', name='output')(x)

    model = Model(model_input, model_output, name=version)
    
    return model
