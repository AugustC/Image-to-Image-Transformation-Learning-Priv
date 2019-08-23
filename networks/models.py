from keras.layers import *
from keras.models import Model, load_model
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint
from keras.optimizers import RMSprop, Adam, SGD
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from tensorflow.keras import backend as K
import numpy as np

def FCN32(d=0.2):
    inp = Input(shape=(None,None,3))
    # Block 1
    c11 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', kernel_regularizer=l2(5e-4))(inp)
    c12 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', kernel_regularizer=l2(5e-4))(c11)
    p1 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(c12)

    # Block 2
    c21 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', kernel_regularizer=l2(5e-4))(p1)
    c22 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', kernel_regularizer=l2(5e-4))(c21)
    p2 = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(c22)

    output = UpSampling2D(size=(4,4))(p2)
    output = Conv2D(2,(1,1), activation='softmax')(output)

    model = Model(inputs=inp, outputs=output, name='FCN')

    return model

def UNet(n):
    inp = Input(shape=(None,None,3))

    conv = [inp]
    pool = [inp]

    # Contracting path
    for i in range(1, n):
        conv.append(Conv2D(16*2**i, (3,3), activation='relu', data_format='channels_last', padding='same')(pool[i-1]))
        conv[i] = Dropout(0.2)(conv[i])
        conv[i] = Conv2D(16*2**i, (3,3), activation='relu', data_format='channels_last', padding='same')(conv[i])
        pool.append(MaxPooling2D((2,2), data_format='channels_last')(conv[i]))

    conv.append(Conv2D(16*2**n, (3,3), activation='relu', data_format='channels_last', padding='same')(pool[n-1]))
    conv[n] = Conv2D(16*2**n, (3,3), activation='relu', data_format='channels_last', padding='same')(conv[n])
    convUp = conv[n]

    # Expanding path
    for i in range(n-1, 0, -1):
        up = UpSampling2D(size=(2,2))(convUp)
        up = concatenate([up, conv[i]], axis=3)
        convUp = Conv2D(16*2**i, (3,3), activation='relu', data_format='channels_last', padding='same')(up)
        convUp = Dropout(0.2)(convUp)
        convUp = Conv2D(16*2**i, (3,3), activation='relu', data_format='channels_last', padding='same')(convUp)

    last_conv = Conv2D(2, (1,1), activation='softmax', data_format='channels_last', padding='same')(convUp)

    model = Model(inputs=inp, outputs=last_conv, name='UNet')

    return model

def SConvNet(window_size, batch_norm=False, model_name='SConvNet', reg_l2=0, n_channels=3):
    inp = Input(shape=(None,None,n_channels))
    n = window_size//2

    conv = Conv2D(8, (3,3), activation='relu', padding='valid')(inp)
    for i in range(1, n):
        conv = Conv2D(2**(i//3 + 3), (3,3), padding='valid', kernel_regularizer=None)(conv)
        if batch_norm:
            conv = BatchNormalization(axis=3)(conv)
        conv = Activation('relu')(conv)
    conv = Conv2D(2, (1,1), activation='softmax')(conv)

    model = Model(inputs=inp, outputs=conv, name=model_name)

    return model

def Discriminator(l_window, s_window, n_channels=3):
    o_img = Input(shape=(None, None, n_channels), name='original_img')
    g_img = Input(shape=(None, None, 2), name='generate_img')

    crop = l_window - s_window
    crop_img = Cropping2D(crop//2, name='crop')(o_img)

    inp = Concatenate(axis=-1, name='concat')([crop_img, g_img])
    n_conv = int(np.log2(s_window))

    conv = Conv2D(8, (3,3), strides=2, padding='same', name='conv0', kernel_regularizer=l2(0.01))(inp)
    conv = LeakyReLU(alpha=0.2, name='relu0')(conv)
    conv = Dropout(0.4)(conv)
    for i in range(1,n_conv):
        conv = Conv2D(2**(i+3), (3,3), strides=2, padding='same', name='conv'+str(i), kernel_regularizer=l2(0.01))(conv)
        conv = Dropout(0.4)(conv)
        conv = BatchNormalization(axis=3, name='bn'+str(i))(conv)
        conv = LeakyReLU(alpha=0.2, name='relu'+str(i))(conv)

    out = Dense(1, activation='sigmoid', name='output')(conv)

    D = Model(inputs=[o_img, g_img], outputs=out, name='D')

    return D

def DCGAN(l_window, s_window, G=None, D=None, n_channels=3):
    if not G or not D:
        raise Exception('generator and discriminator must be defined')

    inp_g = Input(shape=(None, None, n_channels))

    gen_img = G(inp_g)
    d_out = D([inp_g, gen_img])

    DCGAN = Model(inputs=inp_g, output=[gen_img, d_out], name='DCGAN')

    return DCGAN
