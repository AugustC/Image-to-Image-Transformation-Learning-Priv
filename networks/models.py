from keras.layers import *
from keras.models import Model, load_model
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint
from keras.optimizers import RMSprop, Adam, SGD

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

    model = Model(inputs=inp, outputs=output)

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

    model = Model(inputs=inp, outputs=last_conv)

    return model

def SConvNet(window_size):
    inp = Input(shape=(None,None,3))
    n = window_size//2

    conv = Conv2D(8, (3,3), activation='relu', padding='valid')(inp)
    for i in range(1, n):
        conv = Conv2D(2**(i//3 + 3), (3,3), activation='relu', padding='valid')(conv)
    conv = Conv2D(2, (1,1), activation='softmax')(conv)

    model = Model(inputs=inp, outputs=conv)

    return model
