from keras.models import Model
from keras.layers import BatchNormalization
from keras.layers.convolutional import Conv3D, Conv3DTranspose
from keras.layers.pooling import MaxPooling3D
from keras.layers.merge import concatenate


def conv3d_block(input_tensor, n_filters, kernel_size, batchnorm=True):
    # first layer
    x = Conv3D(filters=n_filters, kernel_size=kernel_size,
               activation="relu", kernel_initializer="he_normal")(input_tensor)
    # he_normal --> inicjalizacja wag wartosciami z rozkladu Gaussa z odchyleniem
    # standardowym sqrt(2/N) (wyjasnienie w zeszycie! (str.12)

    if batchnorm:
        x = BatchNormalization()(x)

    # second layer
    x = Conv3D(filters=n_filters, kernel_size=kernel_size,
               activation="relu", kernel_initializer="he_normal")(x)

    if batchnorm:
        x = BatchNormalization()(x)

    return x


def get_unet(input_img, n_filters, batchnorm=True):
    # input_img --> wielkość wejściowego obrazu
    # domyslnie data_format to channels_last

    # contracting path
    c1 = conv3d_block(input_img, n_filters=n_filters * 1, kernel_size=3,
                      data_format="channels_first", batchnorm=batchnorm)
    p1 = MaxPooling3D(pool_size=(2, 2, 2))(c1)

    c2 = conv3d_block(p1, n_filters=n_filters * 2, kernel_size=3,
                      data_format="channels_first", batchnorm=batchnorm)
    p2 = MaxPooling3D((2, 2, 2))(c2)

    c3 = conv3d_block(p2, n_filters=n_filters * 4, kernel_size=3,
                      data_format="channels_first", batchnorm=batchnorm)
    p3 = MaxPooling3D((2, 2, 2))(c3)

    c4 = conv3d_block(p3, n_filters=n_filters * 8, kernel_size=3,
                      data_format="channels_first", batchnorm=batchnorm)
    p4 = MaxPooling3D((2, 2, 2))(c4)

    c5 = conv3d_block(p4, n_filters=n_filters * 16, kernel_size=3,
                      data_format="channels_first", batchnorm=batchnorm)

    # expansive path
    u6 = Conv3DTranspose(n_filters * 8, kernel_size=3, strides=2,
                         data_format="channels_first", batchnorm=batchnorm)(c5)
    u6 = concatenate([u6, c4])
    c6 = conv3d_block(u6, n_filters * 8, kernel_size=3,
                      data_format="channels_first", batchnorm=batchnorm)

    u7 = Conv3DTranspose(n_filters * 4, kernel_size=3, strides=2,
                         data_format="channels_first", batchnorm=batchnorm)(c6)
    u7 = concatenate([u7, c3])
    c7 = conv3d_block(u7, n_filters * 4, kernel_size=3,
                      data_format="channels_first", batchnorm=batchnorm)

    u8 = Conv3DTranspose(n_filters * 2, kernel_size=3, strides=2,
                         data_format="channels_first", batchnorm=batchnorm)(c7)
    u8 = concatenate([u8, c2])
    c8 = conv3d_block(u8, n_filters * 2, kernel_size=3,
                      data_format="channels_first", batchnorm=batchnorm)

    u9 = Conv3DTranspose(n_filters * 1, kernel_size=3, strides=2,
                         data_format="channels_first", batchnorm=batchnorm)(c8)
    u9 = concatenate([u9, c1])
    c9 = conv3d_block(u9, n_filters * 1, kernel_size=3,
                      data_format="channels_first", batchnorm=batchnorm)

    # output - wysegmentowany obraz
    outputs = Conv3D(1, (1, 1), data_format="channels_first", activation='sigmoid')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])

    return model
