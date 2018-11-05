from keras.models import Model
from keras.layers import BatchNormalization
from keras.layers.convolutional import Conv3D, Conv3DTranspose
from keras.layers.pooling import MaxPooling3D
from keras.layers.merge import concatenate


def conv3d_block(input_tensor, n_filters, kernel_size, batchnorm=True):
    # domyslnie data_format to channels_last

    # first layer
    x = Conv3D(filters=n_filters, kernel_size=kernel_size, data_format="channels_first",
               activation="relu", kernel_initializer="he_normal")(input_tensor)
    # he_normal --> inicjalizacja wag wartosciami z rozkladu Gaussa z odchyleniem
    # standardowym sqrt(2/N) (wyjasnienie w zeszycie! (str.12)

    if batchnorm:
        x = BatchNormalization()(x)

    print("Number of filters: %d" % n_filters)
    print("Shape of first conv layer: %s" % x.shape)

    # second layer
    x = Conv3D(filters=n_filters, kernel_size=kernel_size, data_format="channels_first",
               activation="relu", kernel_initializer="he_normal")(x)

    if batchnorm:
        x = BatchNormalization()(x)

    print("Shape of second conv layer: %s" % x.shape)

    return x


def get_unet(input_img, n_filters, batchnorm=True):
    # input_img --> wielkość wejściowego obrazu

    # contracting path
    print("c1:")
    c1 = conv3d_block(input_img, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling3D(pool_size=(2, 2, 2), data_format="channels_first")(c1)
    print("After max pooling: %s" % p1.shape)

    print("c2:")
    c2 = conv3d_block(p1, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling3D((2, 2, 2), data_format="channels_first")(c2)
    print("After max pooling: %s" % p2.shape)

    print("c3:")
    c3 = conv3d_block(p2, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling3D((2, 2, 2), data_format="channels_first")(c3)
    print("After max pooling: %s" % p3.shape)

    print("c4:")
    c4 = conv3d_block(p3, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)
    # p4 = MaxPooling3D((2, 2, 2), data_format="channels_first")(c4)
    # print("After max pooling: %s" % p4.shape)
    #
    # print("c5:")
    # c5 = conv3d_block(p4, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm)

    # expansive path
    # print("u6:")
    # u6 = Conv3DTranspose(n_filters * 8, kernel_size=3, strides=2,
    #                      data_format="channels_first")(c5)
    # print(u6.shape)
    # u6 = concatenate([u6, c4])
    # c6 = conv3d_block(u6, n_filters * 8, kernel_size=3, batchnorm=batchnorm)

    print("u7:")
    u7 = Conv3DTranspose(n_filters * 4, kernel_size=3, strides=2, padding="same",
                         data_format="channels_first")(c4)
    print(u7.shape)
    u7 = concatenate([u7, c3])
    print("Po concatenate: %s" % u7.shape)
    c7 = conv3d_block(u7, n_filters * 4, kernel_size=3, batchnorm=batchnorm)

    print("u8:")
    u8 = Conv3DTranspose(n_filters * 2, kernel_size=3, strides=2, padding="same",
                         data_format="channels_first")(c7)
    print(u8.shape)
    u8 = concatenate([u8, c2])
    print("Po concatenate: %s" % u8.shape)
    c8 = conv3d_block(u8, n_filters * 2, kernel_size=3, batchnorm=batchnorm)

    print("u9:")
    u9 = Conv3DTranspose(n_filters * 1, kernel_size=3, strides=2, padding="same",
                         data_format="channels_first")(c8)
    print(u9.shape)
    u9 = concatenate([u9, c1])
    print("Po concatenate: %s" % u9.shape)
    c9 = conv3d_block(u9, n_filters * 1, kernel_size=3, batchnorm=batchnorm)

    # output - wysegmentowany obraz
    outputs = Conv3D(1, (1, 1), data_format="channels_first", activation='sigmoid')(c9)
    print("output:")
    print(outputs.shape)
    model = Model(inputs=[input_img], outputs=[outputs])

    return model
