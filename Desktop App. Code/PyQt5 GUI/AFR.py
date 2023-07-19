from keras import backend as K
from keras.layers import Layer, Conv2D, Activation, GlobalAveragePooling2D, Reshape, Multiply, Add

def config_afr(input_shape, reduction_ratio=16):
    num_channels = input_shape[-1]
    num_reduced_filters = max(num_channels // reduction_ratio, 1)
    return num_channels, num_reduced_filters



class AFR(Layer):
    def __init__(self, reduction_ratio=16, **kwargs):
        super(AFR, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        self.num_channels, self.num_reduced_filters = config_afr(input_shape, self.reduction_ratio)
        self.fc1 = Conv2D(self.num_reduced_filters, kernel_size=(1, 1), strides=(1, 1), padding='same')
        self.relu1 = Activation('relu')
        self.fc2 = Conv2D(self.num_channels, kernel_size=(1, 1), strides=(1, 1), padding='same')
        self.sigmoid = Activation('sigmoid')
        super(AFR, self).build(input_shape)

    def call(self, inputs):
        x = GlobalAveragePooling2D()(inputs)
        x = Reshape((1, 1, self.num_channels))(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        x = Multiply()([inputs, x])
        x = Add()([inputs, x])
        return x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(AFR, self).get_config()
        config['reduction_ratio'] = self.reduction_ratio
        return config