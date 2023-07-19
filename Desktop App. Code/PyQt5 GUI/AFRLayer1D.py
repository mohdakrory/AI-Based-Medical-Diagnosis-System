from keras.layers import Layer, GlobalAveragePooling1D, GlobalMaxPooling1D, Dense, Reshape
from tensorflow import sigmoid
from tensorflow import multiply 

class AFRLayer1D(Layer):
    def __init__(self, filters, reduction_ratio=16, **kwargs):
        super(AFRLayer1D, self).__init__(**kwargs)
        self.filters = filters
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        self.channel_axis = -1
        self.channel = input_shape[self.channel_axis]
        self.avg_pool = GlobalAveragePooling1D()
        self.max_pool = GlobalMaxPooling1D()
        self.dense1 = Dense(self.channel // self.reduction_ratio,
                                            activation='relu',
                                            kernel_initializer='he_normal',
                                            use_bias=True)
        self.dense2 = Dense(self.channel,
                                            kernel_initializer='he_normal',
                                            use_bias=True)
        super(AFRLayer1D, self).build(input_shape)

    def call(self, inputs):
        avg_pool = self.avg_pool(inputs)
        max_pool = self.max_pool(inputs)
        avg_out = self.dense1(avg_pool)
        max_out = self.dense1(max_pool)
        avg_out = self.dense2(avg_out)
        max_out = self.dense2(max_out)
        channel_wise_out = sigmoid(avg_out + max_out)
        channel_wise_out = Reshape((1, self.channel))(channel_wise_out)
        out = multiply(inputs, channel_wise_out)
        return out

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filters': self.filters,
            'reduction_ratio': self.reduction_ratio,
        })
        return config