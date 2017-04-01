import tensorflow as tf
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers

class SharedBias(Layer):
    
    def __init__(self, filters, bias_initializer='zeros', **kwargs):
        super(SharedBias, self).__init__(**kwargs)
        self.filters = filters
        self.bias_initializer = initializers.get(bias_initializer)
        self.supports_masking = True

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        print input_shape
        assert input_dim == self.filters
        self.bias = self.add_weight((1,),
                                    initializer=self.bias_initializer,
                                    name='bias',
                                    trainable=True)

        super(SharedBias, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):
        return tf.add(inputs, self.bias)
        # return K.bias_add(inputs, self.bias)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.filters)

    def get_config(self):
        config = {
            'filters': self.filters,
            'bias_initializer': initializers.serialize(self.bias_initializer),
        }
        base_config = super(SharedBias, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))