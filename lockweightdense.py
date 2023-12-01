import tensorflow as tf
from keras.layers import Dense

class LockedWeightsDense(Dense):
    def __init__(self, units, locked_neurons=None, **kwargs):
        super(LockedWeightsDense, self).__init__(units, **kwargs)
        self.locked_neurons = locked_neurons
    def get_gradients(self, loss, inputs):
        grads = super(LockedWeightsDense, self).get_gradients(loss, inputs)
        print("stupid python very")
        print(self.locked_neurons)
        if self.locked_neurons is not None:
            for neuron_index in self.locked_neurons:
                print("zero: ", tf.constant(0.0, shape=(inputs[0].shape[-1],)))
                grads[0][:, neuron_index].assign(tf.constant(0.0, shape=(inputs[0].shape[-1],)))
        #print(grads[0])

        return grads