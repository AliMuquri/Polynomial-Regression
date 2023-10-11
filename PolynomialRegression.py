import tensorflow as tf
from tensorflow import Module
from tensorflow.keras.models import Model


class PolynomialComponent(Module):
    '''This class  provides the functionality of a polynomial regression '''

    def __init__(self, input_dim ,degree):
        ''' The constructor for the  class where weights and bias are initialized
        Args:
        input_dim : feature dim
        degree: the degree of the polynomial
        Returns:
        class object
        '''
        self.degree = tf.cast(degree, tf.float32)
        self.weights = tf.Variable(initial_value = tf.random.normal(shape=(input_dim,1)))
        self.bias = tf.Variable(initial_value = tf.zeros(shape=(1,1)))

    def __call__(self, inputs):
        ''' Feedforward of the polynomial

        Args:
        inputs: the data 
        Returns:
        output: the computed feedforward value '''

        #tile the inputs with d copies
        x = tf.expand_dims(inputs, axis=0)
        x_shape = tf.ones(tf.rank(inputs),  dtype=tf.int32)
        x_shape = tf.concat([[self.degree], x_shape], axis=0)


        x = tf.tile(x, x_shape)


        powers = tf.tile(tf.expand_dims(tf.range(1.0, self.degree + 1.0, dtype=tf.float32), axis=1), [1, x_shape[1]])


        polynomial_features =  tf.pow(x, powers)

        output = tf.reduce_sum(tf.reduce_sum(tf.matmul(polynomial_features, self.weights) + self.bias, axis=0), axis=1)
        
        return output

class PolynomialRegression(Model):
    ''' The main model class  of the polynomial where the Polynomial functionality is initilized '''
    def __init__(self,input_dim, degree):
        ''' Initializ the class and  the polynomical component  
        Arg:
        input_dim: feature dim
        degree: degree of the polynomial
        Returns:
        class object '''
        super(PolynomialRegression,self).__init__()
        self.polynomial_component = PolynomialComponent(input_dim,degree)

    def call(self, inputs):
        '''the call method which essential calls the only component in the model
        Arg:
        inputs: the data
        Returns:
        output: computed value of the feedforward '''
        output = self.polynomial_component(inputs)
        return output
