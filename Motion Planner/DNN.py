import keras
from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout
from keras.models import clone_model
from keras.layers import initializers

"""
This class represents a General Purpose Deep Neural Network creates the desired neural network architecture based on
user input for the configuration of layers and nodes
 
Author: Jeet
Date Modified: 04/11/2018
"""


class DNN:

    def __init__(self):
        self.model = None

    def create(self, input_size, output_size, dense_layers, activations, loss_function, optimizer_name='SGD',
               learning_rate=None, decay=0.0, momentum=None, dropouts=None):

        """
        This function creates the network architecture

        :param input_size: size of the input features
        :param output_size: size of the output
        :param dense_layers: 1d numpy array for number of nodes in each Dense layers
        :param activation: A list where the first element specifies the activation to use in hidden layers
                           and 2nd element(if exist) specifies the activation to use in the output layer
        :param optimizer_name: name of the optimizer to use
        :param learning_rate: Initial learning rate
        :param decay: decay factor for learning rate
        :param momentum: to accelerate the optimizer towards the relevant direction
        :dropouts : Either None or 1D array the dropouts to be applied to each layer
        :return:
        """
        input_layer = Input(shape=input_size)

        # To kee[ track the last inserted layer

        last_layer = input_layer
        # Iterate to create the layers
        for i in range(dense_layers.shape[0]):
            last_layer = Dense(dense_layers[i], kernel_initializer='glorot_uniform',
                               activation=activations[0])(last_layer)
            if dropouts is not None and dropouts[i] > 0:
                last_layer = Dropout(dropouts[i])(last_layer)

        if len(activations) == 2:
            output_layer = Dense(output_size, activations[1])(last_layer)
        else:
            output_layer = Dense(output_size)(last_layer)
        self.model = Model(inputs=input_layer, outputs=output_layer)

        optimizer = None
        if optimizer_name == 'SGD':
            if learning_rate is None:
                learning_rate = 0.01
                # raise a warning in case SGd is requested but no learnig rate is provided
                raise Warning("Using the default learning rate", learning_rate)
            if momentum is None:
                optimizer = keras.optimizers.SGD(lr=learning_rate, decay=decay)
            else:
                optimizer = keras.optimizers.SGD(lr=learning_rate, decay=decay, momentum=momentum)
        elif optimizer_name == 'RMSProp':
            optimizer = keras.optimizers.RMSprop()
        elif optimizer_name == 'Adagrad':
            optimizer = keras.optimizers.Adagrad()
        elif optimizer_name == 'Adadelta':
            optimizer = keras.optimizers.Adadelta()
        elif optimizer_name == 'Adam':
            optimizer = keras.optimizers.Adam(0.0001)
        elif optimizer_name == 'Adamax':
            optimizer = keras.optimizers.Adamax()
        elif optimizer_name == 'Nadam':
            optimizer = keras.optimizers.Nadam()

        self.model.compile(loss=loss_function, optimizer=optimizer)

    def train(self, X_train, Y_train, epochs, batch_size=64, X_test=None, Y_test=None, verbose=1):
        """
        This function fits the model to the data

        :param X_train: Features for the training data
        :param Y_train: labels for the training data
        :param X_test: Features for the test data
        :param Y_test: labels for the test data
        :param epochs: Number of epochs to train
        :param batch_size: Number of training example to process in each batch
        :param verbose: value to indicate the printing the training outputs or not
        :return:
        """
        if X_test is None:
            self.model.fit(X_train, Y_train, batch_size=batch_size, verbose=verbose)
        else:
            self.model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=batch_size, verbose=verbose)

    def predict(self, X, batch_size = 256):
        """
        This function predicts the out for the given input
        :param X: Input features
        :param batch_size: Number of inputs to predict in each batch
        :return:
        """
        return self.model.predict(X)

    def save(self, name="model.h5"):
        """
        This function saves teh model to a file
        :param name: name of the file
        :return:
        """
        self.model.save(name)

    def load(self, filename):
        """
        This methods loads the model from the file
        :param filename:
        :return:
        """
        self.model = load_model(filename)

    def copy(self):
        """
        this function creates a copy of the model
        :return:
        """
        model_clone = clone_model(self.model)
        model_clone.set_weights(self.model.get_weights())
        return model_clone
