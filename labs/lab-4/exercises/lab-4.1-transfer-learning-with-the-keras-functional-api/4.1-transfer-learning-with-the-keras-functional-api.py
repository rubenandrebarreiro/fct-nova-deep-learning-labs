"""
Lab 4.1 - Transfer Learning with the Keras Functional API

Author:
- Rodrigo Jorge Ribeiro (rj.ribeiro@campus.fct.unl.pt)
- Ruben Andre Barreiro (r.barreiro@campus.fct.unl.pt)

"""

# Import the Libraries and Packages

# Import the Operative System Library as operative_system
import os as operative_system

# Disable all the Debugging Logs from TensorFlow Library
operative_system.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import Keras from the TensorFlow Library
from tensorflow import keras

# Import the Stochastic Gradient Descent (SGD) Optimizer
# from the TensorFlow.Keras.Optimizers Module
from tensorflow.keras.optimizers import SGD

# Import the Sequential from the TensorFlow.Keras.Models Module
from tensorflow.keras.models import Model

# Import the BatchNormalization, Conv2D, Dense, MaxPooling2D,
# Activation, Flatten, Dropout from the TensorFlow.Keras.Layers Module
from tensorflow.keras.layers import Input, BatchNormalization, Conv2D, Dense,\
    MaxPooling2D, Activation, Flatten, Dropout

# Load the Dataset of the Modified NIST (MNIST),
# retrieving the Training and Testing Datasets
((xs_features_training_data, ys_labels_training_data), (xs_features_testing_data, ys_labels_testing_data)) = \
    keras.datasets.mnist.load_data()

# Reshape the xs (Features) of the Training Data to 28 x 28 x 1,
# in order to fit the Convolution Neural Network (CNN)
xs_features_training_data = \
    xs_features_training_data.reshape((xs_features_training_data.shape[0], 28, 28, 1))

# Reshape the xs (Features) of the Testing Data to 28 x 28 x 1,
# in order to fit the Convolution Neural Network (CNN)
xs_features_testing_data = \
    xs_features_testing_data.reshape((xs_features_testing_data.shape[0], 28, 28, 1))

# Normalize the xs (Features) of the Training Dataset
xs_features_training_data = (xs_features_training_data.astype("float32") / 255.0)

# Normalize the xs (Features) of the Testing Dataset
xs_features_testing_data = (xs_features_testing_data.astype("float32") / 255.0)


# Setup of the old Convolution Neural Network (CNN) Model,
# for the Data of the Fashion Modified NIST (Fashion MNIST),
# using the Keras Functional API

# Set the Input for the Data on the Keras Functional API
inputs = Input(shape=(28, 28, 1), name="inputs")

# Call the Convolution 2D Layer, for the Input of the (28 x 28 x 1) Data,
# using the Keras Functional API, with 32 Filters and a Kernel 3x3, Same Padding and
# Input Shape having a Batch Size of 28, with 28 Steps, as also,
# 1 Input Dimension (for one Color Channel - Grayscale Color)
layer = Conv2D(32, (3, 3), padding="same", input_shape=(28, 28, 1))(inputs)

# Call the ReLU (Rectified Linear Unit Function) as Activation Function Layer,
# after the Convolution 2D Layer, using the Keras Functional API
layer = Activation("relu")(layer)

# Call the Convolution 2D Layer, to the (28 x 28 x 1) Data from
# the ReLU (Rectified Linear Unit) Activation Function,
# using the Keras Functional API, with 32 Filters and a Kernel 3x3, Same Padding and
# Input Shape having a Batch Size of 28, with 28 Steps, as also,
# 1 Input Dimension (for one Color Channel - Grayscale Color)
layer = Conv2D(32, (3, 3), padding="same", input_shape=(28, 28, 1))(layer)

# Call the ReLU (Rectified Linear Unit Function) as Activation Function Layer,
# after the Convolution 2D Layer, using the Keras Functional API
layer = Activation("relu")(layer)

# Call the Normalization of the Batch Layer, using the Keras Functional API
layer = BatchNormalization(axis=-1)(layer)

# Add a Max Pooling 2D Sample-Based Discretization Process Layer,
# for the Data of the Convolution Neural Network (CNN),
# with the Data of the old Fashion Modified NIST (Fashion MNIST) Model and a 2x2 Pool
layer = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(layer)

# Call the Dropout Layer, for Regularization of the Convolution Neural Network (CNN),
# with the Data of the old Fashion Modified NIST (Fashion MNIST) Model,
# by using as hyper-parameter, the rate of 25%
# NOTE:
# - Dropout Layer in Convolution Neural Networks is generally, not very useful;
# - Comment/Uncomment, if you want to try it or not;
layer = Dropout(0.25)(layer)

# Call the Convolution 2D Layer, to the (28 x 28 x 1) Data from
# the ReLU (Rectified Linear Unit) Activation Function,
# using the Keras Functional API, with 64 Filters and a Kernel 3x3, Same Padding and
# Input Shape having a Batch Size of 28, with 28 Steps, as also,
# 1 Input Dimension (for one Color Channel - Grayscale Color)
layer = Conv2D(64, (3, 3), padding="same", input_shape=(28, 28, 1))(layer)

# Call the ReLU (Rectified Linear Unit) Function as Activation Function Layer,
# after the Convolution 2D Layer, using the Keras Functional API
layer = Activation("relu")(layer)

# Call the Convolution 2D Layer, to the (28 x 28 x 1) Data from
# the ReLU (Rectified Linear Unit) Activation Function,
# using the Keras Functional API, with 64 Filters and a Kernel 3x3, Same Padding and
# Input Shape having a Batch Size of 28, with 28 Steps, as also,
# 1 Input Dimension (for one Color Channel - Grayscale Color)
layer = Conv2D(64, (3, 3), padding="same", input_shape=(28, 28, 1))(layer)

# Call the ReLU (Rectified Linear Unit) Function as Activation Function Layer,
# after the Convolution 2D Layer, using the Keras Functional API
layer = Activation("relu")(layer)

# Add a Max Pooling 2D Sample-Based Discretization Process Layer,
# for the Data of the Convolution Neural Network (CNN),
# with the Data of the old Fashion Modified NIST (Fashion MNIST) Model and a 2x2 Pool
layer = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(layer)

# Flatten the Data of the Convolution Neural Network (CNN),
# with the Data of the old Fashion Modified NIST (Fashion MNIST) Model
# NOTE:
# - This is needed to flatten the input into a single dimension for the features,
#   which is what the next Dense Layer needs;
features_flattened = Flatten(name="features")(layer)

# Add a Dense Matrix of the Convolution Neural Network (CNN),
# with the Data of the old Fashion Modified NIST (Fashion MNIST) Model and 512 Units
layer = Dense(512)(features_flattened)

# Call the ReLU (Rectified Linear Unit) Function as Activation Function Layer,
# after the Convolution 2D Layer, using the Keras Functional API
layer = Activation("relu")(layer)

# Call the Normalization of the Batch Layer, using the Keras Functional API
layer = BatchNormalization()(layer)

# Call the Dropout Layer, for Regularization of the Convolution Neural Network (CNN),
# with the Data of the old Fashion Modified NIST (Fashion MNIST) Model,
# by using as hyper-parameter, the rate of 50%
# NOTE:
# - Dropout Layer in Convolution Neural Networks is generally, not very useful;
# - Comment/Uncomment, if you want to try it or not;
layer = Dropout(0.5)(layer)

# Flatten the Data of the Convolution Neural Network (CNN),
# with the Data of the old Fashion Modified NIST (Fashion MNIST) Model
# NOTE:
# - This is needed to flatten the input into a single dimension for the features,
#   which is what the next Dense Layer needs;
layer = Flatten()(layer)

# Add a Dense Matrix of the Convolution Neural Network (CNN),
# with the Data of the old Fashion Modified NIST (Fashion MNIST) Model and 10 Units
layer = Dense(10)(layer)

# Call the SoftMax Function as Activation Function Layer,
# after the Convolution 2D Layer, using the Keras Functional API
layer = Activation("softmax")(layer)


# Build the old Model, composed by the previous defined Layers of
# the Convolution Neural Network (CNN), from the Inputs given,
# resulting on the last Layer of the Convolution Neural Network (CNN)
old_model = Model(inputs=inputs, outputs=layer)

# Compile the old Model, using the Mean Squared Error (MSE) as Loss/Error Function
old_model.compile(optimizer=SGD(), loss="mse")

# Load the Weights from the old Model,
# for the Fashion Modified NIST (Fashion MNIST), in the H5 format
old_model.load_weights("fashion_mnist_model.h5")

# For each Layer of the old Model, for the Fashion Modified NIST (Fashion MNIST)
for layer in old_model.layers:

    # Disable the Training for all the Layer of the old Model for
    # the Fashion Modified NIST (Fashion MNIST)
    layer.trainable = False

# Add a Dense Matrix of the Convolution Neural Network (CNN),
# with the Data of the new Modified NIST (MNIST) Model and 10 Units
layer = Dense(512)(old_model.get_layer("features").output)

# Call the ReLU (Rectified Linear Unit) Function as Activation Function Layer,
# after the Dense Layer, using the Keras Functional API
layer = Activation("relu")(layer)

# Call the Batch Normalization Layer, to normalize the previous Layer,
# by re-centering and re-scaling the Data of the Convolution Neural Network (CNN),
# with the Data of the new Modified NIST (MNIST) Model and
# making the Convolution Neural Network (CNN) faster and more stable
layer = BatchNormalization()(layer)

# Call the Dropout Layer, for Regularization of the Convolution Neural Network (CNN),
# with the Data of the new Modified NIST (MNIST) Model,
# by using as hyper-parameter, the rate of 50%
# NOTE:
# - Dropout Layer in Convolution Neural Networks is generally, not very useful;
# - Comment/Uncomment, if you want to try it or not;
layer = Dropout(0.5)(layer)

# Add a Dense Matrix of the Convolution Neural Network (CNN),
# with the Data of the new Modified NIST (MNIST) Model and 10 Units
layer = Dense(10)(layer)

# Call the Softmax Function as Activation Function Layer,
# after the Dense Layer, using the Keras Functional API
layer = Activation("softmax")(layer)

# Build the new Model, composed by the previous defined Layers of
# the Convolution Neural Network (CNN), from the Inputs given,
# resulting on the last Layer of the Convolution Neural Network (CNN)
new_model = Model(inputs=old_model.get_layer("inputs").output, outputs=layer)


stochastic_gradient_descent_optimiser = SGD(lr=1e-2, momentum=0.9)

new_model.compile(loss="categorical_crossentropy",
                  optimizer=stochastic_gradient_descent_optimiser,
                  metrics=["accuracy"])

new_model.summary()

NUM_EPOCHS = 5
BATCH_SIZE = 512

HISTORY_MODEL_TRAINED = new_model.fit(xs_features_training_data, ys_labels_training_data,
                                      validation_data=(xs_features_testing_data, ys_labels_testing_data),
                                      batch_size=BATCH_SIZE, epochs=NUM_EPOCHS)
