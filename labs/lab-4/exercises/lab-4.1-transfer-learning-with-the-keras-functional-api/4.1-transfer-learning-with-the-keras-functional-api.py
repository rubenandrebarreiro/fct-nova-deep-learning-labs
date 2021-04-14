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

# Set one-hot encode for the ys (Labels) of the Training Set
ys_labels_training_data = keras.utils.to_categorical(ys_labels_training_data, 10)

# Set one-hot encode for the ys (Labels) of the Testing Set
ys_labels_testing_data = keras.utils.to_categorical(ys_labels_testing_data, 10)


# Create the old Convolution Neural Network (CNN) Model
def create_convolution_neural_network_old_model():

    # Setup of the old Convolution Neural Network (CNN) Model,
    # for the Data of the Fashion Modified NIST (Fashion MNIST),
    # using the Keras Functional API

    # Set the Input for the Data on the Keras Functional API
    inputs = Input(shape=(28, 28, 1), name="inputs")

    # 1st CONV => RELU => CONV => RELU => POOL layer set

    # Call the Convolution 2D Layer, for the Input of the (28 x 28 x 1) Data,
    # using the Keras Functional API, with 32 Filters and a Kernel 3x3, Same Padding and
    # Input Shape having a Batch Size of 28, with 28 Steps, as also,
    # 1 Input Dimension (for one Color Channel - Grayscale Color)
    convolution_neural_network = Conv2D(32, (3, 3), padding="same", input_shape=(28, 28, 1))(inputs)

    # Call the ReLU (Rectified Linear Unit Function) as Activation Function Layer,
    # after the Convolution 2D Layer, using the Keras Functional API
    convolution_neural_network = Activation("relu")(convolution_neural_network)

    # Call the Normalization of the Batch Layer, using the Keras Functional API
    convolution_neural_network = BatchNormalization(axis=-1)(convolution_neural_network)

    # Call the Convolution 2D Layer, to the (28 x 28 x 1) Data from
    # the ReLU (Rectified Linear Unit) Activation Function,
    # using the Keras Functional API, with 32 Filters and a Kernel 3x3, Same Padding and
    # Input Shape having a Batch Size of 28, with 28 Steps, as also,
    # 1 Input Dimension (for one Color Channel - Grayscale Color)
    convolution_neural_network = Conv2D(32, (3, 3), padding="same")(convolution_neural_network)

    # Call the ReLU (Rectified Linear Unit Function) as Activation Function Layer,
    # after the Convolution 2D Layer, using the Keras Functional API
    convolution_neural_network = Activation("relu")(convolution_neural_network)

    # Call the Normalization of the Batch Layer, using the Keras Functional API
    convolution_neural_network = BatchNormalization(axis=-1)(convolution_neural_network)

    # Add a Max Pooling 2D Sample-Based Discretization Process Layer,
    # for the Data of the Convolution Neural Network (CNN),
    # with the Data of the old Fashion Modified NIST (Fashion MNIST) Model and a 2x2 Pool
    convolution_neural_network = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(convolution_neural_network)

    # Call the Dropout Layer, for Regularization of the Convolution Neural Network (CNN),
    # with the Data of the old Fashion Modified NIST (Fashion MNIST) Model,
    # by using as hyper-parameter, the rate of 25%
    # NOTE:
    # - Dropout Layer in Convolution Neural Networks is generally, not very useful;
    # - Comment/Uncomment, if you want to try it or not;
    convolution_neural_network = Dropout(0.25)(convolution_neural_network)

    # 2nd CONV => RELU => CONV => RELU => POOL layer set

    # Call the Convolution 2D Layer, to the (28 x 28 x 1) Data from
    # the ReLU (Rectified Linear Unit) Activation Function,
    # using the Keras Functional API, with 64 Filters and a Kernel 3x3, Same Padding and
    # Input Shape having a Batch Size of 28, with 28 Steps, as also,
    # 1 Input Dimension (for one Color Channel - Grayscale Color)
    convolution_neural_network = Conv2D(64, (3, 3), padding="same")(convolution_neural_network)

    # Call the ReLU (Rectified Linear Unit) Function as Activation Function Layer,
    # after the Convolution 2D Layer, using the Keras Functional API
    convolution_neural_network = Activation("relu")(convolution_neural_network)

    # Call the Normalization of the Batch Layer, using the Keras Functional API
    convolution_neural_network = BatchNormalization(axis=-1)(convolution_neural_network)

    # Call the Convolution 2D Layer, to the (28 x 28 x 1) Data from
    # the ReLU (Rectified Linear Unit) Activation Function,
    # using the Keras Functional API, with 64 Filters and a Kernel 3x3, Same Padding and
    # Input Shape having a Batch Size of 28, with 28 Steps, as also,
    # 1 Input Dimension (for one Color Channel - Grayscale Color)
    convolution_neural_network = Conv2D(64, (3, 3), padding="same")(convolution_neural_network)

    # Call the ReLU (Rectified Linear Unit) Function as Activation Function Layer,
    # after the Convolution 2D Layer, using the Keras Functional API
    convolution_neural_network = Activation("relu")(convolution_neural_network)

    # Call the Normalization of the Batch Layer, using the Keras Functional API
    convolution_neural_network = BatchNormalization(axis=-1)(convolution_neural_network)

    # Add a Max Pooling 2D Sample-Based Discretization Process Layer,
    # for the Data of the Convolution Neural Network (CNN),
    # with the Data of the old Fashion Modified NIST (Fashion MNIST) Model and a 2x2 Pool
    convolution_neural_network = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(convolution_neural_network)

    # Call the Dropout Layer, for Regularization of the Convolution Neural Network (CNN),
    # with the Data of the old Fashion Modified NIST (Fashion MNIST) Model,
    # by using as hyper-parameter, the rate of 25%
    # NOTE:
    # - Dropout Layer in Convolution Neural Networks is generally, not very useful;
    # - Comment/Uncomment, if you want to try it or not;
    convolution_neural_network = Dropout(0.25)(convolution_neural_network)

    # 1st (and only) set of FC => RELU layers

    # Flatten the Data of the Convolution Neural Network (CNN),
    # with the Data of the old Fashion Modified NIST (Fashion MNIST) Model
    # NOTE:
    # - This is needed to flatten the input into a single dimension for the features,
    #   which is what the next Dense Layer needs;
    features_flattened = Flatten(name="features")(convolution_neural_network)

    # Add a Dense Matrix of the Convolution Neural Network (CNN),
    # with the Data of the old Fashion Modified NIST (Fashion MNIST) Model and 512 Units
    convolution_neural_network = Dense(512)(features_flattened)

    # Call the ReLU (Rectified Linear Unit) Function as Activation Function Layer,
    # after the Convolution 2D Layer, using the Keras Functional API
    convolution_neural_network = Activation("relu")(convolution_neural_network)

    # Call the Normalization of the Batch Layer, using the Keras Functional API
    convolution_neural_network = BatchNormalization()(convolution_neural_network)

    # Call the Dropout Layer, for Regularization of the Convolution Neural Network (CNN),
    # with the Data of the old Fashion Modified NIST (Fashion MNIST) Model,
    # by using as hyper-parameter, the rate of 50%
    # NOTE:
    # - Dropout Layer in Convolution Neural Networks is generally, not very useful;
    # - Comment/Uncomment, if you want to try it or not;
    convolution_neural_network = Dropout(0.5)(convolution_neural_network)

    # SoftMax Classifier

    # Add a Dense Matrix of the Convolution Neural Network (CNN),
    # with the Data of the old Fashion Modified NIST (Fashion MNIST) Model and 10 Units
    convolution_neural_network = Dense(10)(convolution_neural_network)

    # Call the SoftMax Function as Activation Function Layer,
    # after the Convolution 2D Layer, using the Keras Functional API
    convolution_neural_network = Activation("softmax")(convolution_neural_network)

    # Build the old Model, composed by the previous defined Layers of
    # the Convolution Neural Network (CNN), from the Inputs given,
    # resulting on the last Layer of the Convolution Neural Network (CNN)
    convolution_neural_network_old_model = Model(inputs=inputs, outputs=convolution_neural_network)

    # Compile the old Model, using the Mean Squared Error (MSE) as Loss/Error Function
    convolution_neural_network_old_model.compile(optimizer=SGD(), loss="mse")

    # Load the Weights from the old Model,
    # for the Fashion Modified NIST (Fashion MNIST), in the H5 format
    convolution_neural_network_old_model.load_weights("fashion_mnist_model.h5")

    # For each Layer of the old Model, for the Fashion Modified NIST (Fashion MNIST)
    for convolution_neural_network_layer in convolution_neural_network_old_model.layers:

        # Disable the Training for all the Layer of the old Model for
        # the Fashion Modified NIST (Fashion MNIST)
        convolution_neural_network_layer.trainable = False

    # Return the old Convolution Neural Network (CNN) Model
    return convolution_neural_network_old_model


# Create the new Convolution Neural Network (CNN) Model
def create_convolution_neural_network_new_model(convolution_neural_network_old_model):

    # Add a Dense Matrix of the Convolution Neural Network (CNN),
    # with the Data of the new Modified NIST (MNIST) Model and 10 Units
    convolution_neural_network = Dense(512)(convolution_neural_network_old_model.get_layer("features").output)

    # Call the ReLU (Rectified Linear Unit) Function as Activation Function Layer,
    # after the Dense Layer, using the Keras Functional API
    convolution_neural_network = Activation("relu")(convolution_neural_network)

    # Call the Batch Normalization Layer, to normalize the previous Layer,
    # by re-centering and re-scaling the Data of the Convolution Neural Network (CNN),
    # with the Data of the new Modified NIST (MNIST) Model and
    # making the Convolution Neural Network (CNN) faster and more stable
    convolution_neural_network = BatchNormalization()(convolution_neural_network)

    # Call the Dropout Layer, for Regularization of the Convolution Neural Network (CNN),
    # with the Data of the new Modified NIST (MNIST) Model,
    # by using as hyper-parameter, the rate of 50%
    # NOTE:
    # - Dropout Layer in Convolution Neural Networks is generally, not very useful;
    # - Comment/Uncomment, if you want to try it or not;
    convolution_neural_network = Dropout(0.5)(convolution_neural_network)

    # Add a Dense Matrix of the Convolution Neural Network (CNN),
    # with the Data of the new Modified NIST (MNIST) Model and 10 Units
    convolution_neural_network = Dense(10)(convolution_neural_network)

    # Call the Softmax Function as Activation Function Layer,
    # after the Dense Layer, using the Keras Functional API
    convolution_neural_network = Activation("softmax")(convolution_neural_network)

    # Build the new Model, composed by the previous defined Layers of
    # the Convolution Neural Network (CNN), from the Inputs given,
    # resulting on the last Layer of the Convolution Neural Network (CNN)
    convolution_neural_network_new_model = \
        Model(inputs=convolution_neural_network_old_model.get_layer("inputs").output,
              outputs=convolution_neural_network)

    # Return the new Convolution Neural Network (CNN) Model
    return convolution_neural_network_new_model


# Create the old Convolution Neural Network (CNN) Model,
# from the Data of the Fashion Modified NIST (Fashion MNIST)
cnn_old_model = create_convolution_neural_network_old_model()

# Create the new Convolution Neural Network (CNN) Model
cnn_new_model = create_convolution_neural_network_new_model(cnn_old_model)

# Initialise the Stochastic Gradient Descent (SGD) Optimizer,
# with the Learning Rate of 1%, Momentum of 90%
stochastic_gradient_descent_optimiser = SGD(lr=1e-2, momentum=0.9)

# Compile the Convolution Neural Network (CNN) Model,
# with the Categorical Cross Entropy Loss Function and with the Stochastic Gradient Descent (SGD) Optimiser
cnn_new_model.compile(loss="categorical_crossentropy",
                      optimizer=stochastic_gradient_descent_optimiser,
                      metrics=["accuracy"])

# Print the Summary of the Convolution Neural Network (CNN) Model
cnn_new_model.summary()

# Set the number of Epochs
NUM_EPOCHS = 30

# Set the Size of the Batch
BATCH_SIZE = 512

# Train the Convolution Neural Network (CNN) Model
HISTORY_MODEL_TRAINED = cnn_new_model.fit(xs_features_training_data, ys_labels_training_data,
                                          validation_data=(xs_features_testing_data, ys_labels_testing_data),
                                          batch_size=BATCH_SIZE, epochs=NUM_EPOCHS)
