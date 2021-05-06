"""
Lab 7.3 - Distinguish Odd and Even Digits in MNIST

Author:
- Rodrigo Jorge Ribeiro (rj.ribeiro@campus.fct.unl.pt)
- Ruben Andre Barreiro (r.barreiro@campus.fct.unl.pt)

"""

# Import the Libraries and Packages

# Import the Operative System Library as operative_system
import os as operative_system

# Disable all the Debugging Logs from TensorFlow Library
operative_system.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import the Multiprocessing Library
import multiprocessing

# Import the TensorFlow Library
import tensorflow

# Import Keras from the TensorFlow Library
from tensorflow import keras

# Import the Stochastic Gradient Descent (SGD) Optimizer
# from the TensorFlow.Keras.Optimizers Module
from tensorflow.keras.optimizers import SGD

# Import the Backend Module from the TensorFlow.Python.Keras Python's Module
from tensorflow.python.keras import backend as keras_backend

# Import the Sequential from the TensorFlow.Keras.Models Module
from tensorflow.keras.models import Sequential

# Import the BatchNormalization, Conv2D, MaxPooling2D from the TensorFlow.Keras.Layers Module
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D

# Import the Activation, Flatten, Dropout and Dense from the TensorFlow.Keras.Layers Module
from tensorflow.keras.layers import Activation, Flatten, Dropout, Dense


# Constants

# The boolean flag, to keep information about
# the use of High-Performance Computing (with CPUs and GPUs)
TENSORFLOW_KERAS_HPC_BACKEND_SESSION = True

# The Number of CPU's Processors/Cores
NUM_CPU_PROCESSORS_CORES = multiprocessing.cpu_count()

# The Number of GPU's Devices
NUM_GPU_DEVICES = len(tensorflow.config.list_physical_devices('GPU'))

# The Learning Rate for the Stochastic Gradient Descent (SGD) Optimizer of
# the Convolution Neural Network (CNN), as 1%
INITIAL_LEARNING_RATE = 0.01

# The Number of Epochs for the Stochastic Gradient Descent (SGD) Optimizer of
# the Convolution Neural Network (CNN), as 25
NUM_EPOCHS = 25

# The Size of the Batch for the the Convolution Neural Network (CNN), as 128
BATCH_SIZE = 128


# If the boolean flag, to keep information about
# the use of High-Performance Computing (with CPUs and GPUs) is set to True
if TENSORFLOW_KERAS_HPC_BACKEND_SESSION:

    # Print the information about if the Model will be executed,
    # using High-Performance Computing (with CPUs and GPUs)
    print('\n')
    print('It will be used High-Performance Computing (with CPUs and GPUs):')
    print(' - Num. CPUS: ', NUM_CPU_PROCESSORS_CORES)
    print(' - Num. GPUS: ', NUM_GPU_DEVICES)
    print('\n')

    # Set the Configuration's Proto, for the given number of Devices (CPUs and GPUs)
    configuration_proto = \
        tensorflow.compat.v1.ConfigProto(device_count={'CPU': NUM_CPU_PROCESSORS_CORES,
                                                       'GPU': NUM_GPU_DEVICES})

    # Configure a TensorFlow Session for High-Performance Computing (with CPUs and GPUs)
    session = tensorflow.compat.v1.Session(config=configuration_proto)

    # Set the current Keras' Backend, with previously configured
    # TensorFlow Session for High-Performance Computing (with CPUs and GPUs)
    keras_backend.set_session(session)


# Load the Dataset of the Modified NIST (MNIST),
# retrieving the Training and Testing Datasets
((xs_features_training_data, ys_labels_training_data), (xs_features_testing_data, ys_labels_testing_data)) = \
    keras.datasets.mnist.load_data()

# Reshape the xs (Features) of the Training Data to 28 x 28 x 1,
# in order to fit the Convolution Neural Network (CNN)
xs_features_training_data = xs_features_training_data.reshape((xs_features_training_data.shape[0], 28, 28, 1))

# Reshape the xs (Features) of the Testing Data to 28 x 28 x 1,
# in order to fit the Convolution Neural Network (CNN)
xs_features_testing_data = xs_features_testing_data.reshape((xs_features_testing_data.shape[0], 28, 28, 1))

# Normalize the xs (Features) of the Training Dataset
xs_features_training_data_normalized = (xs_features_training_data.astype("float32") / 255.0)

# Normalize the xs (Features) of the Testing Dataset
xs_features_testing_data_normalized = (xs_features_testing_data.astype("float32") / 255.0)


# Relabel the ys (Labels) of the Datasets to just two classes (Odd and Even)
def relabel_for_odd_and_even_classes(ys_labels):

    # For each original ys (Labels)
    for index, example in enumerate(ys_labels):

        # If it is an Even Number
        if example % 2 == 0:

            ys_labels[index] = 0

        # If it is an Odd Number
        else:

            ys_labels[index] = 1

    # Return the new ys (Labels)
    return ys_labels


# Create the Convolution Neural Network (CNN) Model
def create_convolution_neural_network_model():

    # Create the Sequential Model for the Convolution Neural Network (CNN)
    convolution_neural_network_model = Sequential()

    # 1st CONV => RELU => CONV => RELU => POOL layer set

    # Add a first Convolution 2D Matrix, for the Input Data of
    # the Modified NIST (MNIST),
    # with 32 Filters and a Kernel 3x3, Same Padding and
    # an Input Shape having a Batch Size of 28, with 28 Steps, as also,
    # 1 Input Dimension (for one Color Channel - Grayscale Color)
    convolution_neural_network_model.add(Conv2D(32, (3, 3), padding="same", input_shape=(28, 28, 1)))

    # Add the Rectified Linear Unit (ReLU) as Activation Function Layer,
    # for the Data of the Convolution Neural Network (CNN),
    # with the Data of the Modified NIST (MNIST)
    convolution_neural_network_model.add(Activation("relu"))

    # Add the Batch Normalization Layer, to normalize the previous Layer,
    # by re-centering and re-scaling the Data of the Convolution Neural Network (CNN),
    # with the Data of the Modified NIST (MNIST) and
    # making the Convolution Neural Network (CNN) faster and more stable
    convolution_neural_network_model.add(BatchNormalization(axis=-1))

    # Add a second Convolution 2D Matrix, for the previous Data of
    # the Modified NIST (MNIST),
    # with 32 Filters and a Kernel 3x3, Same Padding and
    # an Input Shape having a Batch Size of 28, with 28 Steps, as also,
    # 1 Input Dimension (for one Color Channel - Grayscale Color)
    convolution_neural_network_model.add(Conv2D(32, (3, 3), padding="same"))

    # Add the Rectified Linear Unit (ReLU) as Activation Function Layer,
    # for the Data of the Convolution Neural Network (CNN),
    # with the Data of the Modified NIST (MNIST)
    convolution_neural_network_model.add(Activation("relu"))

    # Add the Batch Normalization Layer, to normalize the previous Layer,
    # by re-centering and re-scaling the Data of the Convolution Neural Network (CNN),
    # with the Data of the Modified NIST (MNIST) and
    # making the Convolution Neural Network (CNN) faster and more stable
    convolution_neural_network_model.add(BatchNormalization(axis=-1))

    # Add a Max Pooling 2D Sample-Based Discretization Process Layer,
    # for the Data of the Convolution Neural Network (CNN),
    # with the Data of the Modified NIST (MNIST) and a 2x2 Pool
    convolution_neural_network_model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Add the Dropout Layer, for Regularization of the Convolution Neural Network (CNN),
    # with the Data of the Modified NIST (MNIST),
    # by using as hyper-parameter, the rate of 25%
    # NOTE:
    # - Dropout Layer in Convolution Neural Networks is generally, not very useful;
    # - Comment/Uncomment, if you want to try it or not;
    convolution_neural_network_model.add(Dropout(0.25))

    # 2nd CONV => RELU => CONV => RELU => POOL layer set

    # Add a first Convolution 2D Matrix, for the Input Data of
    # the Modified NIST (MNIST),
    # with 64 Filters and a Kernel 3x3, Same Padding and
    # an Input Shape having a Batch Size of 28, with 28 Steps, as also,
    # 1 Input Dimension (for one Color Channel - Grayscale Color)
    convolution_neural_network_model.add(Conv2D(64, (3, 3), padding="same"))

    # Add the Rectified Linear Unit (ReLU) as Activation Function Layer,
    # for the Data of the Convolution Neural Network (CNN),
    # with the Data of the Modified NIST (MNIST)
    convolution_neural_network_model.add(Activation("relu"))

    # Add the Batch Normalization Layer, to normalize the previous Layer,
    # by re-centering and re-scaling the Data of the Convolution Neural Network (CNN),
    # with the Data of the Modified NIST (MNIST) and
    # making the Convolution Neural Network (CNN) faster and more stable
    convolution_neural_network_model.add(BatchNormalization(axis=-1))

    # Add a second Convolution 2D Matrix, for the previous Data of
    # the Modified NIST (MNIST),
    # with 64 Filters and a Kernel 3x3, Same Padding and
    # an Input Shape having a Batch Size of 28, with 28 Steps, as also,
    # 1 Input Dimension (for one Color Channel - Grayscale Color)
    convolution_neural_network_model.add(Conv2D(64, (3, 3), padding="same"))

    # Add the Rectified Linear Unit (ReLU) as Activation Function Layer,
    # for the Data of the Convolution Neural Network (CNN),
    # with the Data of the Modified NIST (MNIST)
    convolution_neural_network_model.add(Activation("relu"))

    # Add the Batch Normalization Layer, to normalize the previous Layer,
    # by re-centering and re-scaling the Data of the Convolution Neural Network (CNN),
    # with the Data of the Modified NIST (MNIST) and
    # making the Convolution Neural Network (CNN) faster and more stable
    convolution_neural_network_model.add(BatchNormalization(axis=-1))

    # Add a Max Pooling 2D Sample-Based Discretization Process Layer,
    # for the Data of the Convolution Neural Network (CNN),
    # with the Data of the Modified NIST (MNIST) and a 2x2 Pool
    convolution_neural_network_model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Add the Dropout Layer, for Regularization of the Convolution Neural Network (CNN),
    # with the Data of the Modified NIST (MNIST),
    # by using as hyper-parameter, the rate of 25%
    # NOTE:
    # - Dropout Layer in Convolution Neural Networks is generally, not very useful;
    # - Comment/Uncomment, if you want to try it or not;
    convolution_neural_network_model.add(Dropout(0.25))

    # 1st (and only) set of FC => RELU layers

    # Flatten the Data of the Convolution Neural Network (CNN),
    # with the Data of the Modified NIST (MNIST)
    # NOTE:
    # - This is needed to flatten the input into a single dimension for the features,
    #   which is what the next Dense Layer needs;
    convolution_neural_network_model.add(Flatten())

    # Add a Dense Matrix of the Convolution Neural Network (CNN),
    # with the Data of the Modified NIST (MNIST) and 512 Units
    convolution_neural_network_model.add(Dense(512))

    # Add the Rectified Linear Unit (ReLU) as Activation Function Layer,
    # for the Data of the Convolution Neural Network (CNN),
    # with the Data of the Modified NIST (MNIST)
    convolution_neural_network_model.add(Activation("relu"))

    # Add the Batch Normalization Layer, to normalize the previous Layer,
    # by re-centering and re-scaling the Data of the Convolution Neural Network (CNN),
    # with the Data of the Modified NIST (MNIST) and
    # making the Convolution Neural Network (CNN) faster and more stable
    convolution_neural_network_model.add(BatchNormalization())

    # Add the Dropout Layer, for Regularization of the Convolution Neural Network (CNN),
    # with the Data of the Modified NIST (MNIST),
    # by using as hyper-parameter, the rate of 50%
    # NOTE:
    # - Dropout Layer in Convolution Neural Networks is generally, not very useful;
    # - Comment/Uncomment, if you want to try it or not;
    convolution_neural_network_model.add(Dropout(0.5))

    # Sigmoid Classifier

    # Add a Dense Matrix of the Convolution Neural Network (CNN),
    # with the Data of the Modified NIST (MNIST) and 10 Units
    convolution_neural_network_model.add(Dense(2))

    # Add the Sigmoid as Activation Function Layer,
    # for the Data of the Convolution Neural Network (CNN),
    # with the Data of the Modified NIST (MNIST)
    convolution_neural_network_model.add(Activation("sigmoid"))

    # Return the Convolution Neural Network (CNN)
    return convolution_neural_network_model


# Relabel the ys (Labels) for the Training Set, for two classes (Odd and Even)
ys_labels_training_data = relabel_for_odd_and_even_classes(ys_labels_training_data)

# Relabel the ys (Labels) for the Testing Set, for two classes (Odd and Even)
ys_labels_testing_data = relabel_for_odd_and_even_classes(ys_labels_testing_data)

# Set one-hot encode for the ys (Labels) of the Training Set, with two classes (Odd and Even)
ys_labels_training_data = keras.utils.to_categorical(ys_labels_training_data, 2)

# Set one-hot encode for the ys (Labels) of the Testing Set, with two classes (Odd and Even)
ys_labels_testing_data = keras.utils.to_categorical(ys_labels_testing_data, 2)


# Initialise the Stochastic Gradient Descent (SGD) Optimizer,
# with the Learning Rate of INIT_LR, Momentum of 90% and Decay of (INIT_LR / NUM_EPOCHS)
stochastic_gradient_descent_optimizer = SGD(learning_rate=INITIAL_LEARNING_RATE,
                                            momentum=0.9,
                                            decay=(INITIAL_LEARNING_RATE / NUM_EPOCHS))

# Create the Convolution Neural Network (CNN) Model for the Data of
# the Modified NIST (MNIST)
cnn_model = create_convolution_neural_network_model()

# Compile the Convolution Neural Network (CNN) Model,
# with the given Binary Cross Entropy Loss/Error Function and
# the Stochastic Gradient Descent (SGD) Optimizer
cnn_model.compile(loss="binary_crossentropy",
                  optimizer=stochastic_gradient_descent_optimizer,
                  metrics=["binary_accuracy"])

# Print the Log for the Fitting of the Convolution Neural Network (CNN) Model
print(f"\nFitting the Convolution Neural Network (CNN) Model for {NUM_EPOCHS} Epochs "
      f"with a Batch Size of {BATCH_SIZE} and an Initial Learning Rate of {INITIAL_LEARNING_RATE}...\n")

# Train the Convolution Neural Network (CNN) Model for NUM_EPOCHS,
# with the Training Data for the Training Set and the Testing Data for the Validation Set
cnn_model_training_history = cnn_model.fit(xs_features_training_data, ys_labels_training_data,
                                           validation_data=(xs_features_testing_data, ys_labels_testing_data),
                                           batch_size=BATCH_SIZE, epochs=NUM_EPOCHS,)


# the use of High-Performance Computing (with CPUs and GPUs) is set to True
if TENSORFLOW_KERAS_HPC_BACKEND_SESSION:

    # Clear the current session of the Keras' Backend
    keras_backend.clear_session()


# Print the final Log for the Fitting of the Convolution Neural Network (CNN) Model
print("\nThe Fitting of the Convolution Neural Network (CNN) Model is complete!!!\n")

# Save the Weights of the Neurons of the Convolution Neural Network (CNN) Model
cnn_model.save_weights("mnist_model_odd_and_even.h5")

# Convert the Convolution Neural Network (CNN) Model to a JSON Object
cnn_model_json_object = cnn_model.to_json()

# Write the Convolution Neural Network (CNN) Model as a JSON Object
with open("mnist_model_odd_and_even.json", "w") as json_file:
    json_file.write(cnn_model_json_object)
