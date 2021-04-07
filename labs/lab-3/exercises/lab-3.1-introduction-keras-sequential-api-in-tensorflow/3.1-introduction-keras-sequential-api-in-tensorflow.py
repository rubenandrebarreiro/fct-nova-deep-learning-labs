"""
Lab 3.1 - Introduction to the Keras Sequential API in Tensorflow

Author:
- Rodrigo Jorge Ribeiro (rj.ribeiro@campus.fct.unl.pt)
- Ruben Andre Barreiro (r.barreiro@campus.fct.unl.pt)

NOTE:
- Just an Introduction to the Keras Sequential API in TensorFlow,
  this is NOT to execute;

"""

# Import the Libraries and Packages

# Import the Stochastic Gradient Descent (SGD) Optimizer
# from the TensorFlow.Keras.Optimizers Module
from tensorflow.keras.optimizers import SGD

# Import the Sequential from the TensorFlow.Keras.Models Module
from tensorflow.keras.models import Sequential

# Import the BatchNormalization, Conv2D, MaxPooling2D from the TensorFlow.Keras.Layers Module
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D

# Import the Activation, Flatten, Dropout and Dense from the TensorFlow.Keras.Layers Module
from tensorflow.keras.layers import Activation, Flatten, Dropout, Dense


# Constants

# The Learning Rate for the Stochastic Gradient Descent (SGD) Optimizer of
# the Convolution Neural Network (CNN)
INITIAL_LEARNING_RATE = 0.05

# The Number of Epochs for the Stochastic Gradient Descent (SGD) Optimizer of
# the Convolution Neural Network (CNN)
NUM_EPOCHS = 1000

# The Size of the Batch for the the Convolution Neural Network (CNN)
BATCH_SIZE = 32


# Create the Dummy/Empty Data for xs (Features) of the Training Datasets
xs_features_training_data = []

# Create the Dummy/Empty Data for ys (Labels) of the Training Datasets
ys_labels_training_data = []


# Create the Dummy/Empty Data for xs (Features) of the Testing Datasets
xs_features_testing_data = []

# Create the Dummy/Empty Data for ys (Labels) of the Testing Datasets
ys_labels_testing_data = []


# Create the Convolution Neural Network (CNN) Model
def create_convolution_neural_network_model():

    # Create the Sequential Model for the Convolution Neural Network (CNN)
    convolution_neural_network_model = Sequential()

    # Add a first Convolution 2D Matrix, for the Input Data of
    # the Fashion Modified NIST (Fashion MNIST),
    # with 32 Filters and a Kernel 3x3, Same Padding and
    # an Input Shape having a Batch Size of 28, with 28 Steps, as also,
    # 1 Input Dimension (for one Color Channel - Grayscale Color)
    convolution_neural_network_model.add(Conv2D(32, (3, 3), padding="same", input_shape=(28, 28, 1)))

    # Add the Rectified Linear Unit (ReLU) as Activation Function Layer,
    # for the Data of the Convolution Neural Network (CNN),
    # with the Data of the Fashion Modified NIST (Fashion MNIST)
    convolution_neural_network_model.add(Activation("relu"))

    # Add a Max Pooling 2D Sample-Based Discretization Process Layer,
    # for the Data of the Convolution Neural Network (CNN),
    # with the Data of the Fashion Modified NIST (Fashion MNIST) and a 2x2 Pool
    convolution_neural_network_model.add(MaxPooling2D(pool_size=(2, 2)))

    # Add a Convolution 2D Matrix Layer,
    # for the Data of the Convolution Neural Network (CNN),
    # with the Data of the Fashion Modified NIST (Fashion MNIST), 64 Filters and a Kernel 3x3
    convolution_neural_network_model.add(Conv2D(64, (3, 3), padding="same"))

    # Flatten the Data of the Convolution Neural Network (CNN),
    # with the Data of the Fashion Modified NIST (Fashion MNIST)
    # NOTE:
    # - This is needed to flatten the input into a single dimension for the features,
    #   which is what the next Dense Layer needs;
    convolution_neural_network_model.add(Flatten())

    # Add a Dense Matrix Layer for the Convolution Neural Network (CNN),
    # with the Data of the Fashion Modified NIST (Fashion MNIST) and 512 Units
    convolution_neural_network_model.add(Dense(512))

    # Add the Rectified Linear Unit (ReLU) as Activation Function Layer,
    # for the Data of the Convolution Neural Network (CNN),
    # with the Data of the Fashion Modified NIST (Fashion MNIST)
    convolution_neural_network_model.add(Activation("relu"))

    # Add the Batch Normalization Layer, to normalize the previous Layer,
    # by re-centering and re-scaling the Data of the Convolution Neural Network (CNN),
    # with the Data of the Fashion Modified NIST (Fashion MNIST) and
    # making the Convolution Neural Network (CNN) faster and more stable
    convolution_neural_network_model.add(BatchNormalization())

    # Add the Dropout Layer, for Regularization of the Convolution Neural Network (CNN),
    # with the Data of the Fashion Modified NIST (Fashion MNIST),
    # by using as hyper-parameter, the rate of 50%
    # NOTE:
    # - Dropout Layer in Convolution Neural Networks is generally, not very useful;
    # - Comment/Uncomment, if you want to try it or not;
    convolution_neural_network_model.add(Dropout(0.5))

    # Add a Dense Matrix of the Convolution Neural Network (CNN),
    # with the Data of the Fashion Modified NIST (Fashion MNIST) and 10 Units
    convolution_neural_network_model.add(Dense(10))

    # Add the Softmax as Activation Function Layer,
    # for the Data of the Convolution Neural Network (CNN),
    # with the Data of the Fashion Modified NIST (Fashion MNIST)
    convolution_neural_network_model.add(Activation("softmax"))

    # Return the Convolution Neural Network (CNN)
    return convolution_neural_network_model


# Initialise the Stochastic Gradient Descent (SGD) Optimizer,
# with the Learning Rate of INIT_LR, Momentum of 90% and Decay of (INIT_LR / NUM_EPOCHS)
stochastic_gradient_descent_optimizer = SGD(learning_rate=INITIAL_LEARNING_RATE,
                                            momentum=0.9,
                                            decay=(INITIAL_LEARNING_RATE / NUM_EPOCHS))

# Create the Convolution Neural Network (CNN) Model for the Data of
# the Fashion Modified NIST (Fashion MNIST)
cnn_model = create_convolution_neural_network_model()

# Compile the Convolution Neural Network (CNN) Model,
# with the given Categorical Cross Entropy Loss/Error Function and
# the Stochastic Gradient Descent (SGD) Optimizer
cnn_model.compile(loss="categorical_crossentropy",
                  optimizer=stochastic_gradient_descent_optimizer,
                  metrics=["accuracy"])

# Train the Convolution Neural Network (CNN) Model for NUM_EPOCHS,
# with the Training Data for the Training Set and the Testing Data for the Validation Set
cnn_model_training_history = cnn_model.fit(xs_features_training_data, ys_labels_training_data,
                                           validation_data=(xs_features_testing_data, ys_labels_testing_data),
                                           batch_size=BATCH_SIZE, epochs=NUM_EPOCHS)

# Save the Weights of the Neurons of the Convolution Neural Network (CNN) Model
cnn_model.save_weights("fashion_mnist_model.h5")

# Convert the Convolution Neural Network (CNN) Model to a JSON Object
cnn_model_json_object = cnn_model.to_json()

# Write the Convolution Neural Network (CNN) Model as a JSON Object
with open("fashion_mnist_model.json", "w") as json_file:
    json_file.write(cnn_model_json_object)
