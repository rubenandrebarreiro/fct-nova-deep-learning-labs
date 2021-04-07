"""
Lab 2.5 - Modified NIST (MNIST) Multi-Class Classification Network

Author:
- Rodrigo Jorge Ribeiro (rj.ribeiro@campus.fct.unl.pt)
- Ruben Andre Barreiro (r.barreiro@campus.fct.unl.pt)

"""

# Import the Libraries and Packages

# Import the Operative System Library as operative_system
import os as operative_system

# Disable all the Debugging Logs from TensorFlow Library
operative_system.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import the TensorFlow Library as tensorflow alias
import tensorflow as tensorflow

# Import Keras Module from TensorFlow Library as keras alias
from tensorflow import keras as keras

# Import the Numpy Library as numpy
import numpy as numpy

# Import the DateTime Module from the DateTime Library
from datetime import datetime as date_time

# Retrieve the current DateTime, as custom format
now_date_time = date_time.utcnow().strftime("%Y%m%d%H%M%S")

# Set the Root Directory for the Logs
root_log_directory = "logs"

# Set the specific Log Directory, according to the current Date and Time (timestamp)
log_directory = "{}/model-{}/".format(root_log_directory, now_date_time)


# Retrieve the Dataset of the Modified NIST (MNIST)
mnist_dataset = keras.datasets.mnist

# Load the xs (Features/Images) and ys (Labels), from the Dataset, for Training and Testing
# The Training Set contains 60,000 examples (Images) and the Testing Set contains 10,000 examples (Images)
(xs_training_images, ys_training_labels), (xs_testing_images, ys_testing_labels) = mnist_dataset.load_data()

# Select the first 50,000 xs (Features/Images) for Training from the Training Set
xs_data_training_images_normalized = (xs_training_images[:50000].reshape(-1, 28*28) / 255.0)

# Select the first 50,000 ys (Labels) for Training from the Training Set
ys_data_training_labels = keras.utils.to_categorical(ys_training_labels[:50000])

# Select the last 10,000 xs (Features/Images) for Validation from the Training Set
xs_data_validation_images_normalized = (xs_training_images[50000:].reshape(-1, 28*28) / 255.0)

# Select the last 10,000 ys (Labels) for Validation from the Training Set
ys_data_validation_labels = keras.utils.to_categorical(ys_training_labels[50000:])


# Function to generate the Variables of a Layer for the Artificial Neural Network (ANN),
# i.e., Weights of the Neurons and the Bias, given the input xs (Features) of
# the Data of the Modified NIST (MNIST) or of the Weights of the previous Layer
def generate_artificial_neural_network_layer(inputs_xs_data, num_neurons):

    # Create the Weights of the Neurons for the Layer of Neurons
    layer_neurons_weights = tensorflow.Variable(tensorflow.random.normal((inputs_xs_data.shape[1]**2, num_neurons),
                                                                         stddev=(1 / num_neurons)))

    # Create the Bias for the Layer of Neurons
    layer_bias = tensorflow.Variable(tensorflow.zeros([num_neurons]))

    # Return the Weights of the Neurons and the Bias for the Layer of Neurons
    return layer_neurons_weights, layer_bias


# Function to create the Artificial Neural Network (ANN), given the initial input xs (Features) of
# the Data of the Modified NIST (MNIST)
def create_artificial_neural_network(inputs_xs_data, num_neurons_layer):

    # Initialise/Create the Artificial Neural Network
    artificial_neural_network = []

    # Initialise/Create the Variables,
    # i.e., the Weights of the Neurons and Bias for each Layer of the Artificial Neural Network (ANN)
    layers_variables = []

    # Initialise/Create the previous xs (Features) of
    # the current Layer of the Artificial Neural Network (ANN),
    # with the initial xs (Features) of the Data of the Modified NIST (MNIST)
    previous_inputs_xs_data = inputs_xs_data

    # For each Layer's Index and Number of Neurons on it
    for layer_index, layer_num_neurons in enumerate(num_neurons_layer):

        # Generate the Variables of a Layer for the Artificial Neural Network (ANN),
        # i.e., Weights of the Neurons and the Bias, given the input xs (Features) of
        # the Data of the Modified NIST (MNIST) or of the Weights of the previous Layer
        layer_neurons_weights, layer_bias = \
            generate_artificial_neural_network_layer(previous_inputs_xs_data, layer_num_neurons)

        # Append the computed Variables (Weights of the Neurons and the Bias) for
        # the current Layer of the Artificial Neural Network (ANN)
        artificial_neural_network.append((layer_neurons_weights, layer_bias))

        # Extend the computed Variables (Weights of the Neurons and the Bias) for
        # the current Layer of the Artificial Neural Network (ANN)
        layers_variables.extend((layer_neurons_weights, layer_bias))

        # Set the previous xs (Features) of
        # the current Layer of the Artificial Neural Network (ANN)
        previous_inputs_xs_data = inputs_xs_data

    # Return the Artificial Neural Network (ANN) and the Variables
    # (Weights of the Neurons and the Bias) for each Layer of it
    return artificial_neural_network, layers_variables


# Set the number of Neurons for each Layer of the Artificial Neural Network (ANN)
num_neurons_for_each_layer = [784, 784, 10]

# Create the Artificial Neural Network (ANN) and the Variables for each Layer
# (i.e., the Weights of the Neurons and the Bias), for the xs (Features) of the Data of the Modified NIST (MNIST)
neural_network, variables_layers = \
    create_artificial_neural_network(xs_training_images, num_neurons_for_each_layer)


# Function to predict the ys (Labels) of the Data of the Modified NIST (MNIST)
def neural_network_prediction(inputs_xs_data):

    # Initialise the Artificial Neural Network (ANN), with the xs (Features) of
    # the Data of the Modified NIST (MNIST)
    artificial_neural_network = inputs_xs_data

    # Initialise/Create the number of the current Layer of the Artificial Neural Network (ANN)
    num_artificial_neural_network_layer = 1

    # For the Weights of Neurons and Bias of each Layer of the Artificial Neural Network (ANN)
    for artificial_neural_network_layer_neurons_weights, \
            artificial_neural_network_layer_bias in neural_network[:-1]:

        # Initialise the Context Manager for the Name of Scopes of the Hidden Layers,
        # to name groups of operations the name attribute of the operations, in order to make the Graph clearer
        with tensorflow.name_scope(f"Layer_{num_artificial_neural_network_layer}"):

            # Update the Artificial Neural Network (ANN) with
            # the Sum of the Multiplication between the input xs (Features) of
            # the current Layer of the Artificial Neural Network (ANN) and the Weights of the Neurons,
            # and the Bias for the current Layer of the Artificial Neural Network (ANN)
            artificial_neural_network = \
                tensorflow.add(tensorflow.matmul(artificial_neural_network,
                                                 artificial_neural_network_layer_neurons_weights),
                               artificial_neural_network_layer_bias, name="net")

            # Compute the Rectified Linear Unit (ReLu) as Activation Function of
            # the Artificial Neural Network (ANN)
            artificial_neural_network = tensorflow.nn.leaky_relu(artificial_neural_network, name="relu")

        # Increment the number of the current Layer of the Artificial Neural Network (ANN)
        num_artificial_neural_network_layer += 1

    # Retrieve the Weights of Neurons and Bias of the last/output Layer of
    # the Artificial Neural Network (ANN)
    last_artificial_neural_network_layer_neurons_weights, \
        last_artificial_neural_network_layer_bias = neural_network[-1]

    # Initialise the Context Manager for the Name of Scopes of the Output Layer,
    # to name groups of operations the name attribute of the operations, in order to make the Graph clearer
    with tensorflow.name_scope("Output"):

        # Update the Artificial Neural Network (ANN) with
        # the last Sum of the Multiplication between the input xs (Features) of
        # the last/output Layer of the Artificial Neural Network (ANN) and the Weights of the Neurons,
        # and the Bias for the last/output Layer of the Artificial Neural Network (ANN)
        artificial_neural_network = \
            tensorflow.add(tensorflow.matmul(artificial_neural_network,
                                             last_artificial_neural_network_layer_neurons_weights),
                           last_artificial_neural_network_layer_bias)

    # Reshape the Artificial Neural Network (ANN), for the return the output of the Neural Network
    return tensorflow.reshape(artificial_neural_network, [-1])


# The function to compute the Cost of the Logistic Error Loss,
# between the predicted ys (Labels) of the Data of the Modified NIST (MNIST),
# through the Artificial Neural Network (ANN) and the real ys (Labels) of the Data of the Modified NIST (MNIST)
def compute_logistic_error_loss_softmax_activation(xs_data_features_to_predict, ys_real_data_labels):

    # Compute the Logit Function to the xs (Features) of the Data of the Genes
    tensorflow_network = neural_network_prediction(xs_data_features_to_predict)

    # Compute the Cost of Logistic Error Loss, with the Logits as argument,
    # which is the activation value before applying the Softmax Activation Function
    logistic_error_loss_cost = tensorflow.reduce_mean(
        tensorflow.nn.softmax_cross_entropy_with_logits(ys_real_data_labels, tensorflow_network)
    )

    # Return the Cost of the Logistic Error Loss
    return logistic_error_loss_cost


# The function to compute the Gradient for the Logistic Squared Error/Loss function
def compute_gradient(xs_data_features_to_predict, ys_real_data_labels, layer_variables):

    # Create the Gradient Tape to trace all the Computations and
    # to compute the Derivatives
    with tensorflow.GradientTape() as tape:

        # Predict the ys (Labels) for the xs (Features) of the Data of
        # the Modified NIST (MNIST) given as arguments of the Artificial Neural Network (ANN)
        neural_network_predicted_ys = neural_network_prediction(xs_data_features_to_predict)

        # Compute the Cost of the Mean Squared Error/Loss function
        loss_cost_value = compute_logistic_error_loss_softmax_activation(neural_network_predicted_ys,
                                                                         ys_real_data_labels)

    # Return the Gradient Tape with all the traced Computations and
    # computed Derivatives, as also, the Weights of Neurons and Bias
    return tape.gradient(loss_cost_value, layer_variables), layer_variables


# The function to create a graph for Tensorboard
# (for the computation of the Predictions, in this case)
@tensorflow.function
def create_graph_tensorboard(xs_data_features_to_predict):

    # Predict the ys (Labels) of the Data of the Modified NIST (MNIST)
    neural_network_prediction(xs_data_features_to_predict)


# The function to write the graph for Tensorboard
def write_graph_tensorboard(xs_data_features_to_predict, file_writer_logs):

    # Starts a trace to record computation graphs and profiling information
    tensorflow.summary.trace_on(graph=True)

    # Create a graph for Tensorboard
    # (for the computation of the Predictions, in this case)
    create_graph_tensorboard(tensorflow.constant(xs_data_features_to_predict.astype(numpy.float32)))

    # With the given File Writer by parameter, as default
    with file_writer_logs.as_default():

        # Stops and exports the active trace as a Summary and/or profile file,
        # and exports all metadata collected during the trace to the default SummaryWriter,
        # if one has been set (the File Writer, in this case)
        tensorflow.summary.trace_export(name="trace", step=0)


# Create a summary File Writer for the given Log Directory, specified before
file_writer = tensorflow.summary.create_file_writer(log_directory)

# Write the graph for the Tensorboard
write_graph_tensorboard(xs_training_images.reshape(-1, 28*28), file_writer)

# Configure the TensorFlow's Optimizer for the Stochastic Gradient Descent (SDG), with a Learning Rate of 10%
stochastic_gradient_descent_optimizer = tensorflow.optimizers.SGD(learning_rate=0.005, momentum=0.9)

# Set the Batch Size (i.e., the number of Samples/Examples) to
# work through before updating the Internal Model Parameters of the Learning Algorithm,
# and, for the Stochastic Gradient Descent (SDG), is common to use the value 1 as hyper-parameter
batch_size = 32

# Set the number of Batches, per Epoch
num_batches_per_epoch = (xs_training_images.shape[0] // batch_size)

# Set the number of Epochs (i.e., the number of times) that the Learning Algorithm
# (the Stochastic Gradient Descent (SDG), in this case) will work through the entire Dataset
num_epochs = 1000

# Initialize the sum of the Mean Squared Loss, for the Training Set
training_mean_squared_loss_sum = 0

# Initialize the sum of the Mean Squared Loss, for the Validation Set
validation_mean_squared_loss_sum = 0


# Execute the Artificial Neural Network (ANN), for the the Prediction of the Data of the Modified NIST (MNIST)
def execute_artificial_neural_network(file_writer_logs):

    # Define the scope for the sum of the Mean Squared Loss, for the Training Set
    global training_mean_squared_loss_sum

    # Define the scope for the sum of the Mean Squared Loss, for the Validation Set
    global validation_mean_squared_loss_sum

    # For each Epoch (i.e., each step of the Learning Algorithm)
    for current_epoch in range(num_epochs):

        # Shuffle the ys (Labels) for the Data of the Modified NIST (MNIST)
        ys_data_labels_shuffled = numpy.arange(len(ys_data_training_labels))
        numpy.random.shuffle(ys_data_labels_shuffled)

        # For each Batch (set of Samples), defined for a single Epoch
        for current_num_batch in range(num_batches_per_epoch):

            # Define the start index of the Samples
            start_num_sample = (current_num_batch * batch_size)

            # Retrieve the chosen Samples from the xs (Features) of the Data of the Modified NIST (MNIST),
            # from the Training Set
            batch_xs_data_training_images_normalized = \
                tensorflow.constant(xs_data_training_images_normalized[ys_data_labels_shuffled[start_num_sample:
                                                                       (start_num_sample + batch_size)], :]
                                    .astype(numpy.float32))

            # Retrieve the chosen Samples from the ys (Labels) of the Data of the Modified NIST (MNIST),
            # from the Training Set
            batch_ys_data_training_labels = \
                tensorflow.constant(ys_data_training_labels[ys_data_labels_shuffled[start_num_sample:
                                                            (start_num_sample + batch_size)]]
                                    .astype(numpy.float32))

            # Compute the Gradient for the Mean Squared Error Loss function for
            # the chosen Samples from the xs (Features) and ys (Labels) of the Data of the Modified NIST (MNIST)
            gradients, variables = compute_gradient(batch_xs_data_training_images_normalized,
                                                    batch_ys_data_training_labels,
                                                    variables_layers)

            # Apply the Gradients previously computed to the Learning Algorithm (Stochastic Gradient Descent (SDG))
            stochastic_gradient_descent_optimizer.apply_gradients(zip(gradients, variables))

        # Compute the predictions from data examples to the ys (Labels) available, from a given set of
        # xs (Features) of the Data of the Modified NIST (MNIST), from the Training Set,
        # using the configured Artificial Neural Network (ANN)
        ys_labels_predicted_for_training_set_data = \
            neural_network_prediction(tensorflow.constant(
                xs_data_training_images_normalized.astype(numpy.float32))
            )

        # Compute the predictions from data examples to the ys (Labels) available, from a given set of
        # xs (Features) of the Data of the Modified NIST (MNIST), from the Validation Set,
        # using the configured Artificial Neural Network (ANN)
        ys_labels_predicted_for_validation_set_data = \
            neural_network_prediction(tensorflow.constant(
                xs_data_validation_images_normalized.astype(numpy.float32))
            )

        # Compute the Cost of the Mean Squared Error Loss, between the predicted ys (Labels) of
        # the Data of the Modified NIST (MNIST), from the Training Set,
        # through the Artificial Neural Network (ANN) and the real ys (Labels) of
        # the Data of the Modified NIST (MNIST)
        training_logistic_loss = \
            compute_logistic_error_loss_softmax_activation(
                tensorflow.constant(ys_labels_predicted_for_training_set_data),
                tensorflow.constant(ys_data_training_labels.astype(numpy.float32)))

        # Compute the Cost of the Mean Squared Error Loss, between the predicted ys (Labels) of
        # the Data of the Modified NIST (MNIST), from the Validation Set,
        # through the Artificial Neural Network (ANN) and the real ys (Labels) of
        # the Data of the Modified NIST (MNIST)
        validation_logistic_loss = \
            compute_logistic_error_loss_softmax_activation(
                tensorflow.constant(ys_labels_predicted_for_validation_set_data),
                tensorflow.constant(ys_data_validation_labels.astype(numpy.float32)))

        # Print the Mean Squared Error Loss for the current Epoch of
        # the execution of the Artificial Neural Network (ANN), for the Training and Validation Sets
        print(f"Current Epoch: {current_epoch}, "
              f"Training Logistic Error Loss: {training_logistic_loss}, "
              f"Validation Logistic Error Loss: {validation_logistic_loss}...")

        # With the given File Writer by parameter, as default
        with file_writer_logs.as_default():

            # Write a scalar summary, regarding the Training Loss
            tensorflow.summary.scalar("Training Loss: ", training_logistic_loss, step=current_epoch)

            # Write a scalar summary, regarding the Validation Loss
            tensorflow.summary.scalar("Validation Loss: ", validation_logistic_loss, step=current_epoch)

        # Sum the current Mean Squared Loss to its accumulator, for the Training Set
        training_mean_squared_loss_sum = (training_mean_squared_loss_sum + training_logistic_loss)

        # Sum the current Mean Squared Loss to its accumulator, for the Validation Set
        validation_mean_squared_loss_sum = (validation_mean_squared_loss_sum + validation_logistic_loss)

    # Close the File Writer, for the Logs
    file_writer_logs.close()

    # Compute the average of the Mean Squared Loss, for the Training Set
    training_mean_squared_loss_average = (training_mean_squared_loss_sum / num_epochs)

    # Compute the average of the Mean Squared Loss, for the Validation Set
    validation_mean_squared_loss_average = (validation_mean_squared_loss_sum / num_epochs)

    # Print the information about the average of the Mean Squared Loss, for the Training Set
    print("\nThe average Mean Squared Loss for {} Epochs, in the Training Set, is: {}\n"
          .format(num_epochs, training_mean_squared_loss_average))

    # Print the information about the average of the Mean Squared Loss, for the Validation Set
    print("\nThe average Mean Squared Loss for {} Epochs, in the Validation Set, is: {}\n"
          .format(num_epochs, validation_mean_squared_loss_average))


# Print the configuration for the Artificial Neural Network (ANN) being used
print("\n\nStart the execution of the Artificial Neural Network (ANN), with {} Epochs, Bath Size of {},\n"
      "with the Learning Algorithm of Stochastic Gradient Descent (SDG), "
      "for the Dataset of the Modified NIST (MNIST)...\n".format(num_epochs, batch_size))

# Start the execution of the Artificial Neural Network (ANN), for the the Prediction of
# the Data of the Modified NIST (MNIST)
execute_artificial_neural_network(file_writer)
