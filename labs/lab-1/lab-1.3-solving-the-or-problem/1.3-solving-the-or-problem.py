"""
Lab 1.3 - Solving the OR Problem

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

# Import the Numpy Library as numpy
import numpy as numpy

# Retrieve the ys (Labels) of the Data of the OR Function
ys_data_labels = numpy.array([0, 1, 1, 1])

# Retrieve the xs (Features) of the Data of the OR Function
xs_data_features = numpy.array([(0, 0), (0, 1), (1, 0), (1, 1)])

# Create the column-vector with dimensions 2x1 for the variables "weights",
# and fill it with pseudo random values from a Normal Distribution
weights_neurons = tensorflow.Variable(tensorflow.random.normal((2, 1)), name="weights")

# Create a single variable for the "bias", initialized with the value 0.0
bias = tensorflow.Variable(0.0, name="bias")


# The function to compute the predictions from data examples to the ys (Labels) available,
# from a given set of xs (Features) of the Data of the OR Function, using the configured Artificial Neural Network (ANN)
def neural_network_prediction(xs_data_features_to_predict):

    # Create a TensorFlow Constant from the given set of xs (Features) of the Data of the OR Function
    tensorflow_constant_xs_data_features_to_predict = \
        tensorflow.constant(xs_data_features_to_predict.astype(numpy.float32))

    # Sum the xs (Features) of the Data of the OR Function weighted by the neurons (features x weights),
    # in order to build the Artificial Neural Network (ANN)
    tensorflow_network = tensorflow.add(tensorflow.matmul(tensorflow_constant_xs_data_features_to_predict,
                                                          weights_neurons), bias, name="net")

    # Reshape the Artificial Neural Network (ANN), for the return the output of the Neural Network,
    # using the Sigmoid Function as Activation Function for the Network
    return tensorflow.reshape(tensorflow.nn.sigmoid(tensorflow_network, name="output"), [-1])


# The function to compute the Cost of the Mean Squared Error Loss,
# between the predicted ys (Labels) of the Data of the OR Function,
# through the Artificial Neural Network (ANN) and the real ys (Labels) of the Data of the OR Function
def mean_squared_error_loss(ys_predictions_for_xs_data_features, ys_real_data_labels):

    # Create a TensorFlow Constant from the given set of ys (Labels) of the Data of the OR Function
    tensorflow_constant_ys_predictions_for_xs_data_features = \
        tensorflow.constant(ys_real_data_labels.astype(numpy.float32))

    # Compute the Cost of the Differences of the Squared Error Loss, individually
    squared_error_loss_differences = tensorflow.math\
        .square(tensorflow_constant_ys_predictions_for_xs_data_features -
                ys_predictions_for_xs_data_features)

    # Compute global the Cost of the Mean Squared Error Loss
    mean_squared_error_loss_cost = tensorflow.reduce_mean(squared_error_loss_differences)

    # Return the global Cost of the Mean Squared Error Loss
    return mean_squared_error_loss_cost


# The function to compute the Gradient for the Logistic Loss function
def compute_gradient(xs_data_features_to_predict, ys_real_data_labels):

    # Create the Gradient Tape to trace all the Computations and
    # to compute the Derivatives
    with tensorflow.GradientTape() as tape:

        # Predict the ys (Labels) for the xs (Features) of the Data of
        # the OR Function given as arguments of the Artificial Neural Network (ANN)
        neural_network_predicted_ys = neural_network_prediction(xs_data_features_to_predict)

        # Compute the Cost of the Logistic Loss function
        loss_cost_value = mean_squared_error_loss(neural_network_predicted_ys, ys_real_data_labels)

    # Return the Gradient Tape with all the traced Computations and
    # computed Derivatives, as also, the Weights of Neurons and Bias
    return tape.gradient(loss_cost_value, [weights_neurons, bias]), [weights_neurons, bias]


# Configure the TensorFlow's Optimizer for the Stochastic Gradient Descent (SDG), with a Learning Rate of 10%
stochastic_gradient_descent_optimizer = tensorflow.optimizers.SGD(learning_rate=0.1)

# Set the Batch Size (i.e., the number of Samples/Examples) to
# work through before updating the Internal Model Parameters of the Learning Algorithm,
# and, for the Stochastic Gradient Descent (SDG), is common to use the value 1 as hyper-parameter
batch_size = 1

# Set the number of Batches, per Epoch
num_batches_per_epoch = (xs_data_features.shape[0] // batch_size)

# Set the number of Epochs (i.e., the number of times) that the Learning Algorithm
# (the Stochastic Gradient Descent (SDG), in this case) will work through the entire Dataset
num_epochs = 4000


# Execute the Artificial Neural Network (ANN), for the the Prediction of the Data of the OR Function
def execute_artificial_neural_network():

    # For each Epoch (i.e., each step of the Learning Algorithm)
    for current_epoch in range(num_epochs):

        # Shuffle the ys (Labels) for the Data of the OR Function
        ys_data_labels_shuffled = numpy.arange(len(ys_data_labels))
        numpy.random.shuffle(ys_data_labels_shuffled)

        # For each Batch (set of Samples), defined for a single Epoch
        for current_num_batch in range(num_batches_per_epoch):

            # Define the start index of the Samples
            start_num_sample = (current_num_batch * batch_size)

            # Retrieve the chosen Samples from the xs (Features) of the Data of the OR Function
            batch_xs_data_features = \
                xs_data_features[ys_data_labels_shuffled[start_num_sample:(start_num_sample + batch_size)], :]

            # Retrieve the chosen Samples from the ys (Labels) of the Data of the OR Function
            batch_ys_data_labels = ys_data_labels[ys_data_labels_shuffled[start_num_sample:
                                                                          (start_num_sample + batch_size)]]

            # Compute the Gradient for the Logistic Loss function for
            # the chosen Samples from the xs (Features) and ys (Labels) of the Data of the OR Function
            gradients, variables = compute_gradient(batch_xs_data_features, batch_ys_data_labels)

            # Apply the Gradients previously computed to the Learning Algorithm (Stochastic Gradient Descent (SDG))
            stochastic_gradient_descent_optimizer.apply_gradients(zip(gradients, variables))

        # Compute the predictions from data examples to the ys (Labels) available, from a given set of
        # xs (Features) of the Data of the OR Function, using the configured Artificial Neural Network (ANN)
        ys_labels_predicted_for_data = neural_network_prediction(xs_data_features)

        # Compute the Cost of the Logistic Loss, between the predicted ys (Labels) of the Data of the OR Function,
        # through the Artificial Neural Network (ANN) and the real ys (Labels) of the Data of the OR Function
        mean_squared_loss = mean_squared_error_loss(ys_labels_predicted_for_data, ys_data_labels)

        # Print the Logistic Loss for the current Epoch of the execution of the Artificial Neural Network (ANN)
        print(f"Current Epoch: {current_epoch}, Mean Squared Loss: {mean_squared_loss}...")


# Print the configuration for the Artificial Neural Network (ANN) being used
print("\n\nStart the execution of the Artificial Neural Network (ANN), with {} Epochs, Bath Size of {},\n"
      "with the Learning Algorithm of Stochastic Gradient Descent (SDG), for the Dataset of the OR Function...\n"
      .format(num_epochs, batch_size))

# Start the execution of the Artificial Neural Network (ANN), for the the Prediction of the Data of the OR Function
execute_artificial_neural_network()
