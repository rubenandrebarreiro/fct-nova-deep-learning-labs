"""
Lab 7.2.1 - Polynomial Regression with TensorFlow

Author:
- Rodrigo Jorge Ribeiro (rj.ribeiro@campus.fct.unl.pt)
- Ruben Andre Barreiro (r.barreiro@campus.fct.unl.pt)

"""

# Import the Libraries and Packages

# Import the Operative System Library as operative_system
import os as operative_system

# Disable all the Debugging Logs from TensorFlow Library
operative_system.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import TensorFlow Python's Module
import tensorflow

# Import Variable from TensorFlow Python's Module
from tensorflow import Variable

# Import Random from TensorFlow Python's Module
from tensorflow import random

# Import NumPy from NumPy Python's Module
import numpy

# Import LoadTxt from NumPy Python's Module
from numpy import loadtxt

# Import Mean from NumPy Python's Module
from numpy import mean

# Import Standard Deviation from NumPy Python's Module
from numpy import std

# Import Zeros from NumPy Python's Module
from numpy import zeros

# Import PyPlot from Matplotlib Python's Module
from matplotlib import pyplot

# Import the Set Style Function from the Seaborn Python's Module
from seaborn import set_style


# Function to expand the Data, in a Polynomial Equation, for a given Degree
def expand_data(xs_features, degree):

    # Create the Matrix for the xs (features) expanded, by the given Degree
    expanded_data = zeros((len(xs_features), degree))

    # Fill the 1st Column with xs (features) of the first Degree
    expanded_data[:, 0] = xs_features

    # For the remaining Degrees of the Polynomial Curve
    for curr_degree in range(2, degree+1):

        # Fill the remaining Columns with xs (features) of the remaining Degrees
        expanded_data[:, curr_degree-1] = xs_features**curr_degree

    # Returned the xs (features) expanded
    return expanded_data


# The function to compute the predictions from data examples to the ys (Labels) available,
# from a given set of xs (Features) of the Data of the Polynomial Regression,
# using the configured Artificial Neural Network (ANN)
def neural_network_prediction(xs_data_features_to_predict):

    # Create a TensorFlow Constant from the given set of xs (Features) of the Data of thePolynomial Regression
    tensorflow_constant_xs_data_features_to_predict = \
        tensorflow.constant(xs_data_features_to_predict.astype(numpy.float32))

    # Sum the xs (Features) of the Data of the Polynomial Regression weighted by the neurons (features x weights),
    # in order to build the Artificial Neural Network (ANN)
    tensorflow_network = tensorflow.add(tensorflow.matmul(tensorflow_constant_xs_data_features_to_predict,
                                                          weights_neurons), bias, name='net')

    # Reshape the Artificial Neural Network (ANN), for the return the output of the Neural Network,
    # using the Sigmoid Function as Activation Function for the Network
    return tensorflow_network


# The function to compute the Cost of the Mean Squared Error Loss,
# between the predicted ys (Labels) of the Data of the Polynomial Regression,
# through the Artificial Neural Network (ANN) and the real ys (Labels) of the Data of the Polynomial Regression
def mean_squared_error_loss(ys_predictions_for_xs_data_features, ys_real_data_labels):

    # Create a TensorFlow Constant from the given set of ys (Labels) of the Data of the Polynomial Regression
    tensorflow_constant_ys_real_data_labels = \
        tensorflow.constant(ys_real_data_labels.astype(numpy.float32))

    # Compute the Cost of the Differences of the Squared Error Loss, individually
    squared_error_loss_differences = tensorflow.math\
        .square(tensorflow_constant_ys_real_data_labels -
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
        # the Polynomial Regression given as arguments of the Artificial Neural Network (ANN)
        neural_network_predicted_ys = neural_network_prediction(xs_data_features_to_predict)

        # Compute the Cost of the Logistic Loss function
        loss_cost_value = mean_squared_error_loss(neural_network_predicted_ys, ys_real_data_labels)

    # Return the Gradient Tape with all the traced Computations and
    # computed Derivatives, as also, the Weights of Neurons and Bias
    return tape.gradient(loss_cost_value, [weights_neurons, bias]), [weights_neurons, bias]


# Execute the Artificial Neural Network (ANN), for the the Prediction of the Data of the Polynomial Regression
def execute_artificial_neural_network():

    # Define the scope for the sum of the Mean Squared Loss
    global mean_squared_loss_sum

    # For each Epoch (i.e., each step of the Learning Algorithm)
    for current_epoch in range(num_epochs):

        # Shuffle the ys (Labels) for the Data of the Polynomial Regression
        ys_data_labels_shuffled = numpy.arange(len(polydata_matrix_ys))
        numpy.random.shuffle(ys_data_labels_shuffled)

        # For each Batch (set of Samples), defined for a single Epoch
        for current_num_batch in range(num_batches_per_epoch):

            # Define the start index of the Samples
            start_num_sample = (current_num_batch * batch_size)

            # Retrieve the chosen Samples from the xs (Features) of the Data of the Polynomial Regression
            batch_xs_data_features = \
                polydata_matrix_xs_expanded[ys_data_labels_shuffled[start_num_sample:
                                                                    (start_num_sample + batch_size)], :]

            # Retrieve the chosen Samples from the ys (Labels) of the Data of the Polynomial Regression
            batch_ys_data_labels = \
                polydata_matrix_ys[ys_data_labels_shuffled[start_num_sample:
                                                           (start_num_sample + batch_size)]]

            # Compute the Gradient for the Logistic Loss function for
            # the chosen Samples from the xs (Features) and ys (Labels) of the Data of the Polynomial Regression
            gradients, variables = compute_gradient(batch_xs_data_features, batch_ys_data_labels)

            # Apply the Gradients previously computed to the Learning Algorithm (Stochastic Gradient Descent (SDG))
            stochastic_gradient_descent_optimizer.apply_gradients(zip(gradients, variables))

        # Compute the predictions from data examples to the ys (Labels) available, from a given set of
        # xs (Features) of the Data of the Polynomial Regression, using the configured Artificial Neural Network (ANN)
        ys_labels_predicted_for_data = neural_network_prediction(polydata_matrix_xs_expanded)

        # Compute the Cost of the Logistic Loss, between the predicted ys (Labels) of
        # the Data of the Polynomial Regression, through the Artificial Neural Network (ANN) and
        # the real ys (Labels) of the Data of the Polynomial Regression
        mean_squared_loss = mean_squared_error_loss(ys_labels_predicted_for_data, polydata_matrix_ys)

        # Print the Logistic Loss for the current Epoch of the execution of the Artificial Neural Network (ANN)
        print(f'Current Epoch: {current_epoch}, Mean Squared Loss: {mean_squared_loss}...')

        # Sum the current Mean Squared Loss to its accumulator
        mean_squared_loss_sum = (mean_squared_loss_sum + mean_squared_loss)

    # Compute the average of the Mean Squared Loss
    mean_squared_loss_average = (mean_squared_loss_sum / num_epochs)

    # Print the information about the average of the Logistic Loss
    print('\nThe average Mean Squared Loss for {} Epochs is: {}\n'.format(num_epochs, mean_squared_loss_average))


# Build the Plot of the Prediction using the Polynomial of Degree 3
def build_plot_for_linear_regression(degree):

    # Set the Style for Seaborn Plotting
    set_style('darkgrid')

    # Create a Linear Space for the ranges [-2,2]
    xs_plot = numpy.linspace(-2, 2)

    # Make the Prediction for the Polynomial of Degree 3
    ys_prediction = neural_network_prediction(expand_data(xs_plot, degree))

    # Plot the ys (True Labels) for the xs (Features)
    pyplot.plot(polydata_matrix_xs, polydata_matrix_ys, 'go', markersize=1)

    # Plot the Predictions
    pyplot.plot(xs_plot, ys_prediction)

    # Set the Title for the X-Axis for the Plot
    pyplot.xlabel('Initial xs (Features)')

    # Set the Title for the Y-Axis for the Plot
    pyplot.ylabel('True and Predicted ys (Labels)')

    # Set the Title of the Predictions for the Plot
    pyplot.title('Predictions with a Polynomial Curve of Degree {} '
                 'for Training/Fitting'.format(degree))

    # Save the Figure of the Plots
    pyplot.savefig('polynomial-curve-linear-regression.png')

    # Show the Plots
    pyplot.show()

    # Close the Plotting
    pyplot.close()


# Load the .txt file, for the Matrix of the Dataset
polydata_matrix = loadtxt('polydata.csv', delimiter=';', skiprows=0)

# Compute the Means of the Matrix of the Dataset
polydata_matrix_means = mean(polydata_matrix, axis=0)

# Compute the Standard Deviations of the Matrix of the Dataset
polydata_matrix_std_devs = std(polydata_matrix, axis=0)

# Standardize the Matrix of the Dataset
polydata_matrix = ((polydata_matrix - polydata_matrix_means) / polydata_matrix_std_devs)

# Retrieve the xs (features) from the Matrix of the Dataset
polydata_matrix_xs = polydata_matrix[:, :-1]

# Retrieve the ys (labels) from the Matrix of the Dataset
polydata_matrix_ys = polydata_matrix[:, -1:]

# Reshape the xs (features) from the Matrix of the Dataset
polydata_matrix_xs = polydata_matrix_xs.reshape((-1,))

# The Degree for the Polynomial Curve
polydata_degree = 3

# Create the column-vector with dimensions 3x1 for the variables 'weights',
# and fill it with pseudo random values from a Normal Distribution
weights_neurons = Variable(random.normal((polydata_degree, 1)), name='weights')

# Create a single variable for the 'bias', initialized with the value 0.0
bias = Variable(0.0, name='bias')

# Expand the Data, for a Polynomial of Degree 3
polydata_matrix_xs_expanded = expand_data(polydata_matrix_xs, polydata_degree)

# Configure the TensorFlow's Optimizer for the Stochastic Gradient Descent (SDG), with a Learning Rate of 10%
stochastic_gradient_descent_optimizer = tensorflow.optimizers.SGD(learning_rate=0.001, momentum=0.9)

# Set the Batch Size (i.e., the number of Samples/Examples) to
# work through before updating the Internal Model Parameters of the Learning Algorithm,
# and, for the Stochastic Gradient Descent (SDG), is common to use the value 1 as hyper-parameter
batch_size = 8

# Set the number of Batches, per Epoch
num_batches_per_epoch = (polydata_matrix_xs_expanded.shape[0] // batch_size)

# Set the number of Epochs (i.e., the number of times) that the Learning Algorithm
# (the Stochastic Gradient Descent (SDG), in this case) will work through the entire Dataset
num_epochs = 4000

# Initialize the sum of the Mean Squared Loss
mean_squared_loss_sum = 0

# Print the configuration for the Artificial Neural Network (ANN) being used
print('\n\nStart the execution of the Artificial Neural Network (ANN), with {} Epochs, Bath Size of {},\n'
      'with the Learning Algorithm of Stochastic Gradient Descent (SDG), '
      'for the Dataset of the Polynomial Regression...\n'
      .format(num_epochs, batch_size))

# Start the execution of the Artificial Neural Network (ANN),
# for the the Prediction of the Data of the Polynomial Regression
execute_artificial_neural_network()

# Build the Plot for the Linear Regression, using the Polynomial Curve
build_plot_for_linear_regression(polydata_degree)
