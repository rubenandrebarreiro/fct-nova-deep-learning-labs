"""
Lab 9.1 - Implementing a Restricted Boltzmann Machine (R.B.M.)

Author:
- Rodrigo Jorge Ribeiro (rj.ribeiro@campus.fct.unl.pt)
- Ruben Andre Barreiro (r.barreiro@campus.fct.unl.pt)

"""

# Import the Libraries and Packages

# Import the Operative System Library as operative_system
import os as operative_system

# Disable all the Debugging Logs from TensorFlow Library
operative_system.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import the TensorFlow Library
import tensorflow as tensorflow

# Import the NumPy Library
import numpy as numpy

# Import the PyPlot Module from the Matplotlib Library
import matplotlib.pyplot as py_plot


# Function to compute the Conditional Probability of
# the Hidden Variables, knowing the Visible Ones, i.e., the probability P(h|v)
def probability_hidden_given_visible(visible_variables, weights_vector, hidden_bias):
    return tensorflow.nn.sigmoid(tensorflow.matmul(visible_variables, weights_vector) + hidden_bias)


# Function to compute the Conditional Probability of
# the Visible Variables, knowing the Hidden Ones, i.e., the probability P(v|h)
def probability_visible_given_hidden(hidden_variables, weights_vector, visible_bias):
    return tensorflow.nn.sigmoid(tensorflow.matmul(hidden_variables, tensorflow.transpose(weights_vector)) + visible_bias)


# Function to sample the Variable States, from the not normalized probabilities
def sample_probabilities_variable_states(probabilities):
    return tensorflow.nn.relu(tensorflow.sign(probabilities - tensorflow.random.uniform(tensorflow.shape(probabilities))))


# Function for the Gibbs' Sampling
def gibbs_sampling(batch, k=1):

    hidden_prob = probability_hidden_given_visible(batch, weights_values, hidden_biases)
    hidden_states_0 = sample_probabilities_variable_states(hidden_prob)
    original_hs = hidden_states_0

    hidden_states = visible_states = 0

    for _ in range(k):
        visible_prob = probability_visible_given_hidden(hidden_states_0, weights_values, visible_biases)
        visible_states = sample_probabilities_variable_states(visible_prob)
        hidden_prob = probability_hidden_given_visible(visible_states, weights_values, hidden_biases)
        hidden_states = sample_probabilities_variable_states(hidden_prob)
        hidden_states_0 = hidden_states

    return original_hs, hidden_states, visible_states


# Function for the Training of the Model for the Restricted Boltzmann Machine (R.B.M.)
def train(xs_data, epochs=10):

    global weights_values, hidden_biases, visible_biases

    reconstruction_error_losses = []

    batch = visible_states = 0

    for epoch in range(epochs):

        for start, end in zip(range(0, len(xs_data), batch_size),
                              range(batch_size, len(xs_data), batch_size)):

            batch = xs_data[start:end]

            observation_hidden_states, hidden_states, visible_states = gibbs_sampling(batch, k=2)

            positive_gradients = tensorflow.matmul(tensorflow.transpose(batch), observation_hidden_states)
            negative_gradients = tensorflow.matmul(tensorflow.transpose(visible_states), hidden_states)

            current_batch_size = tensorflow.dtypes.cast(tensorflow.shape(batch)[0], tensorflow.float32)
            derivative_weights = (positive_gradients - negative_gradients) / current_batch_size

            weights_values += learning_rate * derivative_weights
            visible_biases += learning_rate * tensorflow.reduce_mean(batch - visible_states, 0)
            hidden_biases += learning_rate * tensorflow.reduce_mean(observation_hidden_states - hidden_states, 0)

        reconstruction_error = tensorflow.reduce_mean(tensorflow.square(batch - visible_states))

        print(f"Epoch: {epoch}, reconstruction error: {reconstruction_error}")

        reconstruction_error_losses.append(reconstruction_error)

    return reconstruction_error_losses


# Function for the Reconstruction of the Data, from the Restricted Boltzmann Machine (R.B.M.)
def restricted_boltzmann_machine_reconstruct(xs_data):

    # Compute the Hidden Values
    hidden_values = \
        tensorflow.nn.sigmoid(tensorflow.matmul(xs_data, weights_values) + hidden_biases)

    # Compute the xs (features) of the Reconstructed Data
    reconstructed_xs_data = \
        tensorflow.nn.sigmoid(
            tensorflow.matmul(hidden_values, tensorflow.transpose(weights_values)) + visible_biases
        )

    return reconstructed_xs_data


# Function for the Plotting of the MNIST (Modified NIST) examples,
# using the Reconstruction of the Data, from the Restricted Boltzmann Machine (R.B.M.)
def plot_mnist_examples_reconstruction():

    sample_for_reconstruction = testing_data[:10, :]
    noisy_sample = numpy.copy(sample_for_reconstruction)
    noisy_sample[numpy.random.rand(*sample_for_reconstruction.shape) > 0.98] = 1

    output_reconstruction = restricted_boltzmann_machine_reconstruct(noisy_sample)
    original_outputs = restricted_boltzmann_machine_reconstruct(sample_for_reconstruction)
    rows_reconstruction, columns_reconstruction = 4, 10

    # noinspection PyTypeChecker
    figure_reconstruction, axis_reconstruction = \
        py_plot.subplots(rows_reconstruction, columns_reconstruction, sharex=True, sharey=True, figsize=(20, 8))

    for column_reconstruction in range(columns_reconstruction):

        axis_reconstruction[0, column_reconstruction]\
            .imshow(tensorflow.reshape(sample_for_reconstruction[column_reconstruction], [28, 28]), cmap="Greys_r")
        axis_reconstruction[1, column_reconstruction]\
            .imshow(tensorflow.reshape(original_outputs[column_reconstruction], [28, 28]), cmap="Greys_r")
        axis_reconstruction[2, column_reconstruction]\
            .imshow(tensorflow.reshape(noisy_sample[column_reconstruction], [28, 28]), cmap="Greys_r")
        axis_reconstruction[3, column_reconstruction]\
            .imshow(tensorflow.reshape(output_reconstruction[column_reconstruction], [28, 28]), cmap="Greys_r")

        for axis in axis_reconstruction.flatten():

            axis.get_xaxis().set_visible(False)
            axis.get_yaxis().set_visible(False)

    py_plot.savefig("mnist_examples_reconstruction.png")
    py_plot.show()

    py_plot.close()


# Function for the Plotting of the MNIST (Modified NIST) examples,
# using the Generation of the Data, from the Restricted Boltzmann Machine (R.B.M.)
def plot_mnist_generation_from_noise():

    noisy_sample_for_generation = numpy.random.rand(10, 28*28).astype(numpy.float32)

    k = 1
    rows, columns = 6, 10

    outputs_generation = []

    for _ in range(1, rows):
        _, _, output_generated = gibbs_sampling(noisy_sample_for_generation, k)
        k *= 10
        outputs_generation.append(output_generated)

    # noinspection PyTypeChecker
    figure_generation, axis_generation = \
        py_plot.subplots(rows, columns, sharex=True, sharey=True, figsize=(20, 2*rows))

    for column in range(columns):

        axis_generation[0, column]\
            .imshow(tensorflow.reshape(noisy_sample_for_generation[column], [28, 28]), cmap="Greys_r")

        for row in range(1, rows):
            axis_generation[row, column]\
                .imshow(tensorflow.reshape(outputs_generation[row-1][column], [28, 28]), cmap="Greys_r")

        for ax in axis_generation.flatten():
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    py_plot.savefig("mnist_examples_generation_from_noise.png")
    py_plot.show()

    py_plot.close()


# Load the Data from the MNIST (Modified NIST) Dataset
(training_data, _), (testing_data, _) = tensorflow.keras.datasets.mnist.load_data()

# Normalise the data of the Training and Testing Sets
training_data = training_data / numpy.float32(255)
testing_data = testing_data / numpy.float32(255)

# Reshape the Training and Testing Sets to use 785 Visible Variables (the images are 28x28)
training_data = numpy.reshape(training_data, (training_data.shape[0], 784))
testing_data = numpy.reshape(testing_data, (testing_data.shape[0], 784))

# Set the Learning Rate as 100%
learning_rate = 1.0

# Set the size of the Visible Variables, as 785
visible_variables_size = training_data.shape[1]

# Set the size of the Hidden Variables, as 200
hidden_variables_size = 200

# Set the size of the Batch, as 100
batch_size = 100

# Create the vector of Weights' Values
weights_values = tensorflow.zeros([visible_variables_size, hidden_variables_size], numpy.float32)

# Create the vector for the Hidden Biases' Values
hidden_biases = tensorflow.zeros([hidden_variables_size], numpy.float32)

# Create the vector for the Visible Biases' Values
visible_biases = tensorflow.zeros([visible_variables_size], numpy.float32)

# Train the Model for the Restricted Boltzmann Machine (R.B.M.)
reconstruction_errors = train(training_data, 50)

# Plot the MNIST (Modified NIST) examples,
# using the Reconstruction of the Data, from the Restricted Boltzmann Machine (R.B.M.)
plot_mnist_examples_reconstruction()

# Function for the Plotting of the MNIST (Modified NIST) examples,
# using the Generation of the Data, from the Restricted Boltzmann Machine (R.B.M.)
plot_mnist_generation_from_noise()
