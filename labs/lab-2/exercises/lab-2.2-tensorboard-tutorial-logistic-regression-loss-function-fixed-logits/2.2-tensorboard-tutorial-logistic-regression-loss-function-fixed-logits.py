"""
Lab 2.2 - Tensorboard (Tutorial) - Logistic Regression (Loss Function fixed/stable)

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

# Import the DateTime Module from the DateTime Library
from datetime import datetime as date_time

# Retrieve the current DateTime, as custom format
now_date_time = date_time.utcnow().strftime("%Y%m%d%H%M%S")

# Set the Root Directory for the Logs
root_log_directory = "logs"

# Set the specific Log Directory, according to the current Date and Time (timestamp)
log_directory = "{}/model-{}/".format(root_log_directory, now_date_time)


# Load the text file (.txt) with the data of the genes, having the form of a Matrix
matrix_data = numpy.loadtxt("files/gene_data.txt", delimiter="\t")

# Retrieve the ys (Labels) of the Data of the Genes
ys_data_labels = matrix_data[:, -1]

# Retrieve the xs (Features) of the Data of the Genes
xs_data_features = matrix_data[:, :-1]

# Compute the Means of the xs (Features) of the Data of the Genes
xs_data_features_means = numpy.mean(xs_data_features, 0)

# Compute the Standard Deviation of the xs (Features) of the Data of the Genes
xs_data_features_standard_deviation = numpy.std(xs_data_features, 0)

# Standardize the xs (Features) of the Data of the Genes, from its mean and standard deviation
xs_data_features_standardized = ((xs_data_features - xs_data_features_means) / xs_data_features_standard_deviation)

# Just for Fun, to force the Data of the Genes, to give NaN Errors
# by multiply it for a scalar of 10
# (Uncomment if you want to try it...)
# xs_data_features_standardized = (xs_data_features_standardized * 10)

# Just for Fun, to force the Data of the Genes, to give NaN Errors,
# by multiply it for a scalar of 100
# (Uncomment if you want to try it...)
# xs_data_features_standardized = (xs_data_features_standardized * 100)

# Create the column-vector with dimensions 2x1 for the variables "weights",
# and fill it with pseudo random values from a Normal Distribution
weights_neurons = tensorflow.Variable(tensorflow.random.normal((2, 1)))

# Create a single variable for the "bias", initialized with the value 0.0
bias = tensorflow.Variable(0.0)


# The function to compute the predictions from data examples to the ys (Labels) available,
# from a given set of xs (Features) of the Data of the Genes, using the configured Artificial Neural Network (ANN)
def neural_network_prediction(xs_data_features_to_predict):

    # Sum the xs (Features) of the Data of the Genes weighted by the neurons (features x weights),
    # in order to build the Artificial Neural Network (ANN)
    tensorflow_network = tensorflow.add(tensorflow.matmul(xs_data_features_to_predict,
                                                          weights_neurons), bias, name="net")

    # Reshape the Artificial Neural Network (ANN), for the return the output of the Neural Network,
    # using the Sigmoid Function as Activation Function for the Network
    return tensorflow.reshape(tensorflow.nn.sigmoid(tensorflow_network, name="output"), [-1])


# The Logistic Loss Function, where x is the argument for the Sigmoid Function
# (also known as "logit") and y is the target values
# NOTES:
# - In statistics, the Logit Function (or, the Log-Odds) is the Logarithm of the Odds [p / (1 - p)],
#   where p is a probability. It is the inverse of the Sigmoid "Logistic" Function or
#   Logistic Transform used in mathematics, especially in statistics;
# - This function creates a mapping of the probability values from (0,1) to (-inf,+inf);
# - logit(p) = log( [p / (1 - p)] ) = log(p) - log( [1 - p] ) = - log( [1 - p] ) + log(p) =
#            = - ( log( [1 - p] ) - log(p) ) = - [ log( [1 - p] / p ) ] =
#            = - log( [ (1 / p) - (p / p) ] ) = - log( [ (1 / p) - 1 ] )
def logits(xs_data_features_to_predict):

    # Sum the xs (Features) of the Data of the Genes weighted by the neurons (features x weights),
    # in order to build the Artificial Neural Network (ANN)
    tensorflow_network = tensorflow.add(tensorflow.matmul(xs_data_features_to_predict,
                                                          weights_neurons), bias, name="net")

    # Reshape the Artificial Neural Network (ANN), for the return the output of the Neural Network,
    # using the Sigmoid Function as Activation Function for the Network
    return tensorflow.reshape(tensorflow_network, [-1])


# The function to compute the Cost of the Logistic Error Loss,
# between the predicted ys (Labels) of the Data of the Genes,
# through the Artificial Neural Network (ANN) and the real ys (Labels) of the Data of the Genes
def compute_logistic_error_loss(xs_data_features_to_predict, ys_real_data_labels):

    # Compute the Logit Function to the xs (Features) of the Data of the Genes
    tensorflow_network = logits(xs_data_features_to_predict)

    # Compute the Cost of Logistic Error Loss, with the Logits as argument,
    # which is the activation value before applying the Sigmoid Function
    logistic_error_loss_cost = tensorflow.reduce_mean(
        tensorflow.nn.sigmoid_cross_entropy_with_logits(ys_real_data_labels, tensorflow_network)
    )

    # Return the Cost of the Logistic Error Loss
    return logistic_error_loss_cost


# The function to compute the Gradient for the Logistic Loss function
def compute_gradient(xs_data_features_to_predict, ys_real_data_labels):

    # Create the Gradient Tape to trace all the Computations and
    # to compute the Derivatives
    with tensorflow.GradientTape() as tape:

        # Compute the Cost of the Logistic Error Loss function
        loss_cost_value = compute_logistic_error_loss(xs_data_features_to_predict, ys_real_data_labels)

    # Return the Gradient Tape with all the traced Computations and
    # computed Derivatives, as also, the Weights of Neurons and Bias
    return tape.gradient(loss_cost_value, [weights_neurons, bias]), [weights_neurons, bias]


# The function to create a graph for Tensorboard
# (for the computation of the Gradients, in this case)
@tensorflow.function
def create_graph_tensorboard(xs_data_features_to_predict, ys_real_data_labels):

    # Call the function to compute the Gradient for the Logistic Loss function
    compute_gradient(xs_data_features_to_predict, ys_real_data_labels)


# The function to write the graph for Tensorboard
def write_graph_tensorboard(xs_data_features_to_predict, ys_real_data_labels, file_writer_logs):

    # Starts a trace to record computation graphs and profiling information
    tensorflow.summary.trace_on(graph=True)

    # Create a graph for Tensorboard
    # (for the computation of the Gradients, in this case)
    create_graph_tensorboard(tensorflow.constant(xs_data_features_to_predict.astype(numpy.float32)),
                             tensorflow.constant(ys_real_data_labels.astype(numpy.float32)))

    # With the given File Writer by parameter, as default
    with file_writer_logs.as_default():

        # Stops and exports the active trace as a Summary and/or profile file,
        # and exports all metadata collected during the trace to the default SummaryWriter,
        # if one has been set (the File Writer, in this case)
        tensorflow.summary.trace_export(name="trace", step=0)


# Create a summary File Writer for the given Log Directory, specified before
file_writer = tensorflow.summary.create_file_writer(log_directory)

# Write the graph for the Tensorboard
write_graph_tensorboard(xs_data_features_standardized, ys_data_labels, file_writer)


# Configure the TensorFlow's Optimizer for the Stochastic Gradient Descent (SDG), with a Learning Rate of 10%
stochastic_gradient_descent_optimizer = tensorflow.optimizers.SGD(learning_rate=0.1)

# Set the Batch Size (i.e., the number of Samples/Examples) to
# work through before updating the Internal Model Parameters of the Learning Algorithm,
# and, for the Stochastic Gradient Descent (SDG), is common to use the value 1 as hyper-parameter
batch_size = 1

# Set the number of Batches, per Epoch
num_batches_per_epoch = (xs_data_features_standardized.shape[0] // batch_size)

# Set the number of Epochs (i.e., the number of times) that the Learning Algorithm
# (the Stochastic Gradient Descent (SDG), in this case) will work through the entire Dataset
num_epochs = 50

# Initialize the sum of the Logistic Loss
logistic_loss_sum = 0


# Execute the Artificial Neural Network (ANN), for the the Prediction of the Data of the Genes
def execute_artificial_neural_network(file_writer_logs):

    # Define the scope for the sum of the Logistic Loss
    global logistic_loss_sum

    # For each Epoch (i.e., each step of the Learning Algorithm)
    for current_epoch in range(num_epochs):

        # Shuffle the ys (Labels) for the Data of the Genes
        ys_data_labels_shuffled = numpy.arange(len(ys_data_labels))
        numpy.random.shuffle(ys_data_labels_shuffled)

        # For each Batch (set of Samples), defined for a single Epoch
        for current_num_batch in range(num_batches_per_epoch):

            # Define the start index of the Samples
            start_num_sample = (current_num_batch * batch_size)

            # Retrieve the chosen Samples from the xs (Features) of the Data of the Genes
            batch_xs_data_features_standardized = \
                tensorflow.constant(xs_data_features_standardized[ys_data_labels_shuffled[start_num_sample:
                                                                  (start_num_sample + batch_size)], :]
                                    .astype(numpy.float32))

            # Retrieve the chosen Samples from the ys (Labels) of the Data of the Genes
            batch_ys_data_labels = \
                tensorflow.constant(ys_data_labels[ys_data_labels_shuffled[start_num_sample:
                                                   (start_num_sample + batch_size)]]
                                    .astype(numpy.float32))

            # Compute the Gradient for the Logistic Error Loss function for
            # the chosen Samples from the xs (Features) and ys (Labels) of the Data of the Genes
            gradients, variables = compute_gradient(batch_xs_data_features_standardized, batch_ys_data_labels)

            # Apply the Gradients previously computed to the Learning Algorithm (Stochastic Gradient Descent (SDG))
            stochastic_gradient_descent_optimizer.apply_gradients(zip(gradients, variables))

        # Compute the predictions from data examples to the ys (Labels) available, from a given set of
        # xs (Features) of the Data of the Genes, using the configured Artificial Neural Network (ANN)
        # ys_labels_predicted_for_data = \
        #     neural_network_prediction(tensorflow.constant(xs_data_features_standardized.astype(numpy.float32)))

        # Compute the Cost of the Logistic Error Loss, between the predicted ys (Labels) of the Data of the Genes,
        # through the Artificial Neural Network (ANN) and the real ys (Labels) of the Data of the Genes
        logistic_loss = \
            compute_logistic_error_loss(tensorflow.constant(xs_data_features_standardized.astype(numpy.float32)),
                                        tensorflow.constant(ys_data_labels.astype(numpy.float32)))

        # Print the Logistic Error Loss for the current Epoch of the execution of the Artificial Neural Network (ANN)
        print(f"Current Epoch: {current_epoch}, Logistic Error Loss: {logistic_loss}...")

        # With the given File Writer by parameter, as default
        with file_writer_logs.as_default():

            # Write a scalar summary, regarding the Training Loss
            tensorflow.summary.scalar("Training Loss: ", logistic_loss, step=current_epoch)

        # Sum the current Logistic Loss to its accumulator
        logistic_loss_sum = (logistic_loss_sum + logistic_loss)

    # Close the File Writer, for the Logs
    file_writer_logs.close()

    # Compute the average of the Logistic Loss
    logistic_loss_average = (logistic_loss_sum / num_epochs)

    # Print the information about the average of the Logistic Loss
    print("\nThe average Logistic Loss for {} Epochs is: {}\n".format(num_epochs, logistic_loss_average))


# Print the configuration for the Artificial Neural Network (ANN) being used
print("\n\nStart the execution of the Artificial Neural Network (ANN), with {} Epochs, Bath Size of {},\n"
      "with the Learning Algorithm of Stochastic Gradient Descent (SDG), for the Dataset of the Genes...\n"
      .format(num_epochs, batch_size))

# Start the execution of the Artificial Neural Network (ANN), for the the Prediction of the Data of the Genes
execute_artificial_neural_network(file_writer)
