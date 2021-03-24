"""
Lab 1.1 - Introduction to TensorFlow

Author:
- Rodrigo Ribeiro (rj.ribeiro@campus.fct.unl.pt)
- Ruben Andre Barreiro (r.barreiro@campus.fct.unl.pt)

"""

# Import the Libraries and Packages

# Import the Operative System Library as operative_system
import os as operative_system

# Disable all the Debugging Logs from TensorFlow Library
operative_system.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import the TensorFlow Library as tensorflow alias
import tensorflow as tensorflow

# Constants
LOGGING_FLAG = True

# Create the constant "a" as a float of 32 bits and
# assign to it the value 3
a = tensorflow.constant(3.0, dtype=tensorflow.float32, name="a")

# Create the constant "b" and assign to it the value 4
b = tensorflow.constant(4.0, name="b")

# Create the addition of the constants "a" and "b", as "total",
# i.e., total = a + b
total = tensorflow.add(a, b, name="total")

# If the Logging Flag is set to True
if LOGGING_FLAG:

    # Print the header for the Logging
    tensorflow.print("\n\nLogging of the Execution:\n")

    # Print the Tensor for the constant "a"
    tensorflow.print("a = ", a)

    # Print the Tensor for the constant "b"
    tensorflow.print("b = ", b)

    # Print the Tensor for the addition of
    # the constants "a" and "b", as "total"
    tensorflow.print("total = a + b = ", total)
