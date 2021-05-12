"""
Lab 8.1 - Time Series Prediction

Author:
- Rodrigo Jorge Ribeiro (rj.ribeiro@campus.fct.unl.pt)
- Ruben Andre Barreiro (r.barreiro@campus.fct.unl.pt)

"""

# Import the Libraries and Packages

# Import the Operative System Library as operative_system
import os as operative_system

# Disable all the Debugging Logs from TensorFlow Library
operative_system.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from numpy import array

from numpy import reshape

from numpy import nan

from numpy import empty_like

from matplotlib import pyplot

from pandas import read_csv

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

from tensorflow.keras.layers import LSTM

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error

# Import the Set Style Function from the Seaborn Python's Module
from seaborn import set_style


def create_dataset(dataset_data, look_back_step=1):

    data_xs, data_ys = [], []

    for index_data in range(len(dataset_data) - look_back_step - 1):
        xs = dataset_data[index_data:(index_data + look_back_step)]
        ys = dataset_data[(index_data + look_back_step)]

        data_xs.append(xs)
        data_ys.append(ys)

    return array(data_xs), array(data_ys)


def plot_predictions(dataset_data, training_predictions, testing_predictions):

    # Set the Style for Seaborn Plotting
    set_style('darkgrid')

    training_predictions_plot = empty_like(dataset_data)
    training_predictions_plot[:] = nan
    training_predictions_plot[look_back:len(training_predictions) + look_back] = training_predictions

    testing_predictions_plot = empty_like(dataset_data)
    testing_predictions_plot[:] = nan
    testing_predictions_plot[len(training_predictions) + (look_back * 2) + 1:len(dataset_data) - 1] = \
        testing_predictions

    pyplot.plot(scaler.inverse_transform(dataset_data.reshape(-1, 1)), 'blue', label='Dataset - Real Targets')

    pyplot.plot(training_predictions_plot, 'green', label='Training Set - Predicted Targets')
    pyplot.plot(testing_predictions_plot, 'red', label='Testing Set - Predicted Targets')

    pyplot.title('Time Series - Passengers')

    pyplot.xlabel('Time Series')
    pyplot.ylabel('Num. Passengers')

    pyplot.savefig('passengers_predictions.png')
    pyplot.show()

    pyplot.close()


data_frame = read_csv('passengers.csv')

dataset = data_frame['Passengers'].values.astype('float32')

scaler = MinMaxScaler(feature_range=(0, 1))

dataset = scaler.fit_transform(dataset.reshape(-1, 1)).reshape((-1,))

training_size = int(len(dataset) * 0.67)

training_data, testing_data = dataset[0:training_size], dataset[training_size:len(dataset)]


look_back = 3

training_data_xs, training_data_ys = create_dataset(training_data, look_back)

testing_data_xs, testing_data_ys = create_dataset(testing_data, look_back)

training_data_xs = reshape(training_data_xs, (training_data_xs.shape[0], training_data_xs.shape[1], 1))

testing_data_xs = reshape(testing_data_xs, (testing_data_xs.shape[0], testing_data_xs.shape[1], 1))


batch_size = 1

model = Sequential()

model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True))

# NOTE:
# - By uncommenting this Long Short-Term Memory Layer,
#   probably, it will occur Overfitting;
# - If uncomment this, comment the above Long Short-Term Memory Layer;
# model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1),
#                stateful=True, return_sequences=True))
# model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1),
#                stateful=True))

model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')


# For 100 Iterations
for index in range(100):

    # Fit/Train the Model
    model.fit(training_data_xs, training_data_ys, epochs=1,
              batch_size=batch_size, verbose=2, shuffle=False)

    # Reset the States of the Model
    model.reset_states()

# Make the Predictions on the Training and Testing Sets
training_data_predicted_ys = model.predict(training_data_xs, batch_size=batch_size)
model.reset_states()
testing_data_predicted_ys = model.predict(testing_data_xs, batch_size=batch_size)

# Re-Scaling the Predicted and Real Targets on the Training Set
training_data_predicted_ys = scaler.inverse_transform(training_data_predicted_ys).reshape((-1,))
training_data_ys = scaler.inverse_transform(training_data_ys.reshape(-1, 1)).reshape((-1,))

# Re-Scaling the Predicted and Real Targets on the Testing Set
testing_data_predicted_ys = scaler.inverse_transform(testing_data_predicted_ys).reshape((-1,))
testing_data_ys = scaler.inverse_transform(testing_data_ys.reshape(-1, 1)).reshape((-1,))

# Compute the Score Errors (by the Mean Squared Error)
training_score_error = mean_squared_error(training_data_ys, training_data_predicted_ys)**0.5
testing_score_error = mean_squared_error(testing_data_ys, testing_data_predicted_ys)**0.5

# Compute the Training and Testing Score Errors
print('\n')
print('Training Score Error: ', training_score_error)
print('Testing Score Error: ', testing_score_error)

# Plot the Predictions for the Training and Testing Data
plot_predictions(dataset, training_data_predicted_ys, testing_data_predicted_ys)
