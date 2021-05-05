"""
Lab 7.1 - Occlusion Maps

Author:
- Rodrigo Jorge Ribeiro (rj.ribeiro@campus.fct.unl.pt)
- Ruben Andre Barreiro (r.barreiro@campus.fct.unl.pt)

"""

# Import the Libraries and Packages

# Import the Operative System Library as operative_system
import os as operative_system

# Disable all the Debugging Logs from TensorFlow Library
operative_system.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from numpy import zeros

from numpy import argmax

from numpy import min

from numpy import max

from tensorflow import constant

from tensorflow import reduce_max

from tensorflow import GradientTape

from tensorflow import math

from keras.models import load_model

from keras.backend import epsilon

from matplotlib import pyplot

from tp1_utils import load_data


def iterate_occlusion(image, size=8):

    for row in range(0, image.shape[1] - size + 1, size):

        for col in range(0, image.shape[2] - size + 1, size):

            temporary_image_copy = image.copy()
            temporary_image_copy[0, row:row + size, col:col + size, :] = (0.5, 0.5, 0.5)

            yield row, col, temporary_image_copy


def compute_occlusion_map(image, model, true_class, size=8):

    occlusion_map = zeros(image.shape[1:-1])

    for row, col, occluded in iterate_occlusion(image, size):
        prediction = model.predict(occluded)
        print(row, col, prediction[0, true_class])
        occlusion_map[row:row+size, col:col+size] = prediction[0, true_class]

    return occlusion_map


def compute_saliency_map(image, model, true_class):

    image_tensor = constant(image)

    with GradientTape() as tape:

        tape.watch(image_tensor)
        predictions = model(image_tensor)
        loss = predictions[:, true_class]

    gradient = tape.gradient(loss, image_tensor)
    gradient = reduce_max(math.abs(gradient), axis=-1)
    gradient = gradient.numpy()

    min_val, max_val = min(gradient), max(gradient)
    saliency_map = (gradient - min_val) / (max_val - min_val + epsilon())

    return saliency_map


multi_class_model = load_model('multi-class-model')
dataset = load_data()

# The Index of the Image
image_ix = 8  # NOTE: Change the Index, to try other images

# The Sub-set of data to be used
subset = 'test_X'  # NOTE: Change the Index, to try other Sub-Sets

# The xs (features) of the Test Set
image_xs = dataset[subset][image_ix:image_ix + 1]
class_ix = argmax(dataset['test_classes'][image_ix])

occlusion_map_xs = compute_occlusion_map(image_xs, multi_class_model, class_ix, size=8)
temporary_image = image_xs[0].copy()
temporary_image[:, :, 0] = (1 - occlusion_map_xs)

pyplot.imshow(temporary_image)

pyplot.savefig('occlusion-maps/{}_{}.png'.format(subset.lower(), image_ix))
pyplot.show()

pyplot.close()


saliency_map_xs = compute_saliency_map(image_xs, multi_class_model, class_ix)
temporary_image = image_xs[0].copy()
temporary_image[:, :, 0] = saliency_map_xs

pyplot.imshow(temporary_image)

pyplot.savefig('saliency-maps/{}_{}.png'.format(subset.lower(), image_ix))
pyplot.show()

pyplot.close()
