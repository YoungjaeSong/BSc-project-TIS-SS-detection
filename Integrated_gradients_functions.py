'''
This python file provides all functions needed to run the integrated gradients algorithm
'''
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

arr_size = (600, 4)

def convert_array(arr):
    arr = np.expand_dims(arr,axis=0)
    return arr

def get_gradients(model, arr_input):
    """Computes the gradients of outputs w.r.t input array

    Args:
        model: the prediction model
        arr_input: 2D ohe tensor
        label: predicted label for the input array
    
    Returns:
        Gradients of the predictions w.r.t arr_input
    """
    arr = tf.cast(arr_input, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(arr)
        pred = model(arr)
    
    grads = tape.gradient(pred, arr)

    return grads

def get_integrated_gradients(model, arr_input, baseline=None, num_steps=50):
    """Computes Integrated Gradients for a predicted label
    
    Args:
        arr_input (ndarray): Original array
        baseline (ndarray): The baseline array to start with for interpolation
        num_steps: Number of interpolation steps between the baseline and the
        input used in the computation of integrated gradients. These steps along
        determine the integral approximation error. By default num_steps is set
        to 50.

    Returns:
        Integrated gradients w.r.t input array
    """
    # If baseline is not provided, start with a black image
    # having same size as the input image
    if baseline is None:
        baseline = np.zeros(arr_size).astype(np.float32)
    else:
        baseline = baseline.astype(np.float32)
    
    # 1. Do interpolation
    arr_input = arr_input.astype(np.float32)
    interpolated_array = [
        baseline + (step / num_steps) * (arr_input - baseline)
        for step in range(num_steps)
    ]
    interpolated_array = np.array(interpolated_array).astype(np.float32)

    # 2. Preprocess the intepolated array
    # interpolated_array = xxxxception.preprocess_input(interpolated_image)

    # 3. Get the gradients
    grads = []
    for i,arr in enumerate(interpolated_array):
        arr = tf.expand_dims(arr,axis=0)
        grad = get_gradients(model, arr)
        grads.append(grad[0])
    # print(arr.shape())
    grads = tf.convert_to_tensor(grads, dtype=tf.float32)

    # 4. Approximate the integral using the trapezoidal rule
    grads = (grads[:1] + grads[1:]) / 2.0
    avg_grads = tf.reduce_mean(grads, axis=0)

    # 5. Calculate integrated gradients and return
    integrated_grads = (arr_input - baseline) * avg_grads

    return integrated_grads

def random_baseline_integrated_gradients(
    model, arr_input, num_steps=50, num_runs=2
    ):
    """Generates a number of random baseline arrays

    Args:
        arr_input (ndarray): 2D array
        num_steps: Number of interpolation steps between the baseline
            and the input used in the computation of integrated gradients. These
            steps along determine the integral approximation error. By default,
            num_steps is set to 50.
        num_runs: nubmer of baseline arrays to generate.

    Returns:
        Averaged integrated gradients for 'num_runs' baseline arrays
    """
    # 1. List to keep track of Integrated Gradients (IG) for all the arrays
    integrated_grads = []

    # 2. Get the integrated gradients for all the baselines
    for run in range(num_runs):
        baseline = np.random.random(arr_size) * 255
        igrads = get_integrated_gradients(
            model,
            arr_input=arr_input,
            baseline=baseline,
            num_steps=num_steps,
        )
        integrated_grads.append(igrads)
    
    # 3. Return the average integrated gradients for the image
    integrated_grads = tf.convert_to_tensor(integrated_grads)
    return tf.reduce_mean(integrated_grads, axis=0)