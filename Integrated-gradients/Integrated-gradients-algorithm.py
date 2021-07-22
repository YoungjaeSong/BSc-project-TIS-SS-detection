'''
This code calculates integrated gradients and displays the contribution score 
'''

from InputReader import datafile2ohe
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import InputReader as ir
# from MyKerasNNs import *
import numpy as np
from tensorflow.keras import backend as K
import Metrics as M
import matplotlib.pyplot as plt


# Feature Analysis via Integrated gradients
from Integrated_gradients_functions import *

print("Status report: Loading dataset")

datasetX, datasetY1, datasetY2, datasetY3, datasetY4  = datafile2ohe('TestDataset2.fn')
testX, testY1, testY2, testY3, testY4 = np.array(datasetX), np.array(datasetY1), np.array(datasetY2), np.array(datasetY3), np.array(datasetY4)

# Load model
f1 = M.f1score
rc = M.recall_m
prc = M.precision_m

precision_fn = keras.metrics.Precision(thresholds=0.5)
recall_fn = keras.metrics.Recall(thresholds=0.5)

TIS_bce = keras.losses.BinaryCrossentropy()
SS_bce = keras.losses.BinaryCrossentropy()

TIS_mse = keras.losses.MeanSquaredError()
SS_mse = keras.losses.MeanSquaredError()

print("Status report: Loading model")

model1 = keras.models.load_model('Trial30_1', custom_objects={'f1score':f1, 'recall_fn':recall_fn, 'precision_fn':precision_fn})

print("Status report: Building heatmap")

TIS_heatmap = tf.zeros((11, 4), dtype=tf.float32)

print("Total number of sequences in test set:", len(testX))
index = 0
seq_num = 0
number_of_sequences = 10

while index < number_of_sequences:
    print("Status report: {}th data. TIS heatmap construction from {}th data".format(index+1, seq_num))
    arr = testX[index]
    arr = np.expand_dims(arr, axis=0)

    # Copy of the original data
    orig_arr = np.copy(arr[0]).astype(np.uint8)

    # Preprocess the array
    arr_processed = tf.cast(arr, dtype=tf.float32)

    # Get model prediction
    pred = model1.predict(arr_processed)
    print('predicted probability: {}%'.format(pred[0][0][0]*100))
    print('predicted location {}, actual location{}'.format(pred[2][0][0], testY3[index]))
    # Get integrated gradients for the specific case
    igrads = random_baseline_integrated_gradients(
        model1, np.copy(orig_arr)
    )
    TIS_prob = testY1[index]
    TIS_loc = testY3[index]

    
    if TIS_prob < 0.5 or TIS_loc <= 4:
        index += 1
        continue
    
    else:
        print(testX[index, TIS_loc:TIS_loc+3])
        TIS_heatmap += igrads[TIS_loc-4:TIS_loc+7,:]
        seq_num += 1
        index += 1
        

TIS_heatmap = np.transpose(TIS_heatmap)
TIS_heatmap_avg = TIS_heatmap / seq_num

print(TIS_heatmap_avg)

x_axis = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6])
y_axis = np.array(['A', 'C', 'G', 'T'])

fig, ax = plt.subplots()
im = ax.imshow(TIS_heatmap_avg, cmap='hot', interpolation='nearest')

ax.set_xticks(np.arange(len(x_axis)))
ax.set_yticks(np.arange(len(y_axis)))

ax.set_xticklabels(x_axis)
ax.set_yticklabels(y_axis)

plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

for i in range(len(y_axis)):
    for j in range(len(x_axis)):
        text = ax.text(j, i, TIS_heatmap_avg[i, j], ha="left", va="bottom", rotation=30, color='b')

ax.set_title("Contribution Score Heatmap")

plt.savefig('heatmap4.png')
# # Get model predictions
# preds = model1.predict(arr_processed)
# # top_pred_idx = tf.argmax(preds[0])
# print("Predicted:", preds) # top_pred_idx, preds)

# # 5. Get the gradients of the last layer for the predicted label
# grads = get_gradients(model1, arr_processed)

# print("gradients:")
# print(grads)

# ig = get_integrated_gradients(model1, orig_arr)

# print("integrated gradients:")
# print(ig)

# igrads = random_baseline_integrated_gradients(
#     model1, np.copy(orig_arr)
# )

# print('random_baseline_integrated_gradients:')
# print(igrads)

'''
Now use for loop to get integrated gradients for all sequences and calculate
the contribution score by averaging the gradients and using the positional values,
i.e. the values in testY3, testY4
'''
