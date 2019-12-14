
import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Dropout, Activation
from datetime import datetime
from sklearn.metrics import matthews_corrcoef

import multiprocessing
from multiprocessing import Pool

import time

#load images

print("Hello and welcome.")
print("Scanning image files from path...")

#mypath = "/home/fworg64/gradschool/csci575/salt_data/train/images"
#mypath2 = "/home/fworg64/gradschool/csci575/salt_data/train/masks"

mypath  = "/u/st/da/aoltmanns/Pictures/saltimages/train/images"
mypath2 = "/u/st/da/aoltmanns/Pictures/saltimages/train/masks"

(_, _, image_files) = next(os.walk(mypath))
(_, _, mask_files) = next(os.walk(mypath2))


# Load images and run function
image_list = []
mask_list = []

#my_kernel = np.array([[1, 0, 1],
#                      [0, 1, 0],
#                      [1, 0, 1]], dtype = "uint8")

# my_kernel = np.ones((8,8),dtype = "uint8")
# DFT Tile size
kernel_shape = (8, 8)
# kernel_shape = np.shape(my_kernel)
offset = np.int((kernel_shape[0]) / 2)

for idx, each_name in enumerate(image_files):
    print(each_name, end="  ")
    image = cv2.imread(mypath  + '/' + each_name)
    mask  = cv2.imread(mypath2 + '/' + each_name)
    # extend image by kernal_width/2

    padded_image = np.pad(image[:,:,0], offset, "symmetric")
    padded_mask  = np.pad(mask[:,:,0], offset, "symmetric")

    image_list.append(padded_image)
    mask_list.append(padded_mask)
    if idx > 1000:
        break

num_cores = 10#multiprocessing.cpu_count()
print("Processing image files with %d cores" % num_cores)
t0 = time.time()


image_mask_list = list(zip(image_list, mask_list))

# Pack images and masks
def load_image(image,mask):
    # Do DFT of chunk
    #image = image_mask_tuple[0]
    #mask = image_mask_tuple[1]
    dft_size = 8 # BAD GLOBAL
    dft_size_2 = int(dft_size/2)
    dims = np.shape(image)

    features_vec = []
    labels = []

    #print(np.shape(image))
    #print(np.shape(mask))

    for row_dex in range(0, dims[0]-dft_size):
        for col_dex in range(0, dims[1] - dft_size):
            #print("head")
            #print(row_dex)
            #print(col_dex)
            #print(row_dex + dft_size)
            #print(col_dex + dft_size)
            #print(np.shape(image))
            chunk = image[row_dex:(row_dex+dft_size), col_dex:(col_dex+dft_size)]
            #chunk1 = image[row_dex:(row_dex+dft_size)]
            #print(np.shape(chunk1))
            #chunk = chunk1[:][col_dex:(col_dex+dft_size)]
            #print(np.shape(chunk))
            #print(chunk)
            dft_res = np.fft.fft2(chunk)
            scaller = np.max(dft_res)
            if scaller != 0:
                dft_res = dft_res/np.abs(scaller)
            # Put absolute val in list
            features_vec.append(np.reshape(np.absolute(dft_res), -1).tolist())
            labels.append(mask[row_dex+dft_size_2][col_dex + dft_size_2]/255.0)

    return (features_vec, labels)



with Pool(num_cores) as p:
    feature_labels_data = p.starmap(load_image, image_mask_list)

print(type(feature_labels_data))
print(type(feature_labels_data[0][0]))
print(type(feature_labels_data[0][1]))

input_vector_list = []
output_class_list = []

for feature_label in feature_labels_data:
  input_vector_list.extend(feature_label[0])
  output_class_list.extend(feature_label[1])
print("input and output sizes")
print(np.shape(input_vector_list[0]))
print(np.shape(output_class_list[0]))
t1 = time.time()
total = t1-t0
print("Time taken:")
print(total)
first_nonzero = -1
for idx,val in enumerate(output_class_list):
  if val > 0:
      first_nonzero = idx
      break
print("Example input vector")
print(input_vector_list[first_nonzero])
print("Corresponding lable")
print(output_class_list[first_nonzero])
input_array = np.array(input_vector_list)
output_array = np.array(output_class_list)

x_train, x_test, y_train, y_test = train_test_split(
                                      input_array, output_array, test_size=0.4, random_state=0)

print("Done loading")

#kernel_input_shape = np.sum(my_kernel)
kernel_input_shape = kernel_shape[0]*kernel_shape[1]
layers = [
     Dense(8,input_shape=(kernel_input_shape,), activation='sigmoid'),
     Dense(8, activation='sigmoid'),
#     Dense(320,input_shape=(320,), activation='sigmoid'),
     Dense(1, activation='sigmoid'),
#     Dense(1,input_shape=(320,), activation='sigmoid'),
]
model = keras.Sequential(layers)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[
              "mse","mae", "binary_accuracy"])
model.summary()
model.fit(x_train, y_train, epochs=5, verbose=1)
now = datetime. now()
timestamp = datetime.timestamp(now)
model.save("model_file_" + str(timestamp) + ".h5")
model.summary()
print("model saved to :model_file_" + str(timestamp) + ".h5")
y_pred_float = model.predict(x_test)
print(y_pred_float)
my_threshold = 0.9
y_class_pred = [1.0*(x>my_threshold) for x in y_pred_float]
#bin_acc = keras.metrics.binary_accuracy(y_test, y_pred_float, threshold=my_threshold)
#y_class_pred = np.argmax(y_pred_float, axis=1)
print("Decided %d salt pixels out of %d total pixels with %d true salt" %
        (np.sum(y_class_pred), len(y_class_pred), np.sum(y_test)))
#print(y_class_pred)
matthews_corr_coef = matthews_corrcoef(y_test, y_class_pred)
print("Matthews Correlation Coef: %.4f" % (matthews_corr_coef,))
scores = model.evaluate(x_test, y_test, verbose=0)
print("FFNN Scores, mae, mae, mse, accuracy")
print(scores)
print("binary acc, confusion mat")
num_correct = 0
print("Comparing %d predictions with %d values" % (len(y_class_pred), len(y_test)))
print("Calculating confusion matrix, C[pred][true]")
confusion_mat = [[0,0],[0,0]]
for index in range(0,len(y_class_pred)):
    if (y_class_pred[index] == y_test[index]):
        num_correct = num_correct + 1
    confusion_mat[int(y_class_pred[index])][int(y_test[index])] = \
            confusion_mat[int(y_class_pred[index])][int(y_test[index])] + 1 
bin_acc = num_correct / len(y_test)
#bin_acc = keras.metrics.binary_accuracy(y_test, y_pred_float, threshold=my_threshold)
print(bin_acc)
total_0_label = confusion_mat[0][0]+confusion_mat[0][1]
correct_0_label = confusion_mat[0][0] / total_0_label
incorrect_0_label = 1 - correct_0_label
total_1_label = confusion_mat[1][0] + confusion_mat[1][1]
correct_1_label = confusion_mat[1][1] / total_1_label
incorrect_1_label = 1 - correct_1_label
print("%d total pixels labeled 0. %.3f percent correct, %.3f percent incorrect" % \
        (total_0_label, correct_0_label, incorrect_0_label))
print("%d total pixels labeled 1. %.3f percent correct, %.3f percent incorrect" % \
        (total_1_label, correct_1_label, incorrect_1_label))



