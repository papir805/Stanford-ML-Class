# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Introduction

# %%

# %% [markdown]
# # Imports

# %%
import importlib

# %%
#### Imports

# Standard Imports
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Import custom neural network modules
import mnist_loader
import network
import image_processing
import plotting

# %%
importlib.reload(network)
importlib.reload(image_processing)
importlib.reload(plotting)

# %% [markdown]
# # Loading Data

# %%
# Load datasets
file_path = "../data/mnist.pkl.gz"
training_data, validation_data, test_data = \
mnist_loader.load_data_wrapper(file_path)

# %% [markdown]
# ## Visualizing the Dataset

# %% [markdown]
# ### The First Digit From the Training Dataset

# %%
first_digit = training_data[0][0]

image_processing.print_image(first_digit)

# %% [markdown]
# #### As a Vector(What a Machine Sees)

# %%
plt.imshow(first_digit.T, cmap='gray', interpolation='nearest', aspect='auto')

# %% [markdown]
# #### Side by side comparison

# %%
fig, ax = plt.subplots(1,2)
ax[0].imshow(first_digit.reshape(28,28), cmap='gray')
ax[1].imshow(first_digit.T, cmap='gray', interpolation='nearest', aspect='auto');

# %% [markdown]
# #### As a Bar Graph

# %%
fig, ax = plt.subplots(1,1)
ax.bar(np.arange(0, 784, 1), first_digit.reshape(784))
ax.set_title("Histogram of Pixel Values for the digit 5")
ax.set_xlabel('pixel')
ax.set_ylabel('color_value');

# %% [markdown]
# ### The Second Digit From the Training Dataset

# %%
second_digit = training_data[1][0]

image_processing.print_image(second_digit.reshape(28,28))

# %% [markdown]
# #### As a Vector (What a Machine Sees)

# %%
plt.imshow(second_digit.T, cmap='gray', interpolation='nearest', aspect='auto')

# %% [markdown]
# #### Side by Side Comparison

# %%
fig, ax = plt.subplots(1,2)
ax[0].imshow(second_digit.reshape(28,28), cmap='gray')
ax[1].imshow(second_digit.T, cmap='gray', interpolation='nearest', aspect='auto');

# %% [markdown]
# #### As a Bar Graph

# %%
fig, ax = plt.subplots(1,1)
ax.bar(np.arange(0, 784, 1), second_digit.reshape(784))
ax.set_title("Histogram of Pixel Values for the digit 0")
ax.set_xlabel('pixel')
ax.set_ylabel('color_value');

# %%
fig, ax = plt.subplots(2,1, sharex=True)

fig.suptitle('Comparison of color values for each pixel')
ax[0].bar(np.arange(0, 784, 1), first_digit.reshape(784))
ax[0].set_title("Color values for a single 5 from the dataset")
ax[0].set_xlabel('pixel')
ax[0].set_ylabel('color_value')

ax[1].bar(np.arange(0, 784, 1), second_digit.reshape(784))
ax[1].set_title("Color values for a single 0 from the dataset")
ax[1].set_xlabel('pixel')
ax[1].set_ylabel('color_value')
plt.tight_layout();

# %%
fig, ax = plt.subplots(2,1, sharex=True)

fig.suptitle('Comparison of color values for each pixel')
ax[0].plot(np.arange(0, 784, 1), first_digit.reshape(784))
ax[0].set_title("Color values for a single 5 from the dataset")
ax[0].set_xlabel('pixel')
ax[0].set_ylabel('color_value')

ax[1].plot(np.arange(0, 784, 1), second_digit.reshape(784))
ax[1].set_title("Color values for a single 0 from the dataset")
ax[1].set_xlabel('pixel')
ax[1].set_ylabel('color_value')
plt.tight_layout();

# %%
fig, ax = plt.subplots(1,1)

fig.suptitle('Comparison of color values for each pixel of a single digit')
ax.bar(np.arange(0, 784, 1), first_digit.reshape(784), color='orange', label='five', alpha=0.5)


ax.bar(np.arange(0, 784, 1), second_digit.reshape(784), color='blue', label='zero', alpha=0.5)
#ax.set_title("Color values for a single 0 from the dataset")
ax.set_xlabel('pixel')
ax.set_ylabel('color_value')
plt.legend()
plt.tight_layout();

# %%
fig, ax = plt.subplots(1,1)

ax.plot(np.arange(0, 784, 1), first_digit.reshape(784), color='orange', label='five', alpha=0.5)

ax.plot(np.arange(0, 784, 1), second_digit.reshape(784), color='blue', label='zero', alpha=0.5)

fig.suptitle('Comparison of color values for each pixel of a single digit')
#ax.set_title("Color values for a single 0 from the dataset")
ax.set_xlabel('pixel')
ax.set_ylabel('color_value')
plt.legend()
plt.tight_layout();

# %% [markdown]
# ### Comparing the First and Second Digits

# %%
fig, ax = plt.subplots(2,1)
ax[0].imshow(first_digit.T, cmap='gray', interpolation='nearest', aspect='auto')
ax[1].imshow(second_digit.T, cmap='gray', interpolation='nearest', aspect='auto');

# %% [markdown]
# ### Visualizing Multiple images at Once

# %%
importlib.reload(network)
importlib.reload(image_processing)
importlib.reload(plotting)

# %%
images = plotting.get_images(training_data)

# %%
plotting.plot_heatmap(images, 1, plot_ten=True)

# %%
plotting.plot_heatmap(images, 1)

# %%
plotting.plot_heatmap(images, 7, plot_ten=True)

# %%
plotting.plot_heatmap(images, 7)

# %% [markdown]
# #### Bar Graphs

# %%
test_data = images[1][0:2]

# %%
a = np.array(test_data).reshape(len(test_data), 784)

# %% jupyter={"outputs_hidden": true}
a[0]

# %% jupyter={"outputs_hidden": true}
a[1]

# %%
b = a[0] + a[1] 

# %% jupyter={"outputs_hidden": true}
a.sum(axis=0) == b

# %%
test_data = images[1][0:10]

# %%
test_data[0].shape

# %%
a = np.array(test_data).reshape(10, 784).sum(axis=0)

# %%
plt.bar(np.arange(0, 784, 1), a)

# %%
test_data = images[0][0:10]
a = np.array(test_data).reshape(10, 784).sum(axis=0)

fig, ax = plt.subplots(1,1)

ax.bar(np.arange(0, 784, 1), a)
ax.set_xlabel('pixel')
ax.set_ylabel('color_value');

# %%
test_data = images[0]
a = np.array(test_data).reshape(len(test_data), 784).sum(axis=0)

fig, ax = plt.subplots(1,1)

ax.bar(np.arange(0, 784, 1), a)
ax.set_title('Pixel color values for all zeros in the training dataset')
ax.set_xlabel('pixel')
ax.set_ylabel('color_value');

# %%
test_data = images[1]
a = np.array(test_data).reshape(len(test_data), 784).sum(axis=0)

fig, ax = plt.subplots(1,1)

ax.bar(np.arange(0, 784, 1), a)
ax.set_title('Pixel color values for all ones in the training dataset')
ax.set_xlabel('pixel')
ax.set_ylabel('color_value');

# %%
test_data = images[5]
a = np.array(test_data).reshape(len(test_data), 784).sum(axis=0)

fig, ax = plt.subplots(2,1, sharex=True)

fig.suptitle('Comparison of sum of color values for each pixel')

ax[0].bar(np.arange(0, 784, 1), a)
ax[0].set_title("Color values for all fives from the dataset")
ax[0].set_xlabel('pixel')
ax[0].set_ylabel('color_value');

test_data = images[0]
a = np.array(test_data).reshape(len(test_data), 784).sum(axis=0)

ax[1].bar(np.arange(0, 784, 1), a)
ax[1].set_title("Color values for all zeros from the dataset")
ax[1].set_xlabel('pixel')
ax[1].set_ylabel('color_value')
plt.tight_layout();

# %%
test_data = images[5]
a = np.array(test_data).reshape(len(test_data), 784).sum(axis=0)

fig, ax = plt.subplots(2,1, sharex=True)

fig.suptitle('Comparison of sum of color values for each pixel')

ax[0].plot(np.arange(0, 784, 1), a)
ax[0].set_title("Color values for all fives from the dataset")
ax[0].set_xlabel('pixel')
ax[0].set_ylabel('color_value');

test_data = images[0]
a = np.array(test_data).reshape(len(test_data), 784).sum(axis=0)

ax[1].plot(np.arange(0, 784, 1), a)
ax[1].set_title("Color values for all zeros from the dataset")
ax[1].set_xlabel('pixel')
ax[1].set_ylabel('color_value')
plt.tight_layout();

# %%
test_data = images[5]
a = np.array(test_data).reshape(len(test_data), 784).sum(axis=0)

fig, ax = plt.subplots(1,1)

fig.suptitle('Comparison of sum of color values for each pixel across all of the same digit')

ax.bar(np.arange(0, 784, 1), a, label='five', color='orange', alpha=0.5)


test_data = images[0]
a = np.array(test_data).reshape(len(test_data), 784).sum(axis=0)

ax.bar(np.arange(0, 784, 1), a, label='zero', color='blue', alpha=0.5)
#ax.set_title("Color values for all zeros from the dataset")
ax.set_xlabel('pixel')
ax.set_ylabel('color_value')
plt.legend()
plt.tight_layout();

# %%
test_data = images[5]
a = np.array(test_data).reshape(len(test_data), 784).sum(axis=0)

fig, ax = plt.subplots(1,1)

fig.suptitle('Comparison of sum of color values for each pixel across all of the same digit')

ax.plot(np.arange(0, 784, 1), a, label='five', color='orange', alpha=0.5)


test_data = images[0]
a = np.array(test_data).reshape(len(test_data), 784).sum(axis=0)

ax.plot(np.arange(0, 784, 1), a, label='zero', color='blue', alpha=0.5)
#ax.set_title("Color values for all zeros from the dataset")
ax.set_xlabel('pixel')
ax.set_ylabel('color_value')
plt.legend()
plt.tight_layout();

# %%
test_data = images[2]
a = np.array(test_data).reshape(len(test_data), 784).sum(axis=0)
plt.bar(np.arange(0, 784, 1), a)

# %% [markdown]
# # Training the Neural Network

# %%
# Create a three layer neural network.  The first layer has 784 input neurons
# (one for each of the 784 pixels in the 28px by 28px image being fed in), the
# second layer has 30 neurons, and the third layer has 10 output neurons, one
# for each digit 0-9 representing the model's estimated probability of the
# image being that digit.
net = network.Network([784, 30, 10])

# %%
# Train the model over 30 epochs, with mini batch sizes of 10, and a learning
# rate of 3.0
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

# %% [markdown]
# # Generating Predictions From Your Own Handwriting

# %% [markdown]
# ## Image1

# %%
# Original Image
image1_path = "../images/handwritten_2.png"
image1 = cv2.imread(image1_path)
plt.imshow(image1);

# processed_image1 = \
# image_processing.process_image("../images/handwritten_2.png")

# %%
# Processed image
processed_image1 = \
image_processing.process_image(image1_path)
image_processing.print_image(processed_image1)

# %%
# Make prediction
net.predict(processed_image1)

# %% [markdown]
# ## Image2

# %%
# Original Image
image2_path = "../images/handwritten_8.jpg"
image2 = cv2.imread(image2_path)
plt.imshow(image2);

# %%
# Processed Image
processed_image2 = \
image_processing.process_image(image2_path)
image_processing.print_image(processed_image2)

# %%
net.predict(processed_image2)

# %% [markdown]
# ## Image3

# %%
# Original Image
image3_path = "../images/handwritten_4.jpg"
image3 = cv2.imread(image3_path)
plt.imshow(image3);

# %%
# Processed Image
processed_image3 = \
image_processing.process_image(image3_path)
image_processing.print_image(processed_image3)

# %%
net.predict(processed_image3)
