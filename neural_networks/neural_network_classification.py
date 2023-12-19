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

# %%
import importlib

# %%
import mnist_loader
import network
import image_processing
import plotting

# %%
importlib.reload(network)
importlib.reload(image_processing)
importlib.reload(plotting)

# %%
file_path = "../data/mnist.pkl.gz"
training_data, validation_data, test_data = \
mnist_loader.load_data_wrapper(file_path)

# %%
net = network.Network([784, 30, 10])

# %%
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

# %%
processed_image1 = \
image_processing.process_image("../images/handwritten_2.png")

# %%
image_processing.print_image(processed_image1)

# %%
net.predict(processed_image1)

# %%
processed_image2 = \
image_processing.process_image("../images/handwritten_8.jpg")
image_processing.print_image(processed_image2)

# %%
net.predict(processed_image2)

# %%
processed_image3 = \
image_processing.process_image("../images/handwritten_4.jpg")
image_processing.print_image(processed_image3)

# %%
net.predict(processed_image3)

# %%
type(processed_image3), processed_image3.shape

# %%

# %%
import matplotlib.pyplot as plt
plt.imshow(flattened_image3, cmap='gray')

# %%
plt.imshow(processed_image3, cmap='gray')

# %%
flattened_image1 = processed_image1.reshape(784) / 255.0
flattened_image2 = processed_image2.reshape(784) / 255.0
flattened_image3 = processed_image3.reshape(784) / 255.0

# %%
import seaborn as sns
import numpy as np

# %%
ax = sns.heatmap(flattened_image1, cmap='gray', cbar=False)

# %%
concat = [flattened_image1.T, flattened_image2.T, flattened_image3.T]
concat = np.array(concat)

# %%
concat.shape

# %%
ax = sns.heatmap(concat, cbar=False)

# %%
type(training_data[0])

# %%
training_data[0][0].shape

# %%
sns.heatmap(training_data[0][0].T, cbar=False)

# %%
training_data[0][1]

# %%
np.where(training_data[0][1] == 1)[0][0]

# %%
images = [[], [], [], [], [], [], [], [], [], []]
for data in training_data[0:5]:
    num = np.where(data[1] == 1)[0][0]
    print(num)
    gray = data[0]
    images[num].append(gray)

# %%
images[7][1]

# %%
len(images[3])

# %%
images[3][0].shape

# %%
len(images2[7])

# %%
images2[3][0].shape

# %%
# images_array = np.empty((1,784))
images_list = []

for image in images2[7]:
    #print(image)
    flattened_image = image.reshape(784) / 255.0
    # images_array = np.append(images_array, flattened_image, axis=0)
    images_list.append(flattened_image)

images_array = np.array(images_list)

# %%
images_array.shape

# %%
sns.heatmap(images_array, cbar=False)

# %%
images = [[], [], [], [], [], [], [], [], [], []]

for data in training_data:
    num = np.where(data[1] == 1)[0][0]
    #print(num)
    gray = data[0]
    images[num].append(gray)

# %%
# images_array = np.empty((1,784))
images_list = []

for image in images[0]:
    #print(image)
    flattened_image = image.reshape(784) / 255.0
    # images_array = np.append(images_array, flattened_image, axis=0)
    images_list.append(flattened_image)

images_array = np.array(images_list)

sns.heatmap(images_array, cbar=False)

# %%
sns.heatmap(images_array[0:10], cbar=False)

# %%
# images_array = np.empty((1,784))
images_list = []

for image in images[1]:
    #print(image)
    flattened_image = image.reshape(784) / 255.0
    # images_array = np.append(images_array, flattened_image, axis=0)
    images_list.append(flattened_image)

images_array = np.array(images_list)

sns.heatmap(images_array, cbar=False)

# %%
sns.heatmap(images_array[0:10], cbar=False)

# %%
arr = np.arange(0, images_array[0:10].shape[0], 1)
arr = np.append(arr, images_array[0:10].shape[0]-1)

# %%
arr

# %%
fig, ax = plt.subplots(1,1)
ax.imshow(images_array[0:10], cmap='hot', interpolation='nearest', aspect='auto')
ax.set_title('hello')
ax.xaxis.set_visible(False)
ax.set_yticks(arr, labels=arr+1)
ax.set_ylabel('image')
;

# %%
importlib.reload(network)
importlib.reload(image_processing)
importlib.reload(plotting)

# %%
images2 = plotting.get_images(training_data)

# %%
plotting.plot_heatmap(images2, 1, plot_ten=True)

# %%
plotting.plot_heatmap(images2, 1)

# %%
plotting.plot_heatmap(images2, 7, plot_ten=True)

# %%
plotting.plot_heatmap(images2, 7)

# %%
