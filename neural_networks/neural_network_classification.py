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
# # Imports

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

# %% [markdown]
# # Loading Data

# %%
file_path = "../data/mnist.pkl.gz"
training_data, validation_data, test_data = \
mnist_loader.load_data_wrapper(file_path)

# %% [markdown]
# ## Visualizing the Dataset

# %% [markdown]
# ### The First Digit

# %%
first_digit = training_data[0][0]

image_processing.print_image(first_digit)

# %% [markdown]
# ### The First Digit as a vector

# %%
plt.imshow(first_digit.T, cmap='gray', interpolation='nearest', aspect='auto')

# %% [markdown]
# ### Side by side comparison

# %%
ax[0]

# %%
fig, ax = plt.subplots(1,2)
ax[0].imshow(first_digit.reshape(28,28), cmap='gray')
ax[1].imshow(first_digit.T, cmap='gray', interpolation='nearest', aspect='auto');

# %% [markdown]
# # Training the Neural Network

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
image_processing.print_image(images2[0][0])

# %%
image_processing.print_image(images2[0][0].reshape(28, 28))

# %%
image_processing.print_image(training_data[0][0].reshape(28,28))

# %%
importlib.reload(network)
importlib.reload(image_processing)
importlib.reload(plotting)

# %%
image_processing.print_image(training_data[1][0])

# %%
