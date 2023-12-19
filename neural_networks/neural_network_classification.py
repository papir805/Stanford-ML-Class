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
