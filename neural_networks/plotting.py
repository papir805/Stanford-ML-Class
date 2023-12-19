def get_images(input_data):
    
    import numpy as np
    
    # Initialize list containing ten empty lists.  Each of the ten empty
    # lists will contain the pixels of images corresponding to a 
    # particular digit.  For instance, the 4th empty list will hold
    # the pixels for all images corresponding to the digit 4.
    images = [[], [], [], [], [], [], [], [], [], []]
    
    for data in input_data:
        
        #data is a tuple of the form (pixels, digits)
        
        #pixels is a (784, 1) np.array containing color values
        #representing a 28x28 pixel image of the handwritten digit 
        
        #digits is a (10,1) np.array 0s and a single 1.  The row_idx
        #of the 1 represents which digit the pixels represent
        
        # Store the pixel color values
        pixels = data[0]
        
        # Identify row index corresponding to the 1 entry
        digits = data[1]
        row_idx = np.where(digits == 1)[0][0]
        digit = row_idx
        
        images[digit].append(pixels)
        
    return images
        
        


def plot_heatmap(input_data, number_to_plot, plot_ten=False):
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    images_list = []
    
    images = input_data[number_to_plot]
                        
    if plot_ten == True:
        images = images[0:10]
    
    for image in images:
        flattened_image = image.reshape(784) / 255.0
        images_list.append(flattened_image)
        
    images_array = np.array(images_list)
    num_images = images_array.shape[0]
    
    fig, ax = plt.subplots(1,1)
    ax.imshow(images_array, cmap='gray', interpolation='nearest', aspect='auto')
    ax.set_title(f'Pixel color values for {num_images} handwritten images of the digit {number_to_plot}')
    ax.set_xlabel('color_value')
    ax.set_xticks([])
    ax.set_ylabel('image')
    
    if plot_ten == True:
        max_idx = images_array.shape[0]
        y_tick_arr = np.arange(0, max_idx, 1)
        ax.set_yticks(y_tick_arr, labels=y_tick_arr+1)