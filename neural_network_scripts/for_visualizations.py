def get_and_categorize_images(input_data):
    
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