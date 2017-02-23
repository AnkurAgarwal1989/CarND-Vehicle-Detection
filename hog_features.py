from skimage.feature import hog

'''
Function to return HOG (Histogram of Oriented Gradient features for a given image
input: Single channel image
output: (if vis True): Array of features, output image
        (if vis False): Vector of features
'''

# If vis is True, returns an image for visualization and unrolled vectors
def hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
        
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
                                  visualise=True, feature_vector=False)
        return features, hog_image
    else:      
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
                       visualise=False, feature_vector=feature_vec)
        return features
