from skimage.feature import hog
import cv2
import numpy as np


'''
Function to return HOG (Histogram of Oriented Gradient features for a given image
input: Single channel image
output: (if vis True): Array of features, output image
        (if vis False): Vector of features
'''

#function to convert color space of image from RGB to any other.
#Warning: Initial ColorSpace must be RGB
def convert_cspace(img, cspace='RGB'):
    if cspace != 'RGB':
        if cspace == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif cspace == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif cspace == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif cspace == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif cspace == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: cspace = np.copy(img)
    return feature_image

#aux function to calc hog features for a given channel
def calc_hog_features_(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    if vis == True:
        hog_features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
                                  visualise=True, feature_vector=False)
        return hog_features, hog_image
    else:      
        hog_features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
                       visualise=False, feature_vector=feature_vec)
        return hog_features, None

# If vis is True, returns an image for visualization and unrolled vectors
def calc_hog_features(img, orient, pix_per_cell, cell_per_block, channel='ALL', cspace='RGB', vis=False, feature_vec=True):
    # Convert image to new color space (if specified)
    feature_image = convert_cspace(img, cspace)
    hog_features=[]
    hog_images=[]
    if channel == 'ALL':
        for channel in range(feature_image.shape[2]):
            hog_f, hog_img = calc_hog_features_(feature_image[:,:,channel], 
            orient, pix_per_cell, cell_per_block, vis, feature_vec)
            hog_features.append(hog_f)
            hog_images.append(hog_img)       
    else:
        hog_features, hog_images = calc_hog_features(feature_image[:,:,channel], orient, 
        pix_per_cell, cell_per_block, vis, feature_vec)
    hog_features = np.ravel(hog_features)                    
    return hog_features, hog_images

    
# Function to Spatially bin the pixel values in an image. Returns a single vector        
def calc_bin_spatial_features(img, cspace='RGB', size=(32, 32)):
    # Convert image to new color space (if specified)
    feature_img = convert_cspace(img, cspace)
    features = cv2.resize(feature_img, size).ravel()
    return features
        
        
# Function to extract features from a list of images
#We are using the Spatially binned color features and HOG features
#Color Space for HOG: h_cspace
#Color Space for SB: s_cspace
def extract_features(imgs, h_cspace='RGB', s_cspace='RGB', orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = cv2.imread(file)    

        # Call calc_hog_features() with vis=False, feature_vec=True
        hog_features,_ = calc_hog_features(image, orient, pix_per_cell, cell_per_block, channel=hog_channel, cspace=h_cspace, vis=False, feature_vec=True)
        
        bin_spatial_features = calc_bin_spatial_features(image, cspace=s_cspace, size=(32, 32))
                
        # Append the new feature vector to the features list
        features.append(np.hstack((hog_features, bin_spatial_features)))
    # Return list of feature vectors
    return features