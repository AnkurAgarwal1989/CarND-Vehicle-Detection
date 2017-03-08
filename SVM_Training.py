import pickle
import glob
import time
import numpy as np
from features import extract_features
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

def train_LinearSVC(X_train, y_train, X_val, y_val):
    # Use a linear SVC 
    svc = LinearSVC()
    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_val, y_val), 4))
    # Check the prediction time for a single sample
    t=time.time()
    n_predict = 10
    print('SVC predicts: ', svc.predict(X_val[0:n_predict]))
    print('For these',n_predict, 'labels: ', y_val[0:n_predict])
    t2 = time.time()
    print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')
    return svc

# Divide up into cars and notcars
vehicle_image_dir = 'vehicles/*/*.png'
nonvehicle_image_dir = 'non-vehicles/*/*.png'
cars = []
notcars = []

images = glob.glob(vehicle_image_dir)
for image in images:
    cars.append(image)
    
images = glob.glob(nonvehicle_image_dir)
for image in images:
    notcars.append(image)
#cars = cars[0:50]
#notcars = notcars[0:50]

h_colorspace = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
s_colorspace = 'HSV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
color_scale = 4
t=time.time()
#cars = cars[:100]
#notcars = notcars[:100]
car_features = extract_features(cars, h_cspace=h_colorspace, s_cspace=s_colorspace, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, color_scale=color_scale, feature_vec=True)
notcar_features = extract_features(notcars, h_cspace=h_colorspace, s_cspace=s_colorspace, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, color_scale=color_scale, feature_vec=True)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to extract HOG features...')

# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_val, y_train, y_val = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))

Lin_clf = train_LinearSVC(X_train, y_train, X_val, y_val)

pickle_file = 'Car_NoCar_LinearSVC_1.p'

#Save the trained SVM and the parameters and configs
svc_dict = {}
svc_dict["svc"] = Lin_clf
svc_dict["scaler"] = X_scaler
svc_dict["orientations"] = orient
svc_dict["pix_per_cell"] = pix_per_cell
svc_dict["cell_per_block"] = cell_per_block
svc_dict["h_colorspace"] = h_colorspace
svc_dict["s_colorspace"] = s_colorspace
svc_dict["hog_channel"] = hog_channel
svc_dict["color_scale"] = color_scale
with open(pickle_file, 'w') as f:
    pickle.dump(svc_dict, f)

#To load the trained classifier:
with open(pickle_file, 'r') as f:
    svc_dict = pickle.load(f)
    print('Classifier saved to ', pickle_file)
    print(svc_dict["svc"].coef_)
    print(svc_dict["svc"])
