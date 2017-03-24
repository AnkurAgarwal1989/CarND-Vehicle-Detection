import cProfile

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import time
from features import calc_hog_features, calc_bin_spatial_features
from skimage.util.shape import view_as_windows
from scipy.ndimage.measurements import label


dist_pickle = pickle.load( open('Car_NoCar_LinearSVC_1.p', 'r' ) )
svc = dist_pickle["svc"]
X_scaler = dist_pickle["scaler"]
orient = dist_pickle["orientations"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
color_scale = dist_pickle["color_scale"]
h_colorspace = dist_pickle["h_colorspace"]
s_colorspace = dist_pickle["s_colorspace"]
hog_channel = dist_pickle["hog_channel"]

#we will be using the same indexing of windows for all images. The data changes, but our indices will not
#Pre make this list. Consumes memory, but saves time
def get_hog_features_array_(img_shape, hog_data, patch_size=64, orient=9, channels=3,
                      pix_per_cell=8, cell_per_block=2, cells_per_step=4):
    cells_per_window = patch_size // pix_per_cell
    
    blocks_per_window = (cells_per_window - cell_per_block) + 1
    #nxcells = img_shape[1] // pix_per_cell
    #nycells = img_shape[0] // pix_per_cell
    
    #nxblocks = (nxcells - cell_per_block) + 1 ##(W-F)/S + 1, W=ncells, F=cell_per_block, S=Stride=1
    #nyblocks = (nycells - cell_per_block) + 1 ##(W-F)/S + 1, W=ncells, F=cell_per_block, S=Stride=1
    
    window_shape = (channels, blocks_per_window, blocks_per_window, cell_per_block, cell_per_block, orient)
    step = (1, cells_per_step, cells_per_step, 1, 1, 1)
    feature_vector_len = (channels * (blocks_per_window**2) * (cell_per_block**2) * orient)
    hog_windows = view_as_windows(hog_data, window_shape, step)
    hog_feature_array = hog_windows.reshape(-1, feature_vector_len)
    #print("Hog feature array shape: ", hog_feature_array.shape)
    return hog_feature_array

#Input is an image scaled by 0.5
def get_color_spatial_features_array_(color_data, patch_size, channels=3, pixels_per_step=16):
    #1 cell = 8 pixels for full scale image. Here we use 0.5 scale.
    window_shape = (patch_size, patch_size, channels)
    feature_vector_len = ((patch_size**2) * channels)
    step = (pixels_per_step, pixels_per_step, 1)
    color_spatial_windows = view_as_windows(color_data, window_shape, step)
    color_spatial_feature_array = color_spatial_windows.reshape(-1, feature_vector_len)
    return color_spatial_feature_array    

def get_feature_array(img_roi, hog_channel, h_cspace, s_cspace, c_scale=4, patch_size=64, orient=9,
                      pix_per_cell=8, cell_per_block=2, cells_per_step=4):
    
    hog_data = calc_hog_features(img_roi, orient, pix_per_cell, cell_per_block, 
                                 channel=hog_channel, cspace=h_cspace, vis=False, feature_vec=False)
    if hog_channel=='ALL':
        hog_channel=3;
    hog_feature_array = get_hog_features_array_(img_roi.shape, hog_data, patch_size, orient, hog_channel, 
                                          pix_per_cell, cell_per_block, cells_per_step)
    
    pixels_per_step = pix_per_cell*cells_per_step/c_scale
    color_data = calc_bin_spatial_features(img_roi, cspace=s_cspace, scale=c_scale, feature_vec=False)
    color_spatial_feature_array = get_color_spatial_features_array_(color_data, patch_size=patch_size/c_scale,
                                                              channels=3, pixels_per_step=pixels_per_step)
    #feature_array = np.array(hog_feature_array)
    feature_array = np.hstack((hog_feature_array, color_spatial_feature_array))
    #print("Entire feature array ", feature_array.shape)
    return feature_array

def get_image_windows(img, img_scale, xstart, xend, ystart, yend, window, pix_per_step): 
    nxsteps = ((xend - xstart)/img_scale - window)//pix_per_step + 1;
    nysteps = ((yend - ystart)/img_scale - window)//pix_per_step + 1;
    #print('Windows ', nxsteps, nysteps)
    
    x_list = np.arange(0, nxsteps)*pix_per_step*img_scale
    y_list = np.arange(0, nysteps)*pix_per_step*img_scale
    x_list = x_list.astype(int)
    y_list = y_list.astype(int)
    x_list = x_list + xstart
    y_list = y_list + ystart
    
    draw_window_size = int(window*img_scale)
    return y_list, x_list, draw_window_size

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, xstart, xend, ystart, yend, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, 
              cells_per_step, color_scale=color_scale, hog_channel='ALL'):
    img_roi = img[ystart:yend, xstart:xend, :]
    
    if scale != 1:
        imshape = img_roi.shape
        img_roi = cv2.resize(img_roi, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
    
    feature_array = get_feature_array(img_roi, hog_channel, h_cspace=h_colorspace, s_cspace=s_colorspace, c_scale=color_scale,
          patch_size=64,orient=orient,pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, cells_per_step=cells_per_step)
    
    trans_fa = X_scaler.transform(feature_array)
    car_predictions = svc.predict(trans_fa)
    
    return car_predictions
    #return fa, draw_img


def draw_sliding_windows(img, imgwin_y, imgwin_x, draw_win_size):
    draw_img=np.copy(img)
    for xleft in imgwin_x:
        for ytop in imgwin_y:
            xbox_left = xleft
            ytop_draw = ytop
            color = np.random.randint(0, 256, (3,))
            draw_img = cv2.rectangle(draw_img,(xbox_left, ytop_draw),
                                     (xbox_left+draw_win_size, ytop_draw+draw_win_size),color,6)
    return draw_img

#Gicen a set of predictions, plot a box around Car predictions
#Returs a set of car windows
def draw_car_windows(img, predictions, imgwin_y, imgwin_x, draw_win_size):
    car_pred = predictions.nonzero()[0] ##Non zeros values are the cars
    draw_img = np.copy(img)
    num_cols = imgwin_x.shape[0]
    #num_rows = imgwin_y.shape[0]
    car_bboxes = []
    #print(num_cols, num_rows)
    for pred in car_pred:
        y_idx = pred // num_cols
        x_idx = pred - (y_idx*num_cols)
        #if y_idx == num_rows:
            #print(pred, num_cols, num_rows)
        xbox_left = imgwin_x[x_idx]
        ytop_draw = imgwin_y[y_idx]
        #print(xbox_left, ytop_draw)
        bbox = ((xbox_left, ytop_draw), (xbox_left + draw_win_size, ytop_draw + draw_win_size))
        car_bboxes.append(bbox)
        #color = np.random.randint(0, 256, (3,))
        draw_img = cv2.rectangle(draw_img,(xbox_left, ytop_draw),
                      (xbox_left + draw_win_size, ytop_draw + draw_win_size),(0,0,255),6) 
        
    return car_bboxes, draw_img

def find_cars_scale(img, image_scale, xstart, xend, ystart, yend, cells_per_step, color_scale, hog_channel):
    predictions = find_cars(img, xstart, xend, ystart, yend, image_scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, 
                                     cells_per_step=cells_per_step, color_scale=color_scale, hog_channel=hog_channel)
    return predictions

def run_detection(img, xstart, xend, ystart, yend, image_scale, window, pix_per_cell, cells_per_step, draw_debug=False):
    imgwin_y, imgwin_x, draw_win_size = get_image_windows(img, image_scale, xstart, xend, ystart, yend, 64, pix_per_cell*cells_per_step)
    #t1 = time.time()
    p = find_cars_scale(img, image_scale, xstart, xend, ystart, yend, cells_per_step, color_scale, hog_channel)
    #print("time taken", time.time()-t1)
    car_bboxes, draw_img = draw_car_windows(img, p, imgwin_y, imgwin_x, draw_win_size)
    if draw_debug:
        draw_img = draw_sliding_windows(img, imgwin_y, imgwin_x, draw_win_size)
    return car_bboxes, draw_img

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    #List of centroids of detected cars
    detected_centroid = []
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
        detected_centroid.append(((np.min(nonzerox) + np.max(nonzerox))/2, (np.min(nonzeroy) + np.max(nonzeroy))/2))
    # Return the image
    return img, detected_centroid

def coarse_detection(img, coarse_windows):
    window = 64
    heat = np.zeros(img[:,:,0].shape, dtype=np.float)
    for i, sw_cfg in enumerate(coarse_windows):
        car_bboxes, debug_img = run_detection(img, sw_cfg['xstart'], sw_cfg['xend'], sw_cfg['ystart'], sw_cfg['yend'],
                                             sw_cfg['scale'], window, pix_per_cell, sw_cfg['cell_step'], draw_debug=True)
        #plt.imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
        #plt.title('Car Positions debug')
        #plt.show()
        # Add heat to each box in box list
        heat = add_heat(heat,car_bboxes)
        #plt.subplot(1, 6, i+1)
        #plt.imshow(cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB))
    
    #plt.show()
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat,0)

    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img, centroids = draw_labeled_bboxes(np.copy(img), labels)
    #print(labels[1], centroids)
    return draw_img, heatmap, centroids, debug_img

#scale varoes with distance from center
def xy_to_scale(x,y):
    dist_from_center = np.sqrt((x - 1280/2)**2 + (y - 720/2)**2)
    min_dist = 0
    max_dist = np.sqrt((1279 - 1280/2)**2 + (719 - 720/2)**2)
    scale_min = 0.00
    scale_max = 2.80 #3
    scale = ((dist_from_center - min_dist)*(scale_max - scale_min)/(max_dist - min_dist)) + scale_min
    return scale

def fine_search_windows(centroids):
    fine_windows = []
    search_window_size = 64
    scale_d = [-0.10, -0.05, 0.00, +0.05, +0.10]
    for centroid in centroids:
        #print(centroid)
        search_scale = round(xy_to_scale(centroid[0], centroid[1]), 2)*1.00
        #move on if we are looking too close to center at the vanishing point
        if search_scale < 0.40:
            continue
        print(search_scale)
        for s_d in scale_d:
            search_scale_temp = round(search_scale+s_d , 2)
            window_size = int(search_window_size*search_scale_temp)
            x_start = max(centroid[0]-window_size, 0)
            x_end = min(centroid[0]+window_size, 1280)
            y_start = max(centroid[1]-window_size, 400)
            y_end = min(centroid[1]+window_size, 600)

            fine_windows.append({'cell_step':1, 'scale':search_scale_temp, 'xstart':x_start, 'xend':x_end, 'ystart':y_start, 'yend':y_end})
    return fine_windows

def fine_detection(img, fine_windows):
    window = 64
    heat_thresh = 3
    #print(fine_windows)
    heat = np.zeros(img[:,:,0].shape, dtype=np.float)
    for i, sw_cfg in enumerate(fine_windows):
        car_bboxes, debug_img = run_detection(img, sw_cfg['xstart'], sw_cfg['xend'], sw_cfg['ystart'], sw_cfg['yend'],
                                             sw_cfg['scale'], window, pix_per_cell, sw_cfg['cell_step'], draw_debug=True)
        
        heat = add_heat(heat,car_bboxes)
        
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, heat_thresh)

    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img, centroids = draw_labeled_bboxes(np.copy(img), labels)
    print(labels[1], centroids)
    return draw_img, heatmap, centroids, debug_img

def run_(img, centroids, frame_cntr):
    motion_thresh_per_frame = 25
    frame_skip = 2
    print("Frame number: ", frame_cntr)
    print("Past centroids", centroids)
    #if first image, run larger_coarse detection
    if centroids is None or len(centroids)==0 or frame_cntr%frame_skip==0:
        print("Running coarse detection")
        draw_img, heatmap, new_centroids, debug_img = coarse_detection(img, coarse_windows)
        if centroids is None:
            centroids = new_centroids
        elif len(centroids) > 0:
            #If we already have a bunch of centroids, reject already tracked centroids from new coarse search
            #append to list only if greater than a certain threshold
            prox_thresh = motion_thresh_per_frame*frame_skip
            for centroid in new_centroids:
                already_tracked = False
                for tracked_centroid in centroids:
                    #print("nearness", abs(centroid[0]-tracked_centroid[0]), abs(centroid[1]-tracked_centroid[1]))
                    if abs(centroid[0]-tracked_centroid[0]) <= prox_thresh and abs(centroid[1]-tracked_centroid[1]) <= prox_thresh:
                        already_tracked = True
                        break
                if not already_tracked:
                    print("Found a new plausible car. Adding to fine tracking")
                    centroids.append(centroid)
        if 0:
            plt.figure(figsize=(20,30))
            plt.subplot(1,3,1)
            plt.imshow(cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB))
            plt.title('Car Positions')
            plt.subplot(1,3,3)
            plt.imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
            plt.title('Car Positions debug')
            plt.subplot(1,3,2)
            plt.imshow(heatmap, cmap='hot')
            plt.title('Heat Map')
            plt.show()
    
    if len(centroids) == 0:
        return img, None
    else:
        print("Running Fine detection with centroids")
        print(centroids)
        fine_windows = fine_search_windows(centroids)
        draw_img, heatmap, centroids, debug_img = fine_detection(img, fine_windows)
        if 0:
            plt.figure(figsize=(20,30))
            plt.subplot(1,3,1)
            plt.imshow(cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB))
            plt.title('Car Positions')
            plt.subplot(1,3,3)
            plt.imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
            plt.title('Car Positions debug')
            plt.subplot(1,3,2)
            plt.imshow(heatmap, cmap='hot')
            plt.title('Heat Map')
            plt.show()
    return draw_img, centroids

coarse_windows = []

coarse_windows.append({'cell_step':4, 'scale':2, 'xstart':150, 'xend':450, 'ystart':400, 'yend':650})
coarse_windows.append({'cell_step':4, 'scale':2, 'xstart':900, 'xend':1280, 'ystart':400, 'yend':650})
##coarse_windows.append({'cell_step':4, 'scale':1.8, 'xstart':150, 'xend':450, 'ystart':400, 'yend':650})
##coarse_windows.append({'cell_step':4, 'scale':1.8, 'xstart':900, 'xend':1280, 'ystart':400, 'yend':650})
coarse_windows.append({'cell_step':4, 'scale':1.5, 'xstart':150, 'xend':450, 'ystart':400, 'yend':650})
coarse_windows.append({'cell_step':4, 'scale':1.5, 'xstart':900, 'xend':1280, 'ystart':400, 'yend':650})

cell_step = 2
#coarse_windows.append({'cell_step':cell_step, 'scale':1.1, 'xstart':450, 'xend':1050, 'ystart':380, 'yend':550})
#coarse_windows.append({'cell_step':cell_step, 'scale':1.3, 'xstart':450, 'xend':1050, 'ystart':380, 'yend':550})
coarse_windows.append({'cell_step':cell_step, 'scale':1.15, 'xstart':450, 'xend':1050, 'ystart':380, 'yend':550})
#coarse_windows.append({'cell_step':cell_step, 'scale':1.1, 'xstart':450, 'xend':1050, 'ystart':380, 'yend':550})
coarse_windows.append({'cell_step':cell_step, 'scale':1.2, 'xstart':450, 'xend':1050, 'ystart':380, 'yend':550})
#coarse_windows.append({'cell_step':cell_step, 'scale':1.35, 'xstart':450, 'xend':1050, 'ystart':380, 'yend':550})
#coarse_windows.append({'cell_step':cell_step, 'scale':1.25, 'xstart':450, 'xend':1050, 'ystart':380, 'yend':550})


cap = cv2.VideoCapture('vid2.mp4')
fourcc = cv2.VideoWriter_fourcc(*'MP4V')#'XVID')
recorder = cv2.VideoWriter('out_vid2.mp4', fourcc, 30, (1280, 720))
t1 = time.time()
cnt = -1
centroids = None
while(cap.isOpened()):
    ret, frame = cap.read()
    cnt += 1
    #if cnt < 250:
    #       continue
        
    if ret == True:
        #print("cnt", cnt)
        
        if cnt%50 == 0:
            print('Time taken ', time.time()-t1)
        print(centroids)
        img_boxes, centroids = run_(frame, centroids, cnt)
        
        cv2.putText(img_boxes, str(cnt), (0,100), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
        recorder.write(img_boxes)
        #if cnt == 750:
            #cv2.imwrite('img1.jpg', frame)
            #break
    else:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print('Time taken ', time.time()-t1)
cap.release()
recorder.release()
cv2.destroyAllWindows()


#img = cv2.imread('test_images/test12.jpg') 
#print(img.shape)
#print(img.dtype)
#
#
#draw_img, heatmap, coarse_centroids, debug_img = coarse_detection(img, coarse_windows)
#plt.figure(figsize=(20,30))
#plt.subplot(1,3,1)
#plt.imshow(cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB))
#plt.title('Car Positions')
#plt.subplot(1,3,3)
#plt.imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
#plt.title('Car Positions debug')
#plt.subplot(1,3,2)
#plt.imshow(heatmap, cmap='hot')
#plt.title('Heat Map')
#plt.show()
#
#fine_windows = fine_search_windows(coarse_centroids)
#draw_img, heatmap, coarse_centroids_, debug_img = fine_detection(img, fine_windows)
#plt.figure(figsize=(20,30))
#plt.subplot(1,3,1)
#plt.imshow(cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB))
#plt.title('Car Positions')
#plt.subplot(1,3,3)
#plt.imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
#plt.title('Car Positions debug')
#plt.subplot(1,3,2)
#plt.imshow(heatmap, cmap='hot')
#plt.title('Heat Map')
#plt.show()