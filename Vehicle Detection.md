##Vehicle Detection

### This write-up describes a Computer Vision based approach for detecting and tracking vehicles in view of a camera mounted in a car. An SVM classifier is trained to detect cars in view and a tracker is used to draw bounding boxes.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Train a Linear SVM classifier using Histogram of Oriented Gradients (HOG) features and other color features using a labeled training set of images.
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Create a heat map of recurring detections to reject outliers and follow detected vehicles in a video stream.
* Estimate a bounding box for vehicles detected.

[//]: # "Image References"
[hog_car]: ./output_images/HOG_car.jpg
[hog_no_car]: ./output_images/HOG_no_car.jpg
[scales]: ./output_images/scales.jpg
[c_sw_1]: ./output_images/coarse_sliding_windows_left.jpg
[c_sw_2]: ./output_images/coarse_sliding_windows_center.jpg
[c_sw_3]: ./output_images/coarse_sliding_windows_right.jpg
[f_sw_1]: ./output_images/fine_sliding_windows_1.jpg
[f_sw_2]: ./output_images/fine_sliding_windows_2.jpg
[heat_1]: ./output_images/heat_1.jpg
[heat_2]: ./output_images/heat_2.jpg
[bbox]: ./output_images/output_bboxes.jpg
[test_video]:  ./test_video.mp4
[video1]: ./out_proj.mp4

## Section I

###Code Layout
- **features.py**: File contains functions to extract HOG and color features. These functions are used by the detection code.
- **SVM_Training.py**: File contains code to train and save an SVM classifier. The classifier is saved out as a pickle (.p) file.
- **Car_NoCar_LinearSVC.p**: Pickle file containing the SVM classifier.
- **CarTracker.py**: Class; Tracker objects. Maintains and updates state of the tracked car. The *update* function here can be modified with a smarter tracking algorithm.
- **detectCars.py**: Main file to run detection. Also contains functions to generate sliding windows.




## Section II

### Features

- The code for feature extraction can be found in `extract_features`, `calc_bin_spatial_features` and `calc_hog_features` function in **features.py**.
- 2 types of features have been used for describing cars.
  - HOG Features
  - Color Spatial features
- 64x64 pixel patches are used for extraction.

##### Histogram of Oriented Gradients (HOG)

- The code for this can be found in the `calc_hog_features` function in **features.py**.
- The choice of color space and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`) was done with some trial and error. Since, HOG deals with gradients, it is generally more robust to color spaces.
- We can not make this vector too long as it adversely affects the processing time.
- The `YUV color space` and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` were finally decided on. Here is an example of HOG features on the sample images:
!['HOG Features Car'][hog_car]  !['HOG Features No Car'][hog_no_car]

##### Color Spatial Features

- The code can be found in  `calc_bin_spatial_features` function in **features.py**.
- Vector of `HSV colorspace`. The 64x64 patch is scaled down by 2 and all channel values are flattened into a 1D vector.



### SVM Classifier

- The code for this can be found in **SVM_Training.py**.
- The SVM with the Linear Kernel is trained on all the data form the GTI and KITTI dataset.
- A `scaler` is also fit on the data to be used during prediction.
- The classifier is saved as a pickle file.




### Multi Scale Search:

- From the data we can make an assumption that the cars must fit inside a 64x64 size image patch.
- If cars are closer to the camera, they will be larger and the patch will need to be scaled down.
- If cars are further away, towards the vanishing point, the image patch will need to be scaled up.
- By looking at some data frames and identifying the max and minimum size of the cars, I fit a linear function to identify scales depending on patch location in image frame.
- The code can be found at `xy_to_scale` in **detectCars.py**. The scale at vanishing point is 0.0, and at the farthest points is 2.8. A sample is shown. This is a continuous function; discrete values shown for illustration.!['Multi Scale Search'][scales]




### Sliding Window Search

A sliding window search method is used to detect cars. 2 stages are used.

- Coarse Search Windows:

  - The code can be found at `coarse_detection` in **detectCars.py**.

  - We identify 3 areas where cars can be found. The coarse windows are done at 2 scales for each area. These values have been derived by using the function explained above.

  - The overlap used for the larger areas is 50% and for the smaller central area is 25%.

  - We are very permissive and do not reject outliers here.

  - This coarse detection is also run every 3^rd^ frame.

  - A visualization is shown. 

    |           Left Windows           |           Center Windows           |           Right Windows           |
    | :------------------------------: | :--------------------------------: | :-------------------------------: |
    | !['Coarse Windows Left'][c_sw_1] | !['Coarse Windows Center'][c_sw_2] | !['Coarse Windows Right'][c_sw_3] |

  ​

- Fine Search Windows:

  - The code can be found at `fine_detection` in **detectCars.py**.
  - Once we have a putative match from the Coarse Search Windows, we do a more refined search.
  - Sliding windows are generated around the area using the centroid. The scale is calculated according to the function above.
  - This stage has more restrictive rejection policy. The vehicle needs to be detected in at least 5 windows in the search patch. This detection is run every frame for every tracked object and new putative matches.
  - A visualization is shown.!['Fine Windows 1'][f_sw_1]




#### Performance Optimization

- The final classifier was selected on basis of the training and validation test results.
- For a search region, HOG features are only calculated once. Feature vectors from the patches are then assembled into one large matrix. Predictions are calculated as one matrix multiply.




## Section III

#### Video Implementation

- Here's a [link to my video result](./out_proj.mp4)
- The bounding boxes are smooth and consistent.


#### Filtering and False Positives

- For filtering, we apply a threshold of 5 on the heat map in the fine search. We do not threshold in the coarse search as this allows for consistent detection.
- False positives are culled in the Fine Search.
- Every time a new centroid is found by the Coarse Search, we run a Fine Search. If it is deemed to be a vehicle, we check back with the existing tracked objects. If the new centroid is too close to an existing one, it probably is a redundant overlap and is rejected. If not, it is added to the list of tracked objects.
- The low-pass filter can be found in `update` in the **CarTracker** class.
- For each tracked object, we run the Fine Search independently. This automatically takes care of overlapping objects.
- `scipy.ndimage.measurements.label()`  is used to identify individual blobs in the heatmap. Our heatmap only has one blob each. Bounding boxes are then constructed around the area of each blob detected.

##### Here are the corresponding heat maps for the frame shown above:

!['Heat Map'][heat_1] !['Heat Map'][heat_2]

##### Here are the resulting bounding boxes:

!['Bounding Boxes'][bbox]



---

###Discussion

- The bounding boxes are smooth and fairly consistent. The results look good considering only a low-pass filter is used.
- The detection and tracking are done in multiple scales, but in constrained windows around an expected region. This helps provide speedups even with a variety of scales and overlaps.
- Each tracked car is assigned a tracker object. The pipeline can be easily multi-threaded as all objects can be tracked independent of each other.
  - This also allows for the system to track objects close to each other without merging them into one large bounding box.
- The coarse window search since independent can also be multi-threaded, saving time.
- The implementation is based on the GTI and KITTI datasets. It is possible that the dataset is not representative of all types of cars and their orientations. This may result in a not so optimal classifier.
- The tracker pipeline could be made more robust by using a Kalman filter or any such tracker/filter. Another approach could be to detect the car and then track it's features using a Lucas-Kanade based methods.
- The pipeline does not take cues from the position on the road as of now. The search windows can be adjusted depending on the vehicles position. If we are very close to the center divider, we do not need to search on the left and so on.
- I have vectorised as much as I could. All SVM predictions for one ROI are happening as a matrix multiply saving time, but that is not enough to make the pipeline realtime (currently ~5fps).
- The HOG feature extraction is currently the bottleneck in the pipeline. It takes an order of magnitude more time compared to everything else. More work is required in making this work at 25-30 fps.
