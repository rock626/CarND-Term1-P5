**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/Ex_car_notcar.png
[image2]: ./output_images/Ex_Car_hog.png
[image2]: ./output_images/Ex_NotCar_hog.png
[image4]: ./output_images/Sliding_window.png
[image5]: ./output_images/heatmap.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

###Histogram of Oriented Gradients (HOG)

The code for this step is contained in defined in `get_hog_features` (2nd code cell of the IPython notebook ).Also combining with color and spatial features, its defined in method `extract_features` (7th code cell).

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is a Car and Not Car example using the Gray image and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]
![alt text][image3]

####2.HOG parameters.

Initially i tried using RGB colorspace and spatial size of (16,16) and other parameters `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` , from classroom.Using these didnt actually detect white cars from test images.So i tried different combinations like HSL and other colorspaces, but ended with `YCrCb`.

####3. Classifier with HOG parameters.

A linear SVM is used to train the images by extracting features (code cell 9).To extract features, hog parameters are used along with spatial bins and color classificaiton.Even though , accuracy was above 99% for RGB, it failed to detect all types of car objects especially white cars.So different combinations of colorspace are experimented such that car objects are detected from test set.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The 'find_cars' only has to extract hog features once and then can be sub-sampled to get all of its overlaying windows. Each window is defined by a scaling factor of 1.5.I havent experimented with different scales as the initial value gave good result.

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on one scale (1.5) using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./test_videos_output/project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

To remove false positives, I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap for test images, the result of `scipy.ndimage.measurements.label()` and the bounding boxes:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I mostly used the code from the classroom and initially there were issues in combining feature vector of spatial and hog features.With the help of forum and debugging ,i was able to resolve those.The classroom code gave more than 99% accuracy but failed at detecting white cars and there were too many false positives.Then i played around colorspaces and spatial bins.Though changing spatial bins didnt help much but YCrCb colorspace was better.And also i used all channels to extract hog features instead of one channel.

As you can see, the output video is not perfect and smooth.I observed that white car is not detected in some of the frames like when the car is just entering into the frame (in rightmost corner at the bottom).Also i can see some false positive though its only very few frames. I think may be using Decision Trees or Deep learning techinques might perform better than linear SVM.


