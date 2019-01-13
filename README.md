# CV_AdvancedLaneDetection
Computer Vision Algorithm for Advanced Lane Detection

**Advanced Lane Finding Project**

The goals of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

### Camera Calibration

The code for this step is contained in the `calibrate_camera(debug)` function declared in "laneLineDetection.py".
First of all, I check all images in the "camera_cal" folder, if a chessboard of dimensions 6x9 is detected using the `cv2.findChessboardCorners` function of OpenCV. If a chessboard is successfully detected. 
The array `objpoints` includes the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.
`corners` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  
After processing all calibration images, calibration ist performed using the `cv2.calibrateCamera()` function, with `objectpoints_list` and `corners_list` as parameter to compute the camera calibration and distortion coefficients.  I applied this distortion correction to the all calibration images using the `cv2.undistort()` function. The resulting images are saved in `output_images/calibration_results` folder. 
Here is the function I used for calibration:
```python
def calibrate_camera(debug,save):
#Calibrate Cameras
#Patternparameters
nx = 9 #number of inside corners in x
ny = 6 #number of inside corners in y

image_list = []
ret_list = []
corners_list = []
objectpoints_list = []

#Generate Objectpoints
objpoints = np.zeros((nx*ny,3), np.float32)
objpoints[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

#Detect pattern points in image
for filename in glob.glob('camera_cal/*.jpg'):
im=cv2.imread(filename)
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
image_list.append(im)
ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)
if ret==True:
if debug==True:
print("Chessboard detected")
ret_list.append(ret)
corners_list.append(corners)
objectpoints_list.append(objpoints)
else:
if debug==True:
print("Chessboard not detected")

#Perform calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objectpoints_list, corners_list, gray.shape[::-1], None, None)

if save==True:
for filename in glob.glob('camera_cal/*.jpg'):
im=cv2.imread(filename)
undist = cv2.undistort(im, mtx, dist, None, mtx)
outputname = os.path.dirname(os.path.realpath(__file__))+"/output_images/calibration_results/"+filename.split("/")[1].split(".")[0]+"_undist.jpg"
print(outputname)
cv2.imwrite(outputname,undist)

if debug==True:
print(mtx)
print(dist)
return mtx,dist
```

### Pipeline (single images)
All steps of the pipeline are saved as images to the `output_images/` folder.
#### 1. Example of a distortion-corrected image.
Distrotion correction has been applied in line 100 as first step of the processing pipeline by the following code using the parameters from the calibration step:
```python
undist = cv2.undistort(image, mtx, dist, None, mtx)
```
Here is an image after undistortion has been applied
![](output_images/straight_lines2_1undist.jpg)

#### 2. Color transforms, gradients and other methods to create a thresholded binary image. 
After masking the unneeded top region of the image including the sky,
I used a combination of color and gradient thresholds to generate a binary image.
```python
#Color transforms
hls = cv2.cvtColor(undist, cv2.COLOR_RGB2HLS)
h_channel = hls[:,:,0]
l_channel = hls[:,:,1]
s_channel = hls[:,:,2]
gray = cv2.cvtColor(undist, cv2.COLOR_RGB2GRAY)
#Gradient - Sobel X as presebted in the lecture
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)             # Take the derivative in x
abs_sobelx = np.absolute(sobelx)                 # Absolute x derivative to accentuate lines away from horizontal
scaled_sobelx = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)             # Take the derivative in x
abs_sobely = np.absolute(sobely)                 # Absolute x derivative to accentuate lines away from horizontal
scaled_sobely = np.uint8(255*abs_sobely/np.max(abs_sobely))


mag_thresh=(40, 255)
sobel = np.sqrt(sobelx ** 2 + sobely ** 2)
scaled_sobel = np.uint8(255*sobel/np.max(sobel))
sbinary = np.zeros_like(scaled_sobel)
sbinary[(scaled_sobel > mag_thresh[0]) & (scaled_sobel < mag_thresh[1])] = 1

#Gradient Threshhold as presented in the lecture
thresh_min = 20
thresh_max = 100
sxbinary = np.zeros_like(scaled_sobelx)
sxbinary[(scaled_sobelx >= thresh_min) & (scaled_sobelx <= thresh_max)] = 1
# Threshold h channel - yellow
h_thresh_min = 170
h_thresh_max = 255
h_binary = np.zeros_like(h_channel)
h_binary[(h_channel >= h_thresh_min) & (h_channel <= h_thresh_max)] = 1
# Threshold s channel
s_thresh_min = 170
s_thresh_max = 255
s_binary = np.zeros_like(s_channel)
s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1
# Combine both binary images
combined_binary = np.zeros_like(sbinary)
combined_binary[(s_binary == 1) | (sbinary == 1)] = 1
undist_filtered = cv2.bitwise_and(undist,undist,mask = combined_binary)
```
Even if I extracted the h and the s channel from the hsl image and computed sobel for the x and y direction, I ended up only using the magnitude threshhold for sobel and the s channel
Here's an example of my output for this step.
![](output_images/straight_lines2_3colorandgradient.jpg) 

#### 3. Perspective transform

The code for my perspective transform includes a function called `detect_lanelines_video(image)`. Inside this function the perspective transformation is performed using hardcoded points which are extracted manually from the straight line images. I chosed the points by selecting points on the lane which open a plane rectangle on the street. Here is the code I used inside the `detect_lanelines_video(image)` function to transform the images:

```python
offsetx = 200 #different offset for x and y may be considered to determine curvature
offsety = 0
src = np.float32([(585,458),(706,459),(1158,715),(264,715)])
dst = np.float32([[offsetx, offsety], [img_size[0]-offsetx, offsety], [img_size[0]-offsetx, img_size[1]-offsety], [offsetx, img_size[1]-offsety]])

M = cv2.getPerspectiveTransform(src, dst)
M_inv = cv2.getPerspectiveTransform(dst, src)

transformed = cv2.warpPerspective(undist_filtered, M, img_size)
binary_warped = cv2.warpPerspective(combined_binary, M, img_size)
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585,458      | offsetx, offsety        | 
| 706,459      | img_size[0]-offsetx, offsety      |
| 1158,715    | img_size[0]-offsetx, img_size[1]-offsety      |
| 264,715      | offsetx, img_size[1]-offsety        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![](output_images/straight_lines2_4transformed.jpg) 

#### 4. Identification of lane-line pixels and polynomial fitting?

To identify the lane line I used the window approach based on convolutions as presented in the lecture. Additionally I made some changes:
- After detecting a potential lane element, I thresholded the value of the convoluted region to discard elements, which have a maximum, but are not part of the lane line. Therefore I added an array with name `valid_list`.
- Using this `valid_list` I tried to perform a linear interpolation to add missing element.
- If this was not possible, I discarded the element.

Here is the modified window_centroids function:
```python
def find_window_centroids(binary_warped, window_width, window_height, margin, thresh):
window_centroids = []
valid_list = []
window = np.ones(window_width)
#Reference value
r_value = 0
l_value = 0
####
# Find the starting point - centre of gravity in the bottom quarter of the left and right lane
####
#Sum up the the left half of the image and the bottom quarter
l_sum = np.sum(binary_warped[int(3*binary_warped.shape[0]/4):,:int(binary_warped.shape[1]/2)], axis=0)
#Find the maximum Correct offset which results from kernel size
l_index = np.argmax(np.convolve(window,l_sum))
l_value = np.convolve(window,l_sum)[l_index]
l_center = l_index-window_width/2
#Repeat procedure for right half of the image and the bottom quarter
r_sum = np.sum(binary_warped[int(3*binary_warped.shape[0]/4):,int(binary_warped.shape[1]/2):], axis=0)
r_index = np.argmax(np.convolve(window,r_sum))
r_value = np.convolve(window,r_sum)[r_index]
r_center = r_index-window_width/2+int(binary_warped.shape[1]/2)
#Add points
window_centroids.append((l_center,r_center))
valid_list.append((True,True))

####
#Might be wrong if there is no lane marking in the bottom quarter, but expect this to be true for the testdataset
####

for level in range(1,(int)(binary_warped.shape[0]/window_height)):
r_valid = False
l_valid = False
# convolve the window into the vertical slice of the image -y,x
image_layer = np.sum(binary_warped[int(binary_warped.shape[0]-(level+1)*window_height):int(binary_warped.shape[0]-level*window_height),:], axis=0)
conv_signal = np.convolve(window, image_layer)
# Find the best left centroid by using past left center as a reference
# Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
offset = window_width/2
#Use last l_center which has been set during the last iteration
l_min_index = int(max(l_center+offset-margin,0))
l_max_index = int(min(l_center+offset+margin,binary_warped.shape[1]))
l_index = np.argmax(conv_signal[l_min_index:l_max_index])
l_center_temp = l_index+l_min_index-offset
#print("Value",conv_signal[l_min_index:l_max_index][l_index])
if conv_signal[l_min_index:l_max_index][l_index]>thresh:
l_center = l_center_temp
l_valid = True
#Compute cost function including distance and weight

# Find the best right centroid by using past right center as a reference
r_min_index = int(max(r_center+offset-margin,0))
r_max_index = int(min(r_center+offset+margin,binary_warped.shape[1]))
r_index = np.argmax(conv_signal[r_min_index:r_max_index])
r_center_temp = r_index+r_min_index-offset
if conv_signal[r_min_index:r_max_index][r_index]>thresh:
r_center = r_center_temp
r_valid = True
# Add what we found for that layer
window_centroids.append((l_center,r_center))
valid_list.append((l_valid,r_valid))

print(window_centroids)
#Repair untrusty results
for index in range(0,len(valid_list)):
if valid_list[index][0]==False:
if (index+1)<len(valid_list) and (index-1)>=0:
if valid_list[index-1][0]==True and valid_list[index+1][0]==True:
print("Before:",window_centroids[index])
window_centroids[index] = ((window_centroids[index-1][0]+window_centroids[index+1][0])/2,window_centroids[index][1])
print("After:",window_centroids[index])
valid_list[index]= (True,valid_list[index][1])
if valid_list[index][1]==False:
if (index+1)<len(valid_list) and (index-1)>=0:
if valid_list[index-1][1]==True and valid_list[index+1][1]==True:
print("Before:",window_centroids[index])
window_centroids[index] = (window_centroids[index][0],(window_centroids[index-1][1]+window_centroids[index+1][1])/2)
print("After:",window_centroids[index])
valid_list[index]= (valid_list[index][0],True)
return window_centroids, valid_list
```
Before I fitted the polynom, I only kept valid lane parts:
```python
# Go through each level and draw the windows     
for level in range(0,len(window_centroids)):
# Window_mask is a function to draw window areas
if valid_list[level][0]==True:
l_mask = window_mask(window_width,window_height,binary_warped,window_centroids[level][0],level)
l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
if valid_list[level][1]==True:
r_mask = window_mask(window_width,window_height,binary_warped,window_centroids[level][1],level)
r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255
```

Finally I fitted the polynom:
```python
leftx = []
lefty = []
rightx = []
righty = []
# Fit new polynomials to x,y in world space
for posx in range(l_points.shape[0]):
for posy in range(l_points.shape[1]):
if l_points[posx,posy]==255:
leftx.append(posx*xm_per_pix)
lefty.append(posy*ym_per_pix)

for posx in range(r_points.shape[0]):
for posy in range(r_points.shape[1]):
if r_points[posx,posy]==255:
rightx.append(posx*xm_per_pix)
righty.append(posy*ym_per_pix)

left_fit_cr = np.polyfit(lefty, leftx, 2)
right_fit_cr = np.polyfit(righty, rightx, 2)
# Calculate the new radii of curvature
left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
# Now our radius of curvature is in meters
print(left_curverad, 'm', right_curverad, 'm')
curvature = (left_curverad+right_curverad)/2
print(curvature, 'm')

```

The final result is shown in the following Figure:
![](output_images/straight_lines2_6polyfit.jpg) 

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I computed the radius of the curvature according to the lesson:

```python
ym_per_pix = 30/720
xm_per_pix = 3.7/700

leftx = []
lefty = []
rightx = []
righty = []
# Fit new polynomials to x,y in world space
for posx in range(l_points.shape[0]):
for posy in range(l_points.shape[1]):
if l_points[posx,posy]==255:
leftx.append(posx*xm_per_pix)
lefty.append(posy*ym_per_pix)

for posx in range(r_points.shape[0]):
for posy in range(r_points.shape[1]):
if r_points[posx,posy]==255:
rightx.append(posx*xm_per_pix)
righty.append(posy*ym_per_pix)

left_fit_cr = np.polyfit(lefty, leftx, 2)
right_fit_cr = np.polyfit(righty, rightx, 2)
# Calculate the new radii of curvature
left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
# Now our radius of curvature is in meters
print(left_curverad, 'm', right_curverad, 'm')
curvature = (left_curverad+right_curverad)/2
print(curvature, 'm')
```
For the final radius, I computed the mean of the two curvatures of the left and right lane

#### 6. Result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step using the following lines of code:
```python
#Plot
# Create an image to draw the lines on
warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

# Recast the x and y points into usable format for cv2.fillPoly()
pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
pts = np.hstack((pts_left, pts_right))

# Draw the lane onto the warped blank image
cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

# Warp the blank back to original image space using inverse perspective matrix (Minv)
newwarp = cv2.warpPerspective(color_warp, M_inv, (undist.shape[1], undist.shape[0])) 
# Combine the result with the original image
result = cv2.addWeighted(undist_orig, 1, newwarp, 0.3, 0)
```
![](output_images/test6_7result.jpg) 

Afterwards, I added the text for the position and radius:
```python
cv2.putText(result, "Radius of Curvature "+str(curvature)+" (m)", (int(200), int(100)), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
cv2.putText(result, "Vehicle is "+str(x_centeroffset_m)+" (m) left of center", (int(200), int(150)), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
```
---

### Pipeline (video)

#### 1. Final video output of my pipeline.

Here's a [link to my video result](https://youtu.be/LUkL1XKpqoc)

---

### Discussion

#### 1. Problems and Limitations

The implemented approach is very close to the methods presented in the lecture. To make the approach more robust, I need to implement a tracking procedure, which tracks the line across a sequence of images. This would make it easy to identify and bypass wrong detections. Another step would be to improve the filtering stage further. At the moment, only the s channel is considered, but the h channel probably has also useful informations. Combining different filters and make a case decision could also improfe the performace.
