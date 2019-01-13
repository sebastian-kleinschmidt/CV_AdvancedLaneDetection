import numpy as np
import glob
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
import os

def window_mask(width, height, img_ref, center,level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output

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
    #Might be wrong if there is no lane marking in the bottom quarter
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

def detect_lanelines_video(image):
    return detect_lanelines(image,mtx,dist,False,False,"")

def detect_lanelines(image,mtx,dist,debug,save,filename):
    print("Processing "+filename)
    #Undistort image
    undist = cv2.undistort(image, mtx, dist, None, mtx)
    img_size = (undist.shape[1], undist.shape[0])

    if debug==True:
        cv2.imshow('img',undist)
        cv2.waitKey(2000)
    if save==True:
        outputname = "output_images/"+filename.split(".")[0]+"_1undist.jpg"
        cv2.imwrite(outputname,undist)

    #Mask Sky and dashboard with polygon
    mask = np.zeros(undist.shape, dtype=np.uint8)
    roi_corners = np.array([[(0,430),(undist.shape[1],430),(undist.shape[1],undist.shape[0]),(0,undist.shape[0])]], dtype=np.int32)
    channel_count = undist.shape[2]
    ignore_mask_color = (255,)*channel_count
    cv2.fillPoly(mask, roi_corners,ignore_mask_color)

    undist_orig = undist
    kernel_size = 5
    undist = cv2.GaussianBlur(undist, (kernel_size, kernel_size), 0)
    undist = cv2.bitwise_and(undist,mask)
    if debug==True:
        cv2.imshow('img',undist)
        cv2.waitKey(2000)
    if save==True:
        outputname = "output_images/"+filename.split(".")[0]+"_2mask.jpg"
        cv2.imwrite(outputname,undist)
    #Color transforms
    hls = cv2.cvtColor(undist, cv2.COLOR_RGB2HLS)
    h_channel = hls[:,:,0]
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    gray = cv2.cvtColor(undist, cv2.COLOR_RGB2GRAY)
    #Gradient - Sobel X as presebted in the lecture
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) 			# Take the derivative in x
    abs_sobelx = np.absolute(sobelx) 				# Absolute x derivative to accentuate lines away from horizontal
    scaled_sobelx = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1) 			# Take the derivative in x
    abs_sobely = np.absolute(sobely) 				# Absolute x derivative to accentuate lines away from horizontal
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
    if debug==True:
        cv2.imshow('img',undist_filtered)
        cv2.waitKey(2000)
    if save==True:
        outputname = "output_images/"+filename.split(".")[0]+"_3colorandgradient.jpg"
        cv2.imwrite(outputname,undist_filtered)
    #Perspective transformation
    offsetx = 200 #different offset for x and y may be considered to determine curvature
    offsety = 0
    src = np.float32([(585,458),(706,459),(1158,715),(264,715)])
    dst = np.float32([[offsetx, offsety], [img_size[0]-offsetx, offsety], [img_size[0]-offsetx, img_size[1]-offsety], [offsetx, img_size[1]-offsety]])

    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)

    transformed = cv2.warpPerspective(undist_filtered, M, img_size)
    binary_warped = cv2.warpPerspective(combined_binary, M, img_size)

    if debug==True:
        cv2.imshow('img',transformed)
        cv2.waitKey(2000)
    if save==True:
        outputname = "output_images/"+filename.split(".")[0]+"_4transformed.jpg"
        cv2.imwrite(outputname,transformed)
    #Sliding Window - approach as shown in the lecture
    window_width = 50 
    window_height = 40#80
    margin = 100
    margin = 50

    window_centroids, valid_list = find_window_centroids(binary_warped, window_width, window_height, margin, 500)
    if len(window_centroids) > 0:
        # Points used to draw all the left and right windows
        l_points = np.zeros_like(binary_warped)
        r_points = np.zeros_like(binary_warped)
    
        # Go through each level and draw the windows 	
        for level in range(0,len(window_centroids)):
            # Window_mask is a function to draw window areas
            if valid_list[level][0]==True:
            	l_mask = window_mask(window_width,window_height,binary_warped,window_centroids[level][0],level)
            	l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
            if valid_list[level][1]==True:
                r_mask = window_mask(window_width,window_height,binary_warped,window_centroids[level][1],level)
                r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255
    
    template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
    zero_channel = np.zeros_like(template) # create a zero color channel
    template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
    warpage = np.array(cv2.merge((binary_warped,binary_warped,binary_warped)),np.uint8) # making the original road pixels 3 color channels
    output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the orignal road image with window results
    
    # If no window centers found, just display orginal road image
    if len(window_centroids) == 0:
        output = np.array(cv2.merge((binary_warped,binary_warped,binary_warped)),np.uint8)

    if debug==True:
        cv2.imshow('img',output)
        cv2.waitKey(2000)

    if save==True:
        outputname = "output_images/"+filename.split(".")[0]+"_5transformedandfilter.jpg"
        cv2.imwrite(outputname,output)

    #Estimate curvature - fit polynom
    #ploty = np.linspace(0, img_size[1]-1, num=img_size[1])
    ploty = np.linspace(0, 719, num=720)

    leftx = []
    lefty = []
    rightx = []
    righty = []

    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    for posx in range(l_points.shape[0]):
        for posy in range(l_points.shape[1]):
            if l_points[posx,posy]==255:
                leftx.append(posx)
                lefty.append(posy)

    for posx in range(r_points.shape[0]):
        for posy in range(r_points.shape[1]):
            if r_points[posx,posy]==255:
                rightx.append(posx)
                righty.append(posy)

    left_fit = np.polyfit(leftx, lefty, 2)
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fit = np.polyfit(rightx, righty, 2)
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    if debug==True:
        points_leftpolynoms = []
        points_rightpolynoms = []
        
        for y in range(output.shape[1]):
            points_leftpolynoms.append([(((left_fit[0]*(y)**2 + left_fit[1]*(y) + left_fit[2])),y)])
            points_rightpolynoms.append([(((right_fit[0]*(y)**2 + right_fit[1]*(y) + right_fit[2])),y)])

        points_leftpolynoms = np.array(points_leftpolynoms,dtype=np.int32)
        points_rightpolynoms = np.array(points_rightpolynoms,dtype=np.int32)

        for point in points_leftpolynoms:
            cv2.circle(output, (point[0][0],point[0][1]), 10, (255,255,255))
        for point in points_rightpolynoms:
            cv2.circle(output, (point[0][0],point[0][1]), 10, (255,255,255))
        cv2.imshow('img',output)
        cv2.waitKey(2000)

        if save==True:
            outputname = "output_images/"+filename.split(".")[0]+"_6polyfit.jpg"
            cv2.imwrite(outputname,output)

    #Determine Center
    y_bottom = img_size[1]-1
    left_lane = (((left_fit[0]*(y_bottom)**2 + left_fit[1]*(y_bottom) + left_fit[2])))
    right_lane = (((right_fit[0]*(y_bottom)**2 + right_fit[1]*(y_bottom) + right_fit[2])))
    x_camera_center = mtx[0,2]
    x_car = left_lane+(right_lane-left_lane)/2
    print('camera center',x_camera_center)
    print('car center',x_car)
    x_centeroffset_px = x_camera_center-x_car
    x_centeroffset_m = x_centeroffset_px*xm_per_pix
    print('car center offset [px]', x_centeroffset_px)
    print('car center offset [m]', x_centeroffset_m)

    #Compute Radius
    y_eval = np.max(img_size[1])
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    print(left_curverad, 'px', right_curverad, 'px')

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

    if save==True:
        outputname = "output_images/"+filename.split(".")[0]+"_7result.jpg"
        cv2.imwrite(outputname,result)

    cv2.putText(result, "Radius of Curvature "+str(curvature)+" (m)", (int(200), int(100)), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
    cv2.putText(result, "Vehicle is "+str(x_centeroffset_m)+" (m) left of center", (int(200), int(150)), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)

    return result

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



debug = True
#Calibrate
mtx, dist = calibrate_camera(debug,True)
#Process images
for filename in glob.glob('test_images/*.jpg'):
    #Load image
    im=cv2.imread(filename)
    result = detect_lanelines(im,mtx,dist,debug,True,filename.split("/")[1])
    cv2.imshow('img',result)
    cv2.waitKey(5000)
#Process videos
for filename in glob.glob('*.mp4'):
    print("Processing: "+filename)
    white_output = filename
    white_output = "output_videos/"+white_output.split(".")[0]+"_processed.mp4"
    print(white_output)
    clip = VideoFileClip(filename)

    white_clip = clip.fl_image(detect_lanelines_video)
    white_clip.write_videofile(white_output, audio=False)
