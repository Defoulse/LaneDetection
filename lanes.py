import cv2  # Import the OpenCV library to enable computer vision
import numpy as np  # Import the NumPy scientific computing library
import edge_detection as edge  # Handles the detection of lane lines
import matplotlib.pyplot as plt  # Used for plotting and error checking
from matplotlib.widgets import Cursor, Button
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox as mb


file_size = (1920, 1080)  # Assumes 1920x1080 mp4
scale_ratio = 0.5  # Option to scale to fraction of original size.
output_frames_per_second = 20.0

# Global variables
prev_leftx = None
prev_lefty = None
prev_rightx = None
prev_righty = None
prev_left_fit = []
prev_right_fit = []

prev_leftx2 = None
prev_lefty2 = None
prev_rightx2 = None
prev_righty2 = None
prev_left_fit2 = []
prev_right_fit2 = []


class Lane:
    def __init__(self, orig_frame):
        self.orig_frame = orig_frame

        # This will hold an image with the lane lines
        self.lane_line_markings = None

        # This will hold the image after perspective transformation
        self.warped_frame = None
        self.transformation_matrix = None
        self.inv_transformation_matrix = None

        # (Width, Height) of the original video frame (or image)
        self.orig_image_size = self.orig_frame.shape[::-1][1:]

        width = self.orig_image_size[0]
        height = self.orig_image_size[1]
        self.width = width
        self.height = height

        # Four corners of the trapezoid-shaped region of interest
        # You need to find these corners manually.
        if sTLx != 0 or sTLy != 0 or sTRx != 0 or sTRy != 0 \
                or sBLx != 0 or sBLy != 0 or sBRx != 0 or sBRy != 0:
            self.roi_points = np.float32([
                (sTLx, sTLy),
                (sBLx, sBLy),
                (sBRx, sBRy),
                (sTRx, sTRy)
            ])
        else:
            self.roi_points = np.float32([
                (int(0.500 * width), int(0.600 * height)),  # Top-left corner
                (200, height - 1),  # Bottom-left corner
                (int(0.958 * width), height - 1),  # Bottom-right corner
                (int(0.7000 * width), int(0.600 * height))  # Top-right corner
            ])

        # The desired corner locations  of the region of interest
        # after we perform perspective transformation.
        # Assume image width of 600, padding == 150.
        self.padding = int(0.25 * width)  # padding from side of the image in pixels
        self.desired_roi_points = np.float32([
            [self.padding, 0],  # Top-left corner
            [self.padding, self.orig_image_size[1]],  # Bottom-left corner
            [self.orig_image_size[
                 0] - self.padding, self.orig_image_size[1]],  # Bottom-right corner
            [self.orig_image_size[0] - self.padding, 0]  # Top-right corner
        ])

        # Histogram that shows the white pixel peaks for lane line detection
        self.histogram = None

        # Sliding window parameters
        self.no_of_windows = 10
        self.margin = int((1 / 12) * width)  # Window width is +/- margin
        self.minpix = int((1 / 24) * width)  # Min no. of pixels to recenter window

        # Best fit polynomial lines for left line and right line of the lane
        self.left_fit = None
        self.right_fit = None
        self.left_lane_inds = None
        self.right_lane_inds = None
        self.ploty = None
        self.left_fitx = None
        self.right_fitx = None
        self.leftx = None
        self.rightx = None
        self.lefty = None
        self.righty = None

        # Pixel parameters for x and y dimensions
        self.YM_PER_PIX = 7.0 / 400  # meters per pixel in y dimension
        self.XM_PER_PIX = 3.7 / 255  # meters per pixel in x dimension

        # Radii of curvature and offset
        self.left_curvem = None
        self.right_curvem = None
        self.center_offset = None

    def calculate_car_position(self, print_to_terminal=False):
        # Assume the camera is centered in the image.
        # Get position of car in centimeters
        car_location = self.orig_frame.shape[1] / 2

        # Fine the x coordinate of the lane line bottom
        height = self.orig_frame.shape[0]
        bottom_left = self.left_fit[0] * height ** 2 + self.left_fit[
            1] * height + self.left_fit[2]
        bottom_right = self.right_fit[0] * height ** 2 + self.right_fit[
            1] * height + self.right_fit[2]

        center_lane = (bottom_right - bottom_left) / 2 + bottom_left
        center_offset = (np.abs(car_location) - np.abs(
            center_lane)) * self.XM_PER_PIX * 100

        if print_to_terminal == True:
            print(str(center_offset) + 'cm')

        self.center_offset = center_offset

        return center_offset

    def calculate_curvature(self, print_to_terminal=False):
        # Set the y-value where we want to calculate the road curvature.
        # Select the maximum y-value, which is the bottom of the frame.
        y_eval = np.max(self.ploty)

        # Fit polynomial curves to the real world environment
        left_fit_cr = np.polyfit(self.lefty * self.YM_PER_PIX, self.leftx * (
            self.XM_PER_PIX), 2)
        right_fit_cr = np.polyfit(self.righty * self.YM_PER_PIX, self.rightx * (
            self.XM_PER_PIX), 2)

        # Calculate the radii of curvature
        left_curvem = ((1 + (2 * left_fit_cr[0] * y_eval * self.YM_PER_PIX + left_fit_cr[
            1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])
        right_curvem = ((1 + (2 * right_fit_cr[
            0] * y_eval * self.YM_PER_PIX + right_fit_cr[
                                  1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0])

        # Display on terminal window
        if print_to_terminal == True:
            print(left_curvem, 'm', right_curvem, 'm')

        self.left_curvem = left_curvem
        self.right_curvem = right_curvem

        return left_curvem, right_curvem

    def calculate_histogram(self, frame=None, plot=True):
        if frame is None:
            frame = self.warped_frame

        # Generate the histogram
        self.histogram = np.sum(frame[int(
            frame.shape[0] / 2):, :], axis=0)

        if plot == True:
            # Draw both the image and the histogram
            figure, (ax1, ax2) = plt.subplots(2, 1)  # 2 row, 1 columns
            figure.set_size_inches(10, 5)
            ax1.imshow(frame, cmap='gray')
            ax1.set_title("Warped Binary Frame")
            ax2.plot(self.histogram)
            ax2.set_title("Histogram Peaks")
            plt.show()

        return self.histogram

    def display_curvature_offset(self, frame=None, plot=False):
        image_copy = None
        if frame is None:
            image_copy = self.orig_frame.copy()
        else:
            image_copy = frame

        cv2.putText(image_copy, 'Curve Radius: ' + str((self.left_curvem + self.right_curvem) / 2)[:7] + ' m',
                    (int((5 / 600) * self.width), int((20 / 338) * self.height)),
                    cv2.FONT_HERSHEY_SIMPLEX, (float((0.5 / 600) * self.width)), (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image_copy, 'Center Offset: ' + str(self.center_offset)[:7] + ' cm', (int((5 / 600) * self.width), int((40 / 338) * self.height)),
                    cv2.FONT_HERSHEY_SIMPLEX, (float((0.5 / 600) * self.width)), (255, 255, 255), 2, cv2.LINE_AA)

        if plot == True:
            cv2.imshow("Image with Curvature and Offset", image_copy)

        return image_copy

    def get_lane_line_previous_window(self, left_fit, right_fit, plot=False):
        """
        Use the lane line from the previous sliding window to get the parameters
        for the polynomial line for filling in the lane line
        :param: left_fit Polynomial function of the left lane line
        :param: right_fit Polynomial function of the right lane line
        :param: plot To display an image or not
        """
        # margin is a sliding window parameter
        margin = self.margin

        # Find the x and y coordinates of all the nonzero
        # (i.e. white) pixels in the frame.
        nonzero = self.warped_frame.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Store left and right lane pixel indices
        left_lane_inds = ((nonzerox > (left_fit[0] * (
                nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) & (
                                  nonzerox < (left_fit[0] * (
                                  nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
        right_lane_inds = ((nonzerox > (right_fit[0] * (
                nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) & (
                                   nonzerox < (right_fit[0] * (
                                   nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))
        self.left_lane_inds = left_lane_inds
        self.right_lane_inds = right_lane_inds

        # Get the left and right lane line pixel locations
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        global prev_leftx2
        global prev_lefty2
        global prev_rightx2
        global prev_righty2
        global prev_left_fit2
        global prev_right_fit2

        # Make sure we have nonzero pixels
        if len(leftx) == 0 or len(lefty) == 0 or len(rightx) == 0 or len(righty) == 0:
            leftx = prev_leftx2
            lefty = prev_lefty2
            rightx = prev_rightx2
            righty = prev_righty2

        self.leftx = leftx
        self.rightx = rightx
        self.lefty = lefty
        self.righty = righty

        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Add the latest polynomial coefficients
        prev_left_fit2.append(left_fit)
        prev_right_fit2.append(right_fit)

        # Calculate the moving average
        if len(prev_left_fit2) > 10:
            prev_left_fit2.pop(0)
            prev_right_fit2.pop(0)
            left_fit = sum(prev_left_fit2) / len(prev_left_fit2)
            right_fit = sum(prev_right_fit2) / len(prev_right_fit2)

        self.left_fit = left_fit
        self.right_fit = right_fit

        prev_leftx2 = leftx
        prev_lefty2 = lefty
        prev_rightx2 = rightx
        prev_righty2 = righty

        # Create the x and y values to plot on the image
        ploty = np.linspace(
            0, self.warped_frame.shape[0] - 1, self.warped_frame.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
        self.ploty = ploty
        self.left_fitx = left_fitx
        self.right_fitx = right_fitx

        if plot == True:
            # Generate images to draw on
            out_img = np.dstack((self.warped_frame, self.warped_frame, (
                self.warped_frame))) * 255
            window_img = np.zeros_like(out_img)

            # Add color to the left and right line pixels
            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [
                0, 0, 255]
            # Create a polygon to show the search window area, and recast
            # the x and y points into a usable format for cv2.fillPoly()
            margin = self.margin
            left_line_window1 = np.array([np.transpose(np.vstack([
                left_fitx - margin, ploty]))])
            left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([
                left_fitx + margin, ploty])))])
            left_line_pts = np.hstack((left_line_window1, left_line_window2))
            right_line_window1 = np.array([np.transpose(np.vstack([
                right_fitx - margin, ploty]))])
            right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([
                right_fitx + margin, ploty])))])
            right_line_pts = np.hstack((right_line_window1, right_line_window2))

            # Draw the lane onto the warped blank image
            cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
            cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
            result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

            # Plot the figures
            figure, (ax1, ax2, ax3) = plt.subplots(3, 1)  # 3 rows, 1 column
            figure.set_size_inches(10, 10)
            figure.tight_layout(pad=3.0)
            ax1.imshow(cv2.cvtColor(self.orig_frame, cv2.COLOR_BGR2RGB))
            ax2.imshow(self.warped_frame, cmap='gray')
            ax3.imshow(result)
            ax3.plot(left_fitx, ploty, color='yellow')
            ax3.plot(right_fitx, ploty, color='yellow')
            ax1.set_title("Original Frame")
            ax2.set_title("Warped Frame")
            ax3.set_title("Warped Frame With Search Window")
            plt.show()

    def get_lane_line_indices_sliding_windows(self, plot=False):
        # Sliding window width is +/- margin
        margin = self.margin

        frame_sliding_window = self.warped_frame.copy()

        # Set the height of the sliding windows
        window_height = int(self.warped_frame.shape[0] / self.no_of_windows)

        # Find the x and y coordinates of all the nonzero
        # (i.e. white) pixels in the frame.
        nonzero = self.warped_frame.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Store the pixel indices for the left and right lane lines
        left_lane_inds = []
        right_lane_inds = []

        # Current positions for pixel indices for each window,
        # which we will continue to update
        leftx_base, rightx_base = self.histogram_peak()
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Go through one window at a time
        no_of_windows = self.no_of_windows

        for window in range(no_of_windows):

            # Identify window boundaries in x and y (and right and left)
            win_y_low = self.warped_frame.shape[0] - (window + 1) * window_height
            win_y_high = self.warped_frame.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            cv2.rectangle(frame_sliding_window, (win_xleft_low, win_y_low), (
                win_xleft_high, win_y_high), (255, 255, 255), 2)
            cv2.rectangle(frame_sliding_window, (win_xright_low, win_y_low), (
                win_xright_high, win_y_high), (255, 255, 255), 2)

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                                (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # If you found > minpix pixels, recenter next window on mean position
            minpix = self.minpix
            if len(good_left_inds) > minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract the pixel coordinates for the left and right lane lines
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial curve to the pixel coordinates for
        # the left and right lane lines
        left_fit = None
        right_fit = None

        global prev_leftx
        global prev_lefty
        global prev_rightx
        global prev_righty
        global prev_left_fit
        global prev_right_fit

        # Make sure we have nonzero pixels
        if len(leftx) == 0 or len(lefty) == 0 or len(rightx) == 0 or len(righty) == 0:
            leftx = prev_leftx
            lefty = prev_lefty
            rightx = prev_rightx
            righty = prev_righty

        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Add the latest polynomial coefficients
        prev_left_fit.append(left_fit)
        prev_right_fit.append(right_fit)

        # Calculate the moving average
        if len(prev_left_fit) > 10:
            prev_left_fit.pop(0)
            prev_right_fit.pop(0)
            left_fit = sum(prev_left_fit) / len(prev_left_fit)
            right_fit = sum(prev_right_fit) / len(prev_right_fit)

        self.left_fit = left_fit
        self.right_fit = right_fit

        prev_leftx = leftx
        prev_lefty = lefty
        prev_rightx = rightx
        prev_righty = righty

        if plot == True:
            # Create the x and y values to plot on the image
            ploty = np.linspace(
                0, frame_sliding_window.shape[0] - 1, frame_sliding_window.shape[0])
            left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
            right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

            # Generate an image to visualize the result
            out_img = np.dstack((
                frame_sliding_window, frame_sliding_window, (
                    frame_sliding_window))) * 255

            # Add color to the left line pixels and right line pixels
            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [
                0, 0, 255]

            # Plot the figure with the sliding windows
            figure, (ax1, ax2, ax3) = plt.subplots(3, 1)  # 3 rows, 1 column
            figure.set_size_inches(10, 10)
            figure.tight_layout(pad=3.0)
            ax1.imshow(cv2.cvtColor(self.orig_frame, cv2.COLOR_BGR2RGB))
            ax2.imshow(frame_sliding_window, cmap='gray')
            ax3.imshow(out_img)
            ax3.plot(left_fitx, ploty, color='yellow')
            ax3.plot(right_fitx, ploty, color='yellow')
            ax1.set_title("Original Frame")
            ax2.set_title("Warped Frame with Sliding Windows")
            ax3.set_title("Detected Lane Lines with Sliding Windows")
            plt.show()

        return self.left_fit, self.right_fit

    def get_line_markings(self, frame=None):
        if frame is None:
            frame = self.orig_frame

        # Convert the video frame from BGR (blue, green, red)
        # color space to HLS (hue, saturation, lightness).
        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)

        ################### Isolate possible lane line edges ######################

        # Perform Sobel edge detection on the L (lightness) channel of
        # the image to detect sharp discontinuities in the pixel intensities
        # along the x and y axis of the video frame.
        # sxbinary is a matrix full of 0s (black) and 255 (white) intensity values
        # Relatively light pixels get made white. Dark pixels get made black.
        _, sxbinary = edge.threshold(hls[:, :, 1], thresh=(120, 255))
        sxbinary = edge.blur_gaussian(sxbinary, ksize=3)  # Reduce noise

        # 1s will be in the cells with the highest Sobel derivative values
        # (i.e. strongest lane line edges)
        sxbinary = edge.mag_thresh(sxbinary, sobel_kernel=3, thresh=(110, 255))

        ######################## Isolate possible lane lines ######################

        # Perform binary thresholding on the S (saturation) channel
        # of the video frame. A high saturation value means the hue color is pure.
        # We expect lane lines to be nice, pure colors (i.e. solid white, yellow)
        # and have high saturation channel values.
        # s_binary is matrix full of 0s (black) and 255 (white) intensity values
        # White in the regions with the purest hue colors (e.g. >130...play with
        # this value for best results).
        s_channel = hls[:, :, 2]  # use only the saturation channel data
        _, s_binary = edge.threshold(s_channel, (int(high_saturation.get()), 255))

        # Perform binary thresholding on the R (red) channel of the
        # original BGR video frame.
        # r_thresh is a matrix full of 0s (black) and 255 (white) intensity values
        # White in the regions with the richest red channel values (e.g. >120).
        # Remember, pure white is bgr(255, 255, 255).
        # Pure yellow is bgr(0, 255, 255). Both have high red channel values.
        _, r_thresh = edge.threshold(frame[:, :, 2], thresh=(120, 255))

        # Lane lines should be pure in color and have high red channel values
        # Bitwise AND operation to reduce noise and black-out any pixels that
        # don't appear to be nice, pure, solid colors (like white or yellow lane
        # lines.)
        rs_binary = cv2.bitwise_and(s_binary, r_thresh)

        ### Combine the possible lane lines with the possible lane line edges #####
        # If you show rs_binary visually, you'll see that it is not that different
        # from this return value. The edges of lane lines are thin lines of pixels.
        self.lane_line_markings = cv2.bitwise_or(rs_binary, sxbinary.astype(np.uint8))
        return self.lane_line_markings

    def histogram_peak(self):
        midpoint = int(self.histogram.shape[0] / 2)
        leftx_base = np.argmax(self.histogram[:midpoint])
        rightx_base = np.argmax(self.histogram[midpoint:]) + midpoint

        # (x coordinate of left peak, x coordinate of right peak)
        return leftx_base, rightx_base

    def overlay_lane_lines(self, plot=False):
        # Generate an image to draw the lane lines on
        warp_zero = np.zeros_like(self.warped_frame).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([
            self.left_fitx, self.ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([
            self.right_fitx, self.ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw lane on the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective
        # matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, self.inv_transformation_matrix, (
            self.orig_frame.shape[
                1], self.orig_frame.shape[0]))

        # Combine the result with the original image
        result = cv2.addWeighted(self.orig_frame, 1, newwarp, 0.3, 0)

        if plot == True:
            # Plot the figures
            figure, (ax1, ax2) = plt.subplots(2, 1)  # 2 rows, 1 column
            figure.set_size_inches(10, 10)
            figure.tight_layout(pad=3.0)
            ax1.imshow(cv2.cvtColor(self.orig_frame, cv2.COLOR_BGR2RGB))
            ax2.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            ax1.set_title("Original Frame")
            ax2.set_title("Original Frame With Lane Overlay")
            plt.show()

        return result

    def perspective_transform(self, frame=None, plot=False):
        if frame is None:
            frame = self.lane_line_markings

        # Calculate the transformation matrix
        self.transformation_matrix = cv2.getPerspectiveTransform(
            self.roi_points, self.desired_roi_points)

        # Calculate the inverse transformation matrix
        self.inv_transformation_matrix = cv2.getPerspectiveTransform(
            self.desired_roi_points, self.roi_points)

        # Perform the transform using the transformation matrix
        self.warped_frame = cv2.warpPerspective(
            frame, self.transformation_matrix, self.orig_image_size, flags=(
                cv2.INTER_LINEAR))

        # Convert image to binary
        (thresh, binary_warped) = cv2.threshold(
            self.warped_frame, 127, 255, cv2.THRESH_BINARY)
        self.warped_frame = binary_warped

        # Display the perspective transformed (i.e. warped) frame
        if plot == True:
            warped_copy = self.warped_frame.copy()
            warped_plot = cv2.polylines(warped_copy, np.int32([
                self.desired_roi_points]), True, (147, 20, 255), 3)

            # Display the image
            while (1):
                cv2.imshow('Warped Image', warped_plot)

                # Press any key to stop
                if cv2.waitKey(0):
                    break

            cv2.destroyAllWindows()

        return self.warped_frame

    def plot_roi(self, frame=None, plot=False):
        if plot == False:
            return

        if frame is None:
            frame = self.orig_frame.copy()

        # Overlay trapezoid on the frame
        this_image = cv2.polylines(frame, np.int32([
            self.roi_points]), True, (147, 20, 255), 3)

        # Display the image
        while (1):
            cv2.imshow('ROI Image', this_image)

            # Press any key to stop
            if cv2.waitKey(0):
                break

        cv2.destroyAllWindows()


def main(regofin: bool, wafr: bool, hist: bool, slwpix: bool, prevlin: bool, frwili: bool, calibrate: bool):
    string_sr = str(tk_scale_ratio.get())
    global scale_ratio
    scale_ratio = float(string_sr)

    if not filename.get():
        mb.showerror("Error", "Please select video file")
        return

    # Load a video
    cap = cv2.VideoCapture(filename.get())

    # Create a VideoWriter object so we can save the video output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Process the video
    while (cap.isOpened()):

        # Capture one frame at a time
        success, frame = cap.read()

        # Do we have a video frame? If true, proceed.
        if success:
            # Resize the frame
            width = int(frame.shape[1] * scale_ratio)
            height = int(frame.shape[0] * scale_ratio)
            frame = cv2.resize(frame, (width, height))

            # Store the original frame
            original_frame = frame.copy()

            if calibrate:
                return original_frame

            # Create a Lane object
            lane_obj = Lane(orig_frame=original_frame)

            # Perform thresholding to isolate lane lines
            lane_line_markings = lane_obj.get_line_markings()

            # Plot the region of interest on the image
            lane_obj.plot_roi(plot=regofin)

            # Perform the perspective transform to generate a bird's eye view
            # If Plot == True, show image with new region of interest
            warped_frame = lane_obj.perspective_transform(plot=wafr)

            # Generate the image histogram to serve as a starting point
            # for finding lane line pixels
            histogram = lane_obj.calculate_histogram(plot=hist)

            # Find lane line pixels using the sliding window method
            left_fit, right_fit = lane_obj.get_lane_line_indices_sliding_windows(
                plot=slwpix)

            # Fill in the lane line
            lane_obj.get_lane_line_previous_window(left_fit, right_fit, plot=prevlin)

            # Overlay lines on the original frame
            frame_with_lane_lines = lane_obj.overlay_lane_lines(plot=frwili)

            # Calculate lane line curvature (left and right lane lines)
            lane_obj.calculate_curvature(print_to_terminal=False)

            # Calculate center offset
            lane_obj.calculate_car_position(print_to_terminal=False)

            # Display curvature and center offset on image
            frame_with_lane_lines2 = lane_obj.display_curvature_offset(
                frame=frame_with_lane_lines, plot=False)

            # Display the frame
            cv2.imshow("Frame", frame_with_lane_lines2)
            if filename.get().endswith(".jpg") or filename.get().endswith(".png"):
                cv2.waitKey()

            if cv2.waitKey(1) & 0xFF == ord('s'):
                cv2.waitKey(0)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        # No more video frames left
        else:
            break

    # Stop when the video is finished
    cap.release()

    # Close all windows
    cv2.destroyAllWindows()


root = tk.Tk()

width = 750
height = 400

s_width = root.winfo_screenwidth()
s_height = root.winfo_screenheight()

x = (s_width/2) - (width/2)
y = (s_height/2) - (height/2)

root.title("Lane Line Detection | Control Panel")
root.geometry("%dx%d+%d+%d" % (width, height, x, y))
frame = tk.Frame(root)
frame.pack()

select_file = tk.Label(frame, text="Select video file").grid(row=0, column=0)
filename = tk.StringVar()
entry_select_file = tk.Entry(frame, textvariable=filename).grid(row=0, column=1)


def browse_file():
    file_path = filedialog.askopenfilename(filetypes=(("mp4 files", "*.mp4"),
                                                      ("png files", "*.png"),
                                                      ("jpeg files", "*.jpg")))
    filename.set(file_path)


browse = tk.Button(frame, text="Browse", command=browse_file).grid(row=0, column=2)

roi = tk.BooleanVar()
warped_frame = tk.BooleanVar()
histogram = tk.BooleanVar()
sliding_window_pixels = tk.BooleanVar()
previous_lines = tk.BooleanVar()
frame_with_lines = tk.BooleanVar()
tk_scale_ratio = tk.DoubleVar()
high_saturation_scale = tk.IntVar()

tk.Checkbutton(frame, text="Region of Interest", variable=roi).grid(row=7, column=1, sticky='W')
tk.Checkbutton(frame, text="Bird's eye view, for high saturation", variable=warped_frame).grid(row=8, column=1, sticky='W')
tk.Checkbutton(frame, text="Histogram", variable=histogram).grid(row=9, column=1, sticky='W')
tk.Checkbutton(frame, text="Lane Line pixels", variable=sliding_window_pixels).grid(row=10, column=1, sticky='W')
tk.Checkbutton(frame, text="Lane Line", variable=previous_lines).grid(row=11, column=1, sticky='W')
tk.Checkbutton(frame, text="Lines on the original frame", variable=frame_with_lines).grid(row=12, column=1, sticky='W')

select_scale_ratio = tk.Label(frame, text="Select video scale ratio").grid(row=1, column=0)
scale = tk.Scale(frame, variable=tk_scale_ratio, orient=tk.HORIZONTAL, resolution=0.1, from_=0.5, to=1.0).grid(row=1, column=1)
select_scale_high_saturation = tk.Label(frame, text="Select high saturation of image, best range is between 10 and 80").grid(row=2, column=0)
high_saturation = tk.Scale(frame, variable=high_saturation_scale, orient=tk.HORIZONTAL, resolution=1, from_=1, to=100)
high_saturation.set(10)
high_saturation.grid(row=2, column=1)


def onclick(event):
    x1, y1 = int(event.xdata), int(event.ydata)
    global TLx, TLy, TRx, TRy, BLx, BLy, BRx, BRy
    global sTLx, sTLy, sTRx, sTRy, sBLx, sBLy, sBRx, sBRy

    if choice.get() == 1:
        TLx.set(x1)
        TLy.set(y1)
        sTLx = int(TLx.get())
        sTLy = int(TLy.get())
    if choice.get() == 2:
        TRx.set(x1)
        TRy.set(y1)
        sTRx = int(TRx.get())
        sTRy = int(TRy.get())
    if choice.get() == 3:
        BLx.set(x1)
        BLy.set(y1)
        sBLx = int(BLx.get())
        sBLy = int(BLy.get())
    if choice.get() == 4:
        BRx.set(x1)
        BRy.set(y1)
        sBRx = int(BRx.get())
        sBRy = int(BRy.get())


def calibrate_roi():
    if not filename.get():
        mb.showerror("Error", "Please select video file")
        return

    result = main(regofin=False, wafr=False, hist=False, slwpix=False, prevlin=False, frwili=False, calibrate=True)

    fig, ax = plt.subplots()
    cursor = Cursor(ax,
                    horizOn=True,
                    vertOn=True,
                    color='green',
                    linewidth=2.0)

    fig.canvas.mpl_connect('button_press_event', onclick)

    plt.imshow(result)
    plt.show()


choice = tk.IntVar()
TLx = tk.IntVar()
TLy = tk.IntVar()
TRx = tk.IntVar()
TRy = tk.IntVar()
BLx = tk.IntVar()
BLy = tk.IntVar()
BRx = tk.IntVar()
BRy = tk.IntVar()

sTLx = 0
sTLy = 0
sTRx = 0
sTRy = 0
sBLx = 0
sBLy = 0
sBRx = 0
sBRy = 0

r1 = tk.Radiobutton(frame, text="Top-Left corner", variable=choice, value=1).grid(row=3, column=0, sticky='W')
r2 = tk.Radiobutton(frame, text="Top-Right corner", variable=choice, value=2).grid(row=4, column=0, sticky='W')
r3 = tk.Radiobutton(frame, text="Bottom-Left corner", variable=choice, value=3).grid(row=5, column=0, sticky='W')
r4 = tk.Radiobutton(frame, text="Bottom-Right corner", variable=choice, value=4).grid(row=6, column=0, sticky='W')

ETLx = tk.Entry(frame, textvariable=TLx).grid(row=3, column=0, sticky='E')
ETLy = tk.Entry(frame, textvariable=TLy).grid(row=3, column=1, sticky='W')
ETRx = tk.Entry(frame, textvariable=TRx).grid(row=4, column=0, sticky='E')
ETRy = tk.Entry(frame, textvariable=TRy).grid(row=4, column=1, sticky='W')
EBLx = tk.Entry(frame, textvariable=BLx).grid(row=5, column=0, sticky='E')
EBLy = tk.Entry(frame, textvariable=BLy).grid(row=5, column=1, sticky='W')
EBRx = tk.Entry(frame, textvariable=BRx).grid(row=6, column=0, sticky='E')
EBRy = tk.Entry(frame, textvariable=BRy).grid(row=6, column=1, sticky='W')

calibrate_button = tk.Button(frame, text="Calibrate Region of Interest", command=calibrate_roi).grid(row=2, column=2)
control_keys = tk.Label(frame, text=
                        """Control Keys:
Press "q" to exit the algorithm process.
Press "s" to stop frame during algorithm process.""",
                        justify=tk.LEFT, bg="#dbdbdb").grid(row=7, column=0, rowspan=7)


def start_alg():
    main(roi.get(), warped_frame.get(), histogram.get(), sliding_window_pixels.get(),
         previous_lines.get(), frame_with_lines.get(), calibrate=False)


run_button = tk.Button(frame, text="Run Algorithm", command=start_alg).grid(row=12, column=2)

root.mainloop()
