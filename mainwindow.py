import sys
import traceback
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog
from PySide6 import QtCore
from ui_form import Ui_MainWindow
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# remember to run pyside6-uic form.ui -o ui_form.py to update the UI file
class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.actionopen_a_video.triggered.connect(self.open_video_file)
        self.ui.pushButton.clicked.connect(self.start_video_playback)
        self.ui.pushButton_Grayscale.clicked.connect(self.switch_label2_display_mode)

        self.display_mode = "rgb"

        # Initialize variables
        self.cap = None
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_video_frame)

        # Initialize HSV threshold values
        self.lower_h = 0
        self.lower_s = 0
        self.lower_v = 0
        self.upper_h = 179
        self.upper_s = 255
        self.upper_v = 255

        # Set the correct range for each slider
        self.ui.horizontalSlider_LowerH.setMinimum(0)
        self.ui.horizontalSlider_LowerH.setMaximum(179)
        self.ui.horizontalSlider_UpperH.setMinimum(0)
        self.ui.horizontalSlider_UpperH.setMaximum(179)

        self.ui.horizontalSlider_LowerS.setMinimum(0)
        self.ui.horizontalSlider_LowerS.setMaximum(255)
        self.ui.horizontalSlider_UpperS.setMinimum(0)
        self.ui.horizontalSlider_UpperS.setMaximum(255)

        self.ui.horizontalSlider_LowerV.setMinimum(0)
        self.ui.horizontalSlider_LowerV.setMaximum(255)
        self.ui.horizontalSlider_UpperV.setMinimum(0)
        self.ui.horizontalSlider_UpperV.setMaximum(255)

        # Set initial values for sliders and labels
        self.ui.horizontalSlider_LowerH.setValue(self.lower_h)
        self.ui.horizontalSlider_UpperH.setValue(self.upper_h)
        self.ui.horizontalSlider_LowerS.setValue(self.lower_s)
        self.ui.horizontalSlider_UpperS.setValue(self.upper_s)
        self.ui.horizontalSlider_LowerV.setValue(self.lower_v)
        self.ui.horizontalSlider_UpperV.setValue(self.upper_v)

        # Connect sliders to update_hsv_controls
        self.ui.horizontalSlider_LowerH.valueChanged.connect(self.update_hsv_controls)
        self.ui.horizontalSlider_LowerS.valueChanged.connect(self.update_hsv_controls)
        self.ui.horizontalSlider_LowerV.valueChanged.connect(self.update_hsv_controls)
        self.ui.horizontalSlider_UpperH.valueChanged.connect(self.update_hsv_controls)
        self.ui.horizontalSlider_UpperS.valueChanged.connect(self.update_hsv_controls)
        self.ui.horizontalSlider_UpperV.valueChanged.connect(self.update_hsv_controls)

        # Add a flag to indicate whether HSV values are locked
        self.hsv_locked = False

        # Connect the pushButton_HSV click signal to the toggle_hsv_lock method
        self.ui.pushButton_HSV.clicked.connect(self.toggle_hsv_lock)

        # Call update_hsv_controls to initialize labels
        self.update_hsv_controls()

        # Connect the binary slider to update_binary_threshold method
        self.ui.horizontalSlider_Binary.valueChanged.connect(self.update_binary_threshold)
        
        #Set the range for binary slider
        self.ui.horizontalSlider_Binary.setMinimum(0)
        self.ui.horizontalSlider_Binary.setMaximum(255)

        # Call update_binary_threshold to initialize label
        self.update_binary_threshold()

        # Connect the Canny sliders
        self.ui.horizontalSlider_Canny.valueChanged.connect(self.update_canny_threshold)
        self.ui.horizontalSlider_Canny_2.valueChanged.connect(self.update_canny_threshold_2)

        # Set the range for Canny sliders
        self.ui.horizontalSlider_Canny.setMinimum(0)
        self.ui.horizontalSlider_Canny.setMaximum(255)
        self.ui.horizontalSlider_Canny_2.setMinimum(0)
        self.ui.horizontalSlider_Canny_2.setMaximum(255)

        # Call update_canny_threshold to initialize labels
        self.update_canny_threshold()
        self.update_canny_threshold_2()

    def switch_label2_display_mode(self):
        # global display_mode
        #Switch between grayscale and binary mode
        if self.display_mode == "rgb":
            self.display_mode = "red"
            self.ui.pushButton_Grayscale.setText("Current Display Mode: RED")
        elif self.display_mode == "red":
            self.display_mode = "grayscale & binary"
            self.ui.pushButton_Grayscale.setText("Current Display Mode: Grayscale and Binary")
        else:
            self.display_mode = "rgb"
            self.ui.pushButton_Grayscale.setText("Current Display Mode: RGB")

    def update_hsv_controls(self):
        # Check if HSV values are locked
        if not self.hsv_locked:
            # Get slider values
            self.lower_h = self.ui.horizontalSlider_LowerH.value()
            self.lower_s = self.ui.horizontalSlider_LowerS.value()
            self.lower_v = self.ui.horizontalSlider_LowerV.value()
            self.upper_h = self.ui.horizontalSlider_UpperH.value()
            self.upper_s = self.ui.horizontalSlider_UpperS.value()
            self.upper_v = self.ui.horizontalSlider_UpperV.value()

            # Update labels
            self.ui.label_LowerH.setText(f"Lower H: {self.lower_h}")
            self.ui.label_LowerS.setText(f"Lower S: {self.lower_s}")
            self.ui.label_LowerV.setText(f"Lower V: {self.lower_v}")
            self.ui.label_UpperH.setText(f"Upper H: {self.upper_h}")
            self.ui.label_UpperS.setText(f"Upper S: {self.upper_s}")
            self.ui.label_UpperV.setText(f"Upper V: {self.upper_v}")

    def open_video_file(self):
        file_info = QFileDialog.getOpenFileUrl(self, 'Open Video', '', 'Videos (*.mp4 *.avi *.mov)')
        if not file_info or not isinstance(file_info, tuple) or len(file_info) < 1:
            return

        filepath = file_info[0].toLocalFile()
        self.cap = cv2.VideoCapture(filepath)
        if not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if ret:
            self.display_frame(frame)

    def start_video_playback(self):
        if self.cap is not None:
            if self.hsv_locked == False:
                if not self.timer.isActive():
                    # Start the timer if it's not already active
                    self.timer.start(33)
                    self.ui.pushButton.setText("Pause")  # Change button text to Pause
                else:
                    # Stop the timer if it's active
                    self.timer.stop()
                    self.ui.pushButton.setText("Play")  # Change button text to Play
            



    def update_video_frame(self):   
        ret, frame = self.cap.read()
        if ret:
            self.display_frame(frame)
        else:
            # Stop the timer when the video ends
            self.timer.stop()

            # Reset the video capture to play again
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            if self.hsv_locked == False:
                # Uncomment the line below if you want to restart the video automatically
                self.timer.start(33)
        

    def display_frame(self, frame):
        global binary_threshold, canny_threshold, canny_threshold_2
        global frame_count, fps, time

        # Convert frame to HSV color space
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Apply HSV thresholding
        lower_bound = np.array([self.lower_h, self.lower_s, self.lower_v])
        upper_bound = np.array([self.upper_h, self.upper_s, self.upper_v])
        mask = cv2.inRange(frame_hsv, lower_bound, upper_bound)
        result_frame = cv2.bitwise_and(frame, frame, mask=mask)
        result_frame_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)

        # Find contours and draw rectangles
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea, default=None)

        # Draw rectangles around contours
        frame_with_rectangles = self.draw_rectangles(frame, [largest_contour] if largest_contour is not None else [])

        # Display the processed frame in video_label
        height, width, channel = result_frame.shape
        bytesPerLine = 3 * width
        qImg = QImage(result_frame_rgb.data, width, height, bytesPerLine, QImage.Format_RGB888)
        qImg = qImg.scaled(self.ui.video_label.size(), QtCore.Qt.KeepAspectRatio)
        self.ui.video_label.setPixmap(QPixmap.fromImage(qImg))

        # Display the frame with rectangles in video_label2
        frame_input_for_label3 = self.display_frame_on_label2(frame_with_rectangles, largest_contour)
        self.display_frame_on_label3(frame_input_for_label3, binary_threshold, canny_threshold, canny_threshold_2)

        # Display the time from the video on time_label
        frame_count = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        time = frame_count / fps
        self.ui.time_label.setText(f"Time: {time:.2f} seconds")

        # Calculate and display absorption rate
        self.calculate_and_display_absorption_rate(mask, largest_contour, time)

    def toggle_hsv_lock(self):
        # Toggle the lock state
        self.hsv_locked = True

        #Reset the timer if HSV values are locked
        self.timer.stop()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.timer.start(33)
        self.ui.pushButton.setText("Saving Processed Video...")
        

    def draw_rectangles(self, frame, contours):
            # Draw rectangles around contours
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            return frame
    
    def display_frame_on_label2(self, frame, contour):
        global frame_count, time, fps
        global frame_rgb2
        # global display_mode
        # Display the frame with the largest contour in video_label2
        if contour is not None:
            # Create a mask for the largest contour
            mask = np.zeros_like(frame)
            cv2.drawContours(mask, [contour], 0, (255, 255, 255), thickness=cv2.FILLED)

            # Apply the mask to the original frame
            frame = cv2.bitwise_and(frame, mask)
            #裁切圖片至輪廓大小
            x, y, w, h = cv2.boundingRect(contour)
            frame = frame[y:y+h, x:x+w]
            # #投影變換、解決形變問題、更新frame
            # pts1 = np.float32([[0,0],[0,frame.shape[0]],[frame.shape[1],0],[frame.shape[1],frame.shape[0]]])
            # pts2 = np.float32([[0,0],[0,frame.shape[0]],[frame.shape[1],0],[frame.shape[1],frame.shape[0]]])
            # M = cv2.getPerspectiveTransform(pts1,pts2)
            # frame = cv2.warpPerspective(frame,M,(frame.shape[1],frame.shape[0]))            
            # Create a folder to save the cut frames
            if not os.path.exists('cut_frames'):
                os.makedirs('cut_frames')
            # Save the cut frames to a file if HSV values are locked and the framecount is a multiple of 10
            if self.hsv_locked == True and frame_count % 10 == 0:   
                cv2.imwrite(f"cut_frames/frame_{frame_count}.jpg", frame)
                # print(f"Saved frame {frame_count} to file.")
            #把frame平滑化
            frame = cv2.medianBlur(frame, 13)
            #使顏色交界明顯化
            frame = cv2.GaussianBlur(frame, (5, 5), 0)

            # Convert frame to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
            # Apply binary thresholding
            _, binary_frame = cv2.threshold(gray_frame, binary_threshold, 255, cv2.THRESH_BINARY)
            
            #Create a folder to save the binary frames
            if not os.path.exists('binary_frames'):
                os.makedirs('binary_frames')
            #Save the binary_frame to a file if HSV values are locked and the framecount is a multiple of 10
            if self.hsv_locked == True and frame_count % 10 == 0:
                cv2.imwrite(f"binary_frames/frame_{frame_count}.jpg", binary_frame)
                # print(f"Saved frame {frame_count} to file.")

            #Only keep the R channel
            frame_rgb_r = frame.copy()
            frame_rgb_r[:, :, 0] = 0
            frame_rgb_r[:, :, 1] = 0

            frame_rgb2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #BGR轉RGB
            # Create a folder to save the cut frames
            if not os.path.exists('processed_cut_frames'):
                os.makedirs('processed_cut_frames')
            # Save the cut frames to a file if HSV values are locked and the framecount is a multiple of 10
            if self.hsv_locked == True and frame_count % 10 == 0:   
                cv2.imwrite(f"processed_cut_frames/frame_{frame_count}.jpg", frame)
                # print(f"Saved frame {frame_count} to file.")

        if self.display_mode == "grayscale & binary":
            height, width = binary_frame.shape
            bytesPerLine = width            
            qImg = QImage(binary_frame.data, width, height, bytesPerLine, QImage.Format_Grayscale8)

        elif self.display_mode == "red":            
            height, width = frame_rgb_r.shape[:2]
            bytesPerLine = 3 * width
            qImg = QImage(frame_rgb_r.data, width, height, bytesPerLine, QImage.Format_Grayscale8)

        else:            
            height, width, channel = frame_rgb2.shape
            bytesPerLine = 3 * width
            qImg = QImage(frame_rgb2.data, width, height, bytesPerLine, QImage.Format_RGB888)
        
        qImg = qImg.scaled(self.ui.video_label2.size(), QtCore.Qt.KeepAspectRatio)
        self.ui.video_label2.setPixmap(QPixmap.fromImage(qImg))
        return frame

    def update_binary_threshold(self):
        global binary_threshold

        if not self.hsv_locked:
            # Get the binary threshold value from the slider
            binary_threshold = self.ui.horizontalSlider_Binary.value()

        # Update the label to show the threshold value
        self.ui.label_Binary.setText(f"Binarization Thres: {binary_threshold}")

        # Update video_label3 with the new binary threshold
        if self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                self.display_frame_on_label3(frame, binary_threshold, canny_threshold, canny_threshold_2)


    def display_frame_on_label3(self, frame, binary_threshold, canny_threshold_1, canny_threshold_2):
        global time, frame_count, fps
        global final_frame
        
        # Apply Gaussian blur
        frame = cv2.GaussianBlur(frame, (13, 13), 0)
        #把frame平滑化
        frame = cv2.medianBlur(frame, 5)
        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Canny edge detection with the new thresholds
        gray_frame = cv2.Canny(gray_frame, canny_threshold_1, canny_threshold_2)

        # Apply binary thresholding
        _, binary_frame = cv2.threshold(gray_frame, binary_threshold, 255, cv2.THRESH_BINARY)
        #Use HoughLinesP to find the most obvious horizontal contour line on the frame
        lines = cv2.HoughLinesP(binary_frame, 1, np.pi/180, 50, minLineLength=60, maxLineGap=30)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(gray_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
          
        # Display the processed frame in video_label3
        height, width = gray_frame.shape
        bytesPerLine = width
        qImg = QImage(gray_frame.data, width, height, bytesPerLine, QImage.Format_Grayscale8)
        qImg = qImg.scaled(self.ui.video_label3.size(), QtCore.Qt.KeepAspectRatio)
        self.ui.video_label3.setPixmap(QPixmap.fromImage(qImg))
        final_frame = gray_frame

        #Create a folder to save the processed frames
        if not os.path.exists('final_processed_frames'):
            os.makedirs('final_processed_frames')
        # Save the processed frame to a file if HSV values are locked and the framecount is a multiple of 10
        if self.hsv_locked == True and frame_count % 10 == 0:
            cv2.imwrite(f"final_processed_frames/frame_{frame_count}.jpg", gray_frame)
            # print(f"Saved frame {frame_count} to file.")

        
        # self.y_axis_histogram(final_frame)


    def y_axis_histogram(self, frame):
        y_axis_grayscale_value = []
        for i in range(frame.shape[0]): #frame.shape[0] = height
            # 加入每一行的平均灰階值
            y_axis_grayscale_value.append(np.mean(frame[i, :]))
        #輸出y_axis_grayscale_value到txt檔
        with open('y_axis_grayscale_value.txt', 'w') as f:
            for item in y_axis_grayscale_value:
                f.write("%s\n" % item)
        
        print(len(y_axis_grayscale_value))
        # Plot the histogram
        plt.hist(y_axis_grayscale_value, bins= len(y_axis_grayscale_value), orientation='horizontal')
        #設定y軸座標為y_axis_grayscale_value的index
        plt.yticks(np.arange(len(y_axis_grayscale_value)))
        #標註最大值
        plt.axhline(y_axis_grayscale_value.index(max(y_axis_grayscale_value)), color='r', linestyle='dashed', linewidth=1)
        #設定x軸座標為0~255
        plt.xticks(np.arange(0, 256, 50))


        # Set labels and title
        plt.xlabel('Image X-coordinate')
        plt.ylabel('Grayscale Value')
        plt.title('Y-axis Histogram')
        #設定plt比例為frame比例
        # plt.rcParams["figure.figsize"] = ((frame.shape[1]/100)*1.1, frame.shape[0]/100)
        plt.rcParams["figure.figsize"] = (2.5, 4.8)
        #使文字不被切掉
        plt.tight_layout()

        # Save the plot to a file
        plt.savefig('histogram.png')
        plt.close()

        # Load the saved image using QPixmap
        qPixmap = QPixmap('histogram.png')

        # Resize the image to fit the label
        qPixmap = qPixmap.scaled(self.ui.video_label4.size(), QtCore.Qt.KeepAspectRatio)

        # Display the image in video_label4
        self.ui.video_label4.setPixmap(qPixmap)

    # def y_axis_histogram(self, frame): 
    #     global y_axis_hist
    #     y_axis_hist = []
    #     for i in range(frame.shape[1]):
    #         y_axis_hist.append(np.sum(frame[:, i]))

    #     plt.hist(y_axis_hist, bins=100, orientation='horizontal')
    #     #圖轉90度
    #     # plt.gca().invert_xaxis()
    #     # plt.gca().invert_yaxis()
    #     #標註最大值
    #     plt.axhline(y_axis_hist.index(max(y_axis_hist)), color='r', linestyle='dashed', linewidth=1)
    #     plt.xlabel('Intensity')
    #     plt.ylabel('Frequency')
    #     plt.title('Histogram of Intensity')
    #     #設定plt比例為frame比例
    #     plt.rcParams["figure.figsize"] = (frame.shape[1]/100, frame.shape[0]/100)
                
    #     # Save the plot to a file
    #     plt.savefig('histogram.png')
    #     plt.close()

    #     # Load the saved image using QPixmap
    #     qPixmap = QPixmap('histogram.png')

    #     # Resize the image to fit the label
    #     qPixmap = qPixmap.scaled(self.ui.video_label4.size(), QtCore.Qt.KeepAspectRatio)

    #     # Display the image in video_label4
    #     self.ui.video_label4.setPixmap(qPixmap)

            
                
    def region_growing(self, binary_frame):
        h, w = binary_frame.shape[:2]
        segmented_frame = np.zeros((h, w), dtype=np.uint8)

        # Set seed point (you can modify this based on your specific requirements)
        seed = (h // 2, w // 2)

        # Create a mask to keep track of visited pixels
        mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

        # Set connectivity (8-connectivity)
        connectivity = 8

        # Use cv2.floodFill to perform region growing
        new_mask = cv2.floodFill(segmented_frame, mask, seed, 255, 0, 0, connectivity)[1]

        return new_mask


    def update_canny_threshold(self):
        global canny_threshold
        # Get the Canny threshold value from the slider
        canny_threshold = self.ui.horizontalSlider_Canny.value()

        # Update the label to show the threshold value
        self.ui.label_Canny.setText(f"Canny Thres: {canny_threshold}")

        # Update video_label3 with the new Canny threshold
        if self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                self.display_frame_on_label3(frame, binary_threshold, canny_threshold, canny_threshold_2)

    def update_canny_threshold_2(self):
        global canny_threshold_2
        # Get the second Canny threshold value from the slider
        canny_threshold_2 = self.ui.horizontalSlider_Canny_2.value()

        # Update the label to show the threshold value
        self.ui.label_Canny_2.setText(f"Canny Thres 2: {canny_threshold_2}")

        # Update video_label3 with the new second Canny threshold
        if self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                self.display_frame_on_label3(frame, binary_threshold, canny_threshold, canny_threshold_2)

    def calculate_and_display_absorption_rate(self, mask, largest_contour, time):
        # Calculate area of the largest contour
        contour_area = cv2.contourArea(largest_contour) if largest_contour is not None else 0

        # Display the area on area_label
        self.ui.area_label.setText(f"Area: {contour_area:.2f} pixels")

        # Calculate absorption rate (change in area per unit time)
        if hasattr(self, 'prev_contour_area'):
            area_change = contour_area - self.prev_contour_area
            time_change = time - self.prev_time

            absorption_rate = area_change / time_change if time_change != 0 else 0
            self.ui.absorption_rate_label.setText(f"Absorption Rate: {absorption_rate:.2f} pixels/s")

        # Store current values for the next iteration
        self.prev_contour_area = contour_area
        self.prev_time = time


if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = MainWindow()
    try:
        widget.show()
        sys.exit(app.exec())
    except Exception as e:
        print(f"Unhandled exception: {e}")
        traceback.print_exc()

# import sys
# from PySide6.QtGui import QImage, QPixmap
# from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog
# from PySide6 import QtCore
# from ui_form import Ui_MainWindow
# import cv2
# import numpy as np

# class MainWindow(QMainWindow):
#     def __init__(self, parent=None):
#         super().__init__(parent)
#         self.ui = Ui_MainWindow()
#         self.ui.setupUi(self)
#         self.ui.actionopen_a_video.triggered.connect(self.open_video_file)
#         self.ui.pushButton.clicked.connect(self.start_video_playback)
#         self.ui.pushButton_Grayscale.clicked.connect(self.process_and_display_grayscale_frames)

#         # Initialize variables
#         self.cap = None
#         self.timer = QtCore.QTimer(self)
#         self.timer.timeout.connect(self.update_video_frame)
#         self.timer_grayscale = None

#         # Initialize HSV threshold values
#         self.lower_h = 0
#         self.lower_s = 0
#         self.lower_v = 0
#         self.upper_h = 179
#         self.upper_s = 255
#         self.upper_v = 255

#         # Set the correct range for each slider
#         self.ui.horizontalSlider_LowerH.setMinimum(0)
#         self.ui.horizontalSlider_LowerH.setMaximum(179)
#         self.ui.horizontalSlider_UpperH.setMinimum(0)
#         self.ui.horizontalSlider_UpperH.setMaximum(179)

#         self.ui.horizontalSlider_LowerS.setMinimum(0)
#         self.ui.horizontalSlider_LowerS.setMaximum(255)
#         self.ui.horizontalSlider_UpperS.setMinimum(0)
#         self.ui.horizontalSlider_UpperS.setMaximum(255)

#         self.ui.horizontalSlider_LowerV.setMinimum(0)
#         self.ui.horizontalSlider_LowerV.setMaximum(255)
#         self.ui.horizontalSlider_UpperV.setMinimum(0)
#         self.ui.horizontalSlider_UpperV.setMaximum(255)

#         # Set initial values for sliders and labels
#         self.ui.horizontalSlider_LowerH.setValue(self.lower_h)
#         self.ui.horizontalSlider_UpperH.setValue(self.upper_h)
#         self.ui.horizontalSlider_LowerS.setValue(self.lower_s)
#         self.ui.horizontalSlider_UpperS.setValue(self.upper_s)
#         self.ui.horizontalSlider_LowerV.setValue(self.lower_v)
#         self.ui.horizontalSlider_UpperV.setValue(self.upper_v)

#         # Connect sliders to update_hsv_controls
#         self.ui.horizontalSlider_LowerH.valueChanged.connect(self.update_hsv_controls)
#         self.ui.horizontalSlider_LowerS.valueChanged.connect(self.update_hsv_controls)
#         self.ui.horizontalSlider_LowerV.valueChanged.connect(self.update_hsv_controls)
#         self.ui.horizontalSlider_UpperH.valueChanged.connect(self.update_hsv_controls)
#         self.ui.horizontalSlider_UpperS.valueChanged.connect(self.update_hsv_controls)
#         self.ui.horizontalSlider_UpperV.valueChanged.connect(self.update_hsv_controls)

#         # Add a flag to indicate whether HSV values are locked
#         self.hsv_locked = False

#         # Connect the pushButton_HSV click signal to the toggle_hsv_lock method
#         self.ui.pushButton_HSV.clicked.connect(self.toggle_hsv_lock)

#         # Call update_hsv_controls to initialize labels
#         self.update_hsv_controls()

#         # Set default values for contrast and brightness
#         self.default_contrast = 1.0
#         self.default_brightness = 0.0

#         # Apply default contrast and brightness to initialize the displayed frame
#         self.apply_contrast_brightness(self.default_contrast, self.default_brightness)

#         # Connect sliders to update methods
#         self.ui.horizontalSlider_Contrast.valueChanged.connect(self.update_contrast)
#         self.ui.horizontalSlider_Brightness.valueChanged.connect(self.update_brightness)

#     def update_hsv_controls(self):
#         # Check if HSV values are locked
#         if not self.hsv_locked:
#             # Get slider values
#             self.lower_h = self.ui.horizontalSlider_LowerH.value()
#             self.lower_s = self.ui.horizontalSlider_LowerS.value()
#             self.lower_v = self.ui.horizontalSlider_LowerV.value()
#             self.upper_h = self.ui.horizontalSlider_UpperH.value()
#             self.upper_s = self.ui.horizontalSlider_UpperS.value()
#             self.upper_v = self.ui.horizontalSlider_UpperV.value()

#             # Update labels
#             self.ui.label_LowerH.setText(f"Lower H: {self.lower_h}")
#             self.ui.label_LowerS.setText(f"Lower S: {self.lower_s}")
#             self.ui.label_LowerV.setText(f"Lower V: {self.lower_v}")
#             self.ui.label_UpperH.setText(f"Upper H: {self.upper_h}")
#             self.ui.label_UpperS.setText(f"Upper S: {self.upper_s}")
#             self.ui.label_UpperV.setText(f"Upper V: {self.upper_v}")

#     def open_video_file(self):
#         file_info = QFileDialog.getOpenFileUrl(self, 'Open Video', '', 'Videos (*.mp4 *.avi *.mov)')
#         if not file_info or not isinstance(file_info, tuple) or len(file_info) < 1:
#             return

#         filepath = file_info[0].toLocalFile()
#         self.cap = cv2.VideoCapture(filepath)
#         if not self.cap.isOpened():
#             return

#         ret, frame = self.cap.read()
#         if ret:
#             self.display_frame(frame)

#     def start_video_playback(self):
#         if self.cap is not None:
#             if not self.timer.isActive():
#                 # Start the timer if it's not already active
#                 self.timer.start(33)
#                 self.ui.pushButton.setText("Pause")  # Change button text to Pause
#             else:
#                 # Stop the timer if it's active
#                 self.timer.stop()
#                 self.ui.pushButton.setText("Play")  # Change button text to Play

#     def update_video_frame(self):   
#         ret, frame = self.cap.read()
#         if ret:
#             self.display_frame(frame)
#         else:
#             # Stop the timer when the video ends
#             self.timer.stop()

#             # Reset the video capture to play again
#             self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

#             # Uncomment the line below if you want to restart the video automatically
#             # self.timer.start(33)

#     def display_frame(self, frame):
#         # Convert frame to HSV color space
#         frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

#         # Apply HSV thresholding
#         lower_bound = np.array([self.lower_h, self.lower_s, self.lower_v])
#         upper_bound = np.array([self.upper_h, self.upper_s, self.upper_v])
#         mask = cv2.inRange(frame_hsv, lower_bound, upper_bound)
#         result_frame = cv2.bitwise_and(frame, frame, mask=mask)

#         # Display the processed frame
#         height, width, channel = result_frame.shape
#         bytesPerLine = 3 * width

#         # Maintain aspect ratio during resizing
#         qImg = QImage(result_frame.data, width, height, bytesPerLine, QImage.Format_RGB888)
#         qImg = qImg.scaled(self.ui.video_label.size(), QtCore.Qt.KeepAspectRatio)

#         self.ui.video_label.setPixmap(QPixmap.fromImage(qImg))

#     def toggle_hsv_lock(self):
#         # Toggle the lock state
#         self.hsv_locked = not self.hsv_locked

#     def process_and_display_grayscale_frames(self):
#         if self.cap is not None:
#             self.timer.stop()  # Stop the main video playback timer

#             # Reset video capture to the beginning
#             self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

#             # Start a new timer for grayscale processing
#             self.timer_grayscale = QtCore.QTimer(self)
#             self.timer_grayscale.timeout.connect(self.update_and_display_grayscale_frames)
#             self.timer_grayscale.start(33)

#     def update_and_display_grayscale_frames(self):
#         ret, frame = self.cap.read()
#         if ret:
#             # Convert frame to grayscale
#             grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#             # Convert single-channel (grayscale) frame to RGB for display
#             result_frame_rgb = cv2.cvtColor(grayscale_frame, cv2.COLOR_GRAY2RGB)

#             # Display the processed frame in video_label2
#             height, width, channel = result_frame_rgb.shape
#             bytesPerLine = 3 * width
#             qImg = QImage(result_frame_rgb.data, width, height, bytesPerLine, QImage.Format_RGB888)
#             qImg = qImg.scaled(self.ui.video_label2.size(), QtCore.Qt.KeepAspectRatio)
#             self.ui.video_label2.setPixmap(QPixmap.fromImage(qImg))
#         else:
#             # Stop the timer when the video ends
#             self.timer_grayscale.stop()
#             self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     widget = MainWindow()
#     widget.show()
#     sys.exit(app.exec())
