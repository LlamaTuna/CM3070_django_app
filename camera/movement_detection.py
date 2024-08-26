import cv2
import numpy as np

class MovementDetection:
    """
    A class for detecting movement in video frames by comparing the difference between
    the current frame and the previous frame.

    Attributes:
        previous_frame (ndarray): The grayscale image of the previous frame.
    """

    def __init__(self):
        """
        Initializes the MovementDetection class with no previous frame.
        """
        self.previous_frame = None

    def detect_movement(self, frame):
        """
        Detects movement in the current frame by comparing it to the previous frame.

        Args:
            frame (ndarray): The current video frame in which movement is to be detected.

        Returns:
            tuple: A tuple containing a boolean indicating whether movement was detected,
                   and the bounding box (x, y, w, h) of the detected movement, or None if no movement is detected.
        """
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # If there is no previous frame, store the current frame and return no movement
        if self.previous_frame is None:
            self.previous_frame = gray
            return False, None

        # Compute the absolute difference between the current frame and the previous frame
        frame_diff = cv2.absdiff(self.previous_frame, gray)
        self.previous_frame = gray

        # Apply thresholding and dilation to highlight regions of movement
        thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < 500:
                continue
            (x, y, w, h) = cv2.boundingRect(contour)
            return True, (x, y, w, h)

        # Return no movement detected if no contours meet the criteria
        return False, None
