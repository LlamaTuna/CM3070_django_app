import os
import cv2
import numpy as np
from collections import deque
from django.conf import settings

class ObjectClassifier:
    """
    A class used to perform object classification using a MobileNetV3 model trained on the COCO dataset.

    Attributes:
        configPath (str): Path to the configuration file of the MobileNetV3 model.
        weightsPath (str): Path to the pre-trained weights of the MobileNetV3 model.
        classFile (str): Path to the file containing the class names of the COCO dataset.
        classNames (list): List of class names from the COCO dataset.
        net (cv2.dnn_DetectionModel): The MobileNetV3 model used for object detection.
        confidence_threshold (float): The confidence threshold for filtering predictions.
        prediction_buffer (deque): A buffer to store recent predictions for smoothing.
        buffer_size (int): The size of the prediction buffer.
    """

    def __init__(self, buffer_size=15, confidence_threshold=0.5):
        """
        Initializes the ObjectClassifier by loading the MobileNetV3 model and its configuration.

        Args:
            buffer_size (int): The size of the buffer for smoothing predictions. Default is 10.
            confidence_threshold (float): The minimum confidence required for a prediction to be considered. Default is 0.5.
        """
        # Load the MobileNetV3 model from your local files
        model_dir = os.path.join(settings.MODEL_DIR, 'mobilenet')
        self.configPath = os.path.join(model_dir, 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt')
        self.weightsPath = os.path.join(model_dir, 'frozen_inference_graph.pb')

        # Load the class names (COCO dataset classes)
        self.classFile = os.path.join(model_dir, 'coco.names')
        with open(self.classFile, "rt") as f:
            self.classNames = f.read().rstrip("\n").split("\n")

        # Set up the MobileNetV3 model for object detection
        self.net = cv2.dnn_DetectionModel(self.weightsPath, self.configPath)
        self.net.setInputSize(320, 320)
        self.net.setInputScale(1.0 / 127.5)
        self.net.setInputMean((127.5, 127.5, 127.5))
        self.net.setInputSwapRB(True)

        print("ObjectClassifier: Loaded MobileNetV3 model for object classification")

        # Define the confidence threshold for predictions
        self.confidence_threshold = confidence_threshold

        # Buffer for smoothing predictions
        self.prediction_buffer = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size

    def classify_object(self, image):
        """
        Classifies objects in the provided image using the loaded MobileNetV3 model.

        Args:
            image (ndarray): The image in which objects need to be classified.

        Returns:
            str: The label of the detected object with the highest average confidence over the buffer.
                 Returns 'unknown' if no confident prediction is made.
        """
        # Perform object detection
        classIds, confs, bbox = self.net.detect(image, confThreshold=self.confidence_threshold, nmsThreshold=0.4)

        predictions = {}
        if len(classIds) != 0:
            for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                className = self.classNames[classId - 1]
                if confidence >= self.confidence_threshold:
                    predictions[className] = confidence

        # Add the predictions to the buffer
        self.prediction_buffer.append(predictions)

        # Smooth the predictions over the buffer
        averaged_predictions = {class_name: 0 for class_name in self.classNames}
        for preds in self.prediction_buffer:
            for class_name in preds:
                averaged_predictions[class_name] += preds[class_name]

        # Determine the final prediction based on the highest average confidence
        final_label = max(averaged_predictions, key=averaged_predictions.get)

        # Return the final label, or 'unknown' if no confident prediction was made
        if averaged_predictions[final_label] > 0:
            return final_label
        else:
            return 'unknown'

    def annotate_image(self, image, text, position=(10, 50), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, color=(255, 255, 255), thickness=2):
        """
        Annotates the image with the specified text.

        Args:
            image (ndarray): The image frame to annotate.
            text (str): The text to annotate on the image.
            position (tuple): A tuple (x, y) for the position of the text. Default is (10, 50).
            font (int): The font type to use for the text. Default is cv2.FONT_HERSHEY_SIMPLEX.
            font_scale (float): The scale of the text font. Default is 1.
            color (tuple): The color of the text in (B, G, R) format. Default is (255, 255, 255).
            thickness (int): The thickness of the text. Default is 2.
        """
        cv2.putText(image, text, position, font, font_scale, color, thickness)


