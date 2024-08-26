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

    def __init__(self, buffer_size=10, confidence_threshold=0.5):
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


#*************** yolo8n ****************
# import os
# import django
# from django.conf import settings
# import cv2
# import numpy as np
# from collections import deque

# # Set up Django environment manually
# os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'camera_app.settings')
# django.setup()

# class ObjectClassifier:
#     def __init__(self, buffer_size= 10, confidence_threshold=0.2, nms_threshold=0.4):
#         # Load class names
#         self.classes = self._load_class_names()

#         # Path to the YOLOv8 ONNX model
#         model_path = os.path.join(settings.MODEL_DIR, 'yolo/yolov8n.onnx')

#         try:
#             print("Loading YOLOv8 ONNX model from:", model_path)
#             self.net = cv2.dnn.readNetFromONNX(model_path)
#             print("YOLOv8 model loaded successfully")
#         except cv2.error as e:
#             print(f"Error loading YOLOv8 model: {e}")
#             self.net = None  # Ensure that self.net is defined even in case of error

#         # Ensure net is not None before proceeding
#         if self.net is None:
#             raise ValueError("Failed to load YOLOv8 model")

#         # Confidence threshold for predictions
#         self.confidence_threshold = confidence_threshold
#         self.nms_threshold = nms_threshold

#         # Buffer for smoothing predictions
#         self.prediction_buffer = deque(maxlen=buffer_size)
#         self.buffer_size = buffer_size

#         # Interested categories related to human recognition
#         self.interested_categories = {'person': 'person'}

#     def _load_class_names(self):
#         # List of COCO class names
#         return [
#             "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", 
#             "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", 
#             "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", 
#             "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", 
#             "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", 
#             "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", 
#             "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", 
#             "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", 
#             "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", 
#             "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", 
#             "toothbrush"
#         ]

#     def classify_object(self, image):
#         # Prepare the image for YOLOv8
#         blob = cv2.dnn.blobFromImage(image, 1/255.0, (640, 640), swapRB=True, crop=False)
#         self.net.setInput(blob)
#         outputs = self.net.forward()

#         # Process detections (YOLOv8 has a slightly different output format)
#         boxes, confidences, class_ids = self._process_detections(outputs, image.shape[:2])

#         # Apply non-maxima suppression to remove redundant overlapping boxes with lower confidences
#         indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)

#         predictions = self._extract_predictions(indices, class_ids, confidences)

#         # Add the predictions to the buffer
#         self.prediction_buffer.append(predictions)

#         # Smooth the predictions over the buffer
#         final_label = self._smooth_predictions()

#         # Return the final label, or 'unknown' if no confident prediction was made
#         return final_label if final_label else 'unknown'

#     def _process_detections(self, outputs, image_shape):
#         h, w = image_shape
#         boxes, confidences, class_ids = [], [], []

#         for i in range(outputs.shape[2]):
#             detection = outputs[0, :, i]
#             scores = detection[5:]  # Skip the first 5 elements (bbox + obj score)
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]

#             # Ensure the class_id is valid
#             if class_id >= len(self.classes):
#                 print(f"Invalid class id: {class_id} - skipping detection")
#                 continue

#             if confidence > self.confidence_threshold:
#                 x_center, y_center, width, height = detection[:4] * np.array([w, h, w, h])
#                 x = int(x_center - (width / 2))
#                 y = int(y_center - (height / 2))

#                 boxes.append([x, y, int(width), int(height)])
#                 confidences.append(float(confidence))
#                 class_ids.append(class_id)

#         return boxes, confidences, class_ids



#     def _extract_predictions(self, indices, class_ids, confidences):
#         predictions = {}
#         if len(indices) > 0:
#             for i in indices.flatten():
#                 if class_ids[i] < len(self.classes):  # Check if index is valid
#                     label = self.classes[class_ids[i]]
#                     confidence = confidences[i]
#                     predictions[label] = confidence  # Log the detected object and its confidence
#                     print(f"Detected: {label} with confidence: {confidence:.2f}")
#                 else:
#                     print(f"Invalid class id: {class_ids[i]}")
#         return predictions

#     def _smooth_predictions(self):
#         averaged_predictions = {category: 0 for category in self.interested_categories.keys()}
#         for preds in self.prediction_buffer:
#             for category in preds:
#                 if category in averaged_predictions:
#                     averaged_predictions[category] += preds[category]

#         final_label = max(averaged_predictions, key=averaged_predictions.get)
#         return final_label if averaged_predictions[final_label] > 0 else None

#     def annotate_image(self, image, text, position=(10, 50), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, color=(255, 255, 255), thickness=2):
#         """Annotates the image with the specified text."""
#         cv2.putText(image, text, position, font, font_scale, color, thickness)



# ****** broken yolo10n ********
# import os
# import onnxruntime as ort
# import numpy as np
# import cv2
# from django.conf import settings

# class ObjectClassifier:
#     def __init__(self, model_filename='yolov10n.onnx', confidence_threshold=0.2, nms_threshold=0.4):
#         self.confidence_threshold = confidence_threshold
#         self.nms_threshold = nms_threshold

#         # Path to the YOLO ONNX model
#         model_path = os.path.join(settings.MODEL_DIR, 'yolo', model_filename)

#         if not os.path.exists(model_path):
#             raise ValueError(f"Model file does not exist: {model_path}")

#         print(f"Loading YOLO model from: {model_path}")
#         self.session = ort.InferenceSession(model_path)
#         print("YOLO model loaded successfully")

#     def classify_object(self, image):
#         # Prepare the image for YOLO model
#         input_blob = cv2.dnn.blobFromImage(image, 1/255.0, (640, 640), swapRB=True, crop=False)
#         input_name = self.session.get_inputs()[0].name
#         outputs = self.session.run(None, {input_name: input_blob})

#         # Process detections
#         boxes, confidences, class_ids = self._process_detections(outputs, image.shape[:2])

#         # Apply non-maxima suppression
#         indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)

#         # Extract predictions
#         predictions = self._extract_predictions(indices, class_ids, confidences)

#         # Check if any prediction is made
#         if predictions:
#             # Get the label of the prediction with the highest confidence
#             object_label = max(predictions, key=predictions.get)
#         else:
#             object_label = "unknown"

#         return object_label


#     def _process_detections(self, outputs, image_shape):
#         h, w = image_shape
#         boxes, confidences, class_ids = [], [], []

#         for detection in outputs[0]:
#             scores = detection[4:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]

#             # Ensure confidence is a scalar value
#             if np.isscalar(confidence) and confidence > self.confidence_threshold:
#                 x_center, y_center, width, height = detection[:4]
#                 x_center, y_center, width, height = x_center * w, y_center * h, width * w, height * h
#                 x = int(x_center - (width / 2))
#                 y = int(y_center - (height / 2))
#                 boxes.append([x, y, int(width), int(height)])
#                 confidences.append(float(confidence))
#                 class_ids.append(class_id)
#             else:
#                 print(f"Skipping detection with invalid confidence: {confidence}")

#         return boxes, confidences, class_ids



#     def _extract_predictions(self, indices, class_ids, confidences):
#         predictions = {}

#         # Check if indices is valid
#         if indices is None or len(indices) == 0:
#             return predictions  # No valid detections, return empty predictions

#         # Handle case where indices is not a list or array
#         if isinstance(indices, tuple):
#             indices = indices[0]  # Get the first element of the tuple

#         for i in indices.flatten():
#             if class_ids[i] < len(self._load_class_names()):
#                 label = self._load_class_names()[class_ids[i]]
#                 confidence = confidences[i]
#                 predictions[label] = confidence
#                 print(f"Detected: {label} with confidence: {confidence:.2f}")
#             else:
#                 print(f"Invalid class id: {class_ids[i]}")
#         return predictions


#     def _load_class_names(self):
#         return [
#             "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", 
#             "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", 
#             "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", 
#             "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", 
#             "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", 
#             "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", 
#             "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", 
#             "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", 
#             "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", 
#             "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", 
#             "toothbrush"
#         ]

#     def annotate_image(self, image, text, position=(10, 50), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, color=(255, 255, 255), thickness=2):
#         """Annotates the image with the specified text."""
#         cv2.putText(image, text, position, font, font_scale, color, thickness)


# #*************** yolo8n ****************
# import os
# import django
# from django.conf import settings
# import cv2
# import numpy as np
# from collections import deque

# # Set up Django environment manually
# os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'camera_app.settings')
# django.setup()

# class ObjectClassifier:
#     def __init__(self, buffer_size=5, confidence_threshold=0.2, nms_threshold=0.4):
#         # Load class names
#         self.classes = self._load_class_names()

#         # Path to the YOLOv8 ONNX model
#         model_path = os.path.join(settings.MODEL_DIR, 'yolo/yolov8n.onnx')

#         try:
#             print("Loading YOLOv8 ONNX model from:", model_path)
#             self.net = cv2.dnn.readNetFromONNX(model_path)
#             print("YOLOv8 model loaded successfully")
#         except cv2.error as e:
#             print(f"Error loading YOLOv8 model: {e}")
#             self.net = None  # Ensure that self.net is defined even in case of error

#         # Ensure net is not None before proceeding
#         if self.net is None:
#             raise ValueError("Failed to load YOLOv8 model")

#         # Confidence threshold for predictions
#         self.confidence_threshold = confidence_threshold
#         self.nms_threshold = nms_threshold

#         # Buffer for smoothing predictions
#         self.prediction_buffer = deque(maxlen=buffer_size)
#         self.buffer_size = buffer_size

#         # Interested categories related to human recognition
#         self.interested_categories = {'person': 'person'}

#     def _load_class_names(self):
#         # List of COCO class names
#         return [
#             "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", 
#             "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", 
#             "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", 
#             "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", 
#             "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", 
#             "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", 
#             "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", 
#             "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", 
#             "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", 
#             "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", 
#             "toothbrush"
#         ]

#     def classify_object(self, image):
#         # Prepare the image for YOLOv8
#         blob = cv2.dnn.blobFromImage(image, 1/255.0, (640, 640), swapRB=True, crop=False)
#         self.net.setInput(blob)
#         outputs = self.net.forward()

#         # Process detections (YOLOv8 has a slightly different output format)
#         boxes, confidences, class_ids = self._process_detections(outputs, image.shape[:2])

#         # Apply non-maxima suppression to remove redundant overlapping boxes with lower confidences
#         indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)

#         predictions = self._extract_predictions(indices, class_ids, confidences)

#         # Add the predictions to the buffer
#         self.prediction_buffer.append(predictions)

#         # Smooth the predictions over the buffer
#         final_label = self._smooth_predictions()

#         # Return the final label, or 'unknown' if no confident prediction was made
#         return final_label if final_label else 'unknown'

#     def _process_detections(self, outputs, image_shape):
#         h, w = image_shape
#         boxes, confidences, class_ids = [], [], []

#         for detection in outputs[0]:  # YOLOv8 output is typically a single array
#             scores = detection[4:]
#             class_id = np.argmax(scores)
#             print
#             confidence = scores[class_id]

#             print(f"Name of Class ID: {class_id}, Confidence: {confidence}")  # Print to debug

#             if confidence > self.confidence_threshold:
#                 box = detection[:4] * np.array([w, h, w, h])
#                 center_x, center_y, width, height = box.astype('int')

#                 x = int(center_x - (width / 2))
#                 y = int(center_y - (height / 2))

#                 boxes.append([x, y, int(width), int(height)])
#                 confidences.append(float(confidence))
#                 class_ids.append(class_id)

#         return boxes, confidences, class_ids


#     def _extract_predictions(self, indices, class_ids, confidences):
#         predictions = {}
#         if len(indices) > 0:
#             for i in indices.flatten():
#                 if class_ids[i] < len(self.classes):  # Check if index is valid
#                     label = self.classes[class_ids[i]]
#                     confidence = confidences[i]
#                     predictions[label] = confidence  # Log the detected object and its confidence
#                     print(f"Detected: {label} with confidence: {confidence:.2f}")
#                 else:
#                     print(f"Invalid class id: {class_ids[i]}")
#         return predictions

#     def _smooth_predictions(self):
#         averaged_predictions = {category: 0 for category in self.interested_categories.keys()}
#         for preds in self.prediction_buffer:
#             for category in preds:
#                 if category in averaged_predictions:
#                     averaged_predictions[category] += preds[category]

#         final_label = max(averaged_predictions, key=averaged_predictions.get)
#         return final_label if averaged_predictions[final_label] > 0 else None

#     def annotate_image(self, image, text, position=(10, 50), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, color=(255, 255, 255), thickness=2):
#         """Annotates the image with the specified text."""
#         cv2.putText(image, text, position, font, font_scale, color, thickness)

# # Usage example
# if __name__ == "__main__":
#     camera = cv2.VideoCapture(0)
#     classifier = ObjectClassifier()

#     while True:
#         ret, frame = camera.read()
#         if not ret:
#             break

#         label = classifier.classify_object(frame)
#         classifier.annotate_image(frame, label)

#         cv2.imshow('Frame', frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     camera.release()
#     cv2.destroyAllWindows()

#*************** resnet50 ***************
# import os
# import django
# from django.conf import settings
# import cv2
# import numpy as np
# from collections import deque
# from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
# from tensorflow.keras.preprocessing.image import img_to_array

# # Set up Django environment manually
# os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'camera_app.settings')
# django.setup()

# class ObjectClassifier:

#     def __init__(self, buffer_size=10, confidence_threshold=0.5):
#         # Load ResNet50 model
#         self.model = ResNet50(weights='imagenet')
#         print("ObjectClassifier: Loaded ResNet50 model for object classification")

#         # Confidence threshold for predictions
#         self.confidence_threshold = confidence_threshold

#         # Buffer for smoothing predictions
#         self.prediction_buffer = deque(maxlen=buffer_size)
#         self.buffer_size = buffer_size

#     def classify_object(self, image):
#         # Preprocess the image for ResNet50
#         img = cv2.resize(image, (224, 224))  # Resize to match ResNet50 input size
#         img = img_to_array(img)  # Convert image to array
#         img = np.expand_dims(img, axis=0)  # Add batch dimension
#         img = preprocess_input(img)  # Preprocess the image

#         # Predict the object category
#         preds = self.model.predict(img)

#         # Decode predictions
#         decoded_preds = decode_predictions(preds, top=3)[0]  # Get the top 3 predictions

#         # Initialize an empty dictionary to hold predictions
#         predictions = {}

#         # Filter predictions to apply confidence threshold
#         for _, label, confidence in decoded_preds:
#             if confidence >= self.confidence_threshold:
#                 predictions[label] = confidence
#                 print(f"Detected: {label} with confidence: {confidence:.2f}")

#         # Add the predictions to the buffer
#         self.prediction_buffer.append(predictions)

#         # Smooth the predictions over the buffer
#         final_label = self._smooth_predictions()

#         # Return the final label, or 'unknown' if no confident prediction was made
#         return final_label if final_label else 'unknown'

#     def _smooth_predictions(self):
#         averaged_predictions = {}
#         for preds in self.prediction_buffer:
#             for category, confidence in preds.items():
#                 if category in averaged_predictions:
#                     averaged_predictions[category] += confidence
#                 else:
#                     averaged_predictions[category] = confidence

#         final_label = max(averaged_predictions, key=averaged_predictions.get, default=None)
#         return final_label if final_label and averaged_predictions[final_label] > 0 else None

#     def annotate_image(self, image, text, position=(10, 50), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, color=(255, 255, 255), thickness=2):
#         """Annotates the image with the specified text."""
#         cv2.putText(image, text, position, font, font_scale, color, thickness)

# # Usage example
# if __name__ == "__main__":
#     camera = cv2.VideoCapture(0)
#     classifier = ObjectClassifier()

#     while True:
#         ret, frame = camera.read()
#         if not ret:
#             break

#         label = classifier.classify_object(frame)
#         classifier.annotate_image(frame, label)

#         cv2.imshow('Frame', frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     camera.release()
#     cv2.destroyAllWindows()

#//// yolov3-tiny /////////

# import os
# import django
# from django.conf import settings
# import cv2
# import numpy as np
# from collections import deque

# # Set up Django environment manually
# os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'camera_app.settings')
# django.setup()

# class ObjectClassifier:
#     def __init__(self, buffer_size=5, confidence_threshold=0.2, nms_threshold=0.4):
#         # Load class names
#         self.classes = self._load_class_names()

#         # Use Django settings to locate the model files
#         cfg_path = os.path.join(settings.MODEL_DIR, 'yolo/yolov3-tiny.cfg')
#         weights_path = os.path.join(settings.MODEL_DIR, 'yolo/yolov3-tiny.weights')

#         try:
#             print("Loading YOLO model from:", cfg_path, "and", weights_path)
#             self.net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
#             print("YOLO model loaded successfully")
#         except cv2.error as e:
#             print(f"Error loading YOLO model: {e}")
#             self.net = None  # Ensure that self.net is defined even in case of error

#         # Ensure net is not None before proceeding
#         if self.net is None:
#             raise ValueError("Failed to load YOLO model")

#         self.layer_names = self.net.getLayerNames()
#         unconnected_out_layers = self.net.getUnconnectedOutLayers()

#         if isinstance(unconnected_out_layers, np.ndarray) and len(unconnected_out_layers.shape) == 2:
#             self.output_layers = [self.layer_names[i[0] - 1] for i in unconnected_out_layers]
#         else:
#             self.output_layers = [self.layer_names[i - 1] for i in unconnected_out_layers]

#         # Confidence threshold for predictions
#         self.confidence_threshold = confidence_threshold
#         self.nms_threshold = nms_threshold

#         # Buffer for smoothing predictions
#         self.prediction_buffer = deque(maxlen=buffer_size)
#         self.buffer_size = buffer_size

#         # Interested categories related to human recognition
#         self.interested_categories = {'person': 'person'}

#     def _load_class_names(self):
#         # List of COCO class names
#         return [
#             "person", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", 
#             "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", 
#             "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", 
#             "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "cell phone", "laptop", 
#             "mouse", "remote", "keyboard", "book", "clock", "teddy bear", "hair drier", "toothbrush", 
#             "glasses"  # Add "glasses" if it's included in the custom list for better detection
# ]


#     def classify_object(self, image):
#         # Prepare the image for YOLOv3-Tiny
#         blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
#         self.net.setInput(blob)
#         outputs = self.net.forward(self.output_layers)

#         # Initialize lists to hold detected bounding boxes, confidences, and class IDs
#         boxes, confidences, class_ids = self._process_detections(outputs, image.shape[:2])

#         # Apply non-maxima suppression to remove redundant overlapping boxes with lower confidences
#         indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)

#         predictions = self._extract_predictions(indices, class_ids, confidences)

#         # Add the predictions to the buffer
#         self.prediction_buffer.append(predictions)

#         # Smooth the predictions over the buffer
#         final_label = self._smooth_predictions()

#         # Return the final label, or 'unknown' if no confident prediction was made
#         return final_label if final_label else 'unknown'

#     def _process_detections(self, outputs, image_shape):
#         h, w = image_shape
#         boxes, confidences, class_ids = [], [], []

#         for output in outputs:
#             for detection in output:
#                 scores = detection[5:]
#                 class_id = np.argmax(scores)
#                 confidence = scores[class_id]

#                 if confidence > self.confidence_threshold:
#                     box = detection[0:4] * np.array([w, h, w, h])
#                     center_x, center_y, width, height = box.astype('int')

#                     x = int(center_x - (width / 2))
#                     y = int(center_y - (height / 2))

#                     boxes.append([x, y, int(width), int(height)])
#                     confidences.append(float(confidence))
#                     class_ids.append(class_id)

#         return boxes, confidences, class_ids

#     def _extract_predictions(self, indices, class_ids, confidences):
#         predictions = {}
#         if len(indices) > 0:
#             for i in indices.flatten():
#                 if class_ids[i] < len(self.classes):  # Check if index is valid
#                     label = self.classes[class_ids[i]]
#                     confidence = confidences[i]
#                     predictions[label] = confidence  # Log the detected object and its confidence
#                     print(f"Detected: {label} with confidence: {confidence:.2f}")
#                 else:
#                     print(f"Invalid class id: {class_ids[i]}")
#         return predictions


#     def _smooth_predictions(self):
#         averaged_predictions = {category: 0 for category in self.interested_categories.keys()}
#         for preds in self.prediction_buffer:
#             for category in preds:
#                 if category in averaged_predictions:
#                     averaged_predictions[category] += preds[category]

#         final_label = max(averaged_predictions, key=averaged_predictions.get)
#         return final_label if averaged_predictions[final_label] > 0 else None


#     def annotate_image(self, image, text, position=(10, 50), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, color=(255, 255, 255), thickness=2):
#         """Annotates the image with the specified text."""
#         cv2.putText(image, text, position, font, font_scale, color, thickness)

# # Usage example
# if __name__ == "__main__":
#     camera = cv2.VideoCapture(0)
#     classifier = ObjectClassifier()

#     while True:
#         ret, frame = camera.read()
#         if not ret:
#             break

#         label = classifier.classify_object(frame)
#         classifier.annotate_image(frame, label)

#         cv2.imshow('Frame', frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     camera.release()
#     cv2.destroyAllWindows()




# import cv2
# import numpy as np
# from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
# from tensorflow.keras.preprocessing.image import img_to_array
# from collections import deque

# class ObjectClassifier:
#     def __init__(self, buffer_size=5, confidence_threshold=0.2):
#         # Load the pre-trained MobileNetV2 model for object classification
#         self.model = MobileNetV2(weights='imagenet')
#         print("ObjectClassifier: Loaded MobileNetV2 model for object classification")

#         # Define the categories related to human recognition
#         self.interested_categories = {
#             'person': ['person', 'man', 'woman', 'boy', 'girl', 'baby'],
#             'age': ['baby', 'boy', 'girl', 'man', 'woman'],
#             'gender': ['man', 'woman', 'boy', 'girl'],
#             'body_part': ['face', 'head', 'foot', 'hand', 'arm', 'leg', 'ear', 'eye', 'mouth']
#         }

#         # Confidence threshold for predictions
#         self.confidence_threshold = confidence_threshold

#         # Buffer for smoothing predictions
#         self.prediction_buffer = deque(maxlen=buffer_size)
#         self.buffer_size = buffer_size

#     def classify_object(self, image):
#         # Preprocess the image for MobileNetV2
#         img = cv2.resize(image, (224, 224))  # Resize to match MobileNetV2 input size
#         img = img_to_array(img)  # Convert image to array
#         img = np.expand_dims(img, axis=0)  # Add batch dimension
#         img = preprocess_input(img)  # Preprocess the image

#         # Predict the object category
#         preds = self.model.predict(img)
#         decoded_preds = decode_predictions(preds, top=3)[0]  # Get the top 3 predictions

#         print("Decoded Predictions:", decoded_preds)  # Print all predictions and their confidence levels

#         # Filter predictions to the categories of interest and apply confidence threshold
#         predictions = {}
#         for _, label, confidence in decoded_preds:
#             print(f"Detected: {label} with confidence: {confidence:.2f}")  # Log every prediction
#             if confidence >= self.confidence_threshold:
#                 if label in self.interested_categories['person']:
#                     predictions['person'] = confidence
#                 elif label in self.interested_categories['age']:
#                     predictions['age'] = confidence
#                 elif label in self.interested_categories['gender']:
#                     predictions['gender'] = confidence
#                 elif label in self.interested_categories['body_part']:
#                     predictions['body_part'] = confidence

#         # Add the predictions to the buffer
#         self.prediction_buffer.append(predictions)

#         # Smooth the predictions over the buffer
#         averaged_predictions = {category: 0 for category in ['person', 'age', 'gender', 'body_part']}
#         for preds in self.prediction_buffer:
#             for category in preds:
#                 averaged_predictions[category] += preds[category]

#         # Determine the final prediction based on the highest average confidence
#         final_label = max(averaged_predictions, key=averaged_predictions.get)

#         # Return the final label, or 'unknown' if no confident prediction was made
#         if averaged_predictions[final_label] > 0:
#             return final_label
#         else:
#             return 'unknown'

#     def annotate_image(self, image, text, position=(10, 50), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, color=(255, 255, 255), thickness=2):
#         """
#         Annotates the image with the specified text.
        
#         Args:
#             image: The image frame to annotate.
#             text: The text to annotate on the image.
#             position: A tuple (x, y) for the position of the text.
#             font: The font type to use for the text.
#             font_scale: The scale of the text font.
#             color: The color of the text (B, G, R).
#             thickness: The thickness of the text.
#         """
#         cv2.putText(image, text, position, font, font_scale, color, thickness)

#********** Mobilenetv3 **********
# import cv2
# import numpy as np
# import tensorflow as tf
# import tensorflow_hub as hub  # Import TensorFlow Hub
# from tensorflow.keras.applications.mobilenet_v3 import preprocess_input, decode_predictions 
# from tensorflow.keras.preprocessing.image import img_to_array
# from collections import deque

# class ObjectClassifier:
#     def __init__(self, buffer_size=10, confidence_threshold=1):
#         # Load the MobileNetV3Large model from TensorFlow Hub
#         self.model = tf.keras.Sequential([
#             hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v3_large_100_224/classification/5")
#         ])
#         self.model.build([None, 224, 224, 3])
#         print("ObjectClassifier: Loaded MobileNetV3Large model for object classification")

#         # Define the categories related to human recognition
#         self.interested_categories = {
#             'person': ['person', 'man', 'woman', 'boy', 'girl', 'baby'],
#             'age': ['baby', 'boy', 'girl', 'man', 'woman'],
#             'gender': ['man', 'woman', 'boy', 'girl'],
#             'body_part': ['face', 'head', 'foot', 'hand', 'arm', 'leg', 'ear', 'eye', 'mouth']
#         }

#         # Confidence threshold for predictions
#         self.confidence_threshold = confidence_threshold

#         # Buffer for smoothing predictions
#         self.prediction_buffer = deque(maxlen=buffer_size)
#         self.buffer_size = buffer_size

#     def classify_object(self, image):
#         # Preprocess the image for MobileNetV3
#         img = cv2.resize(image, (224, 224))  # Resize to match MobileNetV3 input size
#         img = img_to_array(img)  # Convert image to array
#         img = np.expand_dims(img, axis=0)  # Add batch dimension
#         img = preprocess_input(img)  # Preprocess the image

#         # Predict the object category
#         preds = self.model.predict(img)

#         # Slice the predictions to only keep the first 1000 classes
#         preds = preds[:, :1000]

#         # Apply softmax to convert logits to probabilities
#         probabilities = tf.nn.softmax(preds, axis=-1).numpy()

#         # Decode the predictions
#         decoded_preds = decode_predictions(probabilities, top=3)[0]  # Get the top 3 predictions

#         print("Decoded Predictions:", decoded_preds)  # Print all predictions and their confidence levels
#         # Filter predictions to the categories of interest and apply confidence threshold
#         predictions = {}
#         for _, label, confidence in decoded_preds:
#             print(f"Detected: {label} with confidence: {confidence:.2f}")  # Log every prediction
#             if confidence >= self.confidence_threshold:
#                 if label in self.interested_categories['person']:
#                     predictions['person'] = confidence
#                 elif label in self.interested_categories['age']:
#                     predictions['age'] = confidence
#                 elif label in self.interested_categories['gender']:
#                     predictions['gender'] = confidence
#                 elif label in self.interested_categories['body_part']:
#                     predictions['body_part'] = confidence

#         # Add the predictions to the buffer
#         self.prediction_buffer.append(predictions)

#         # Smooth the predictions over the buffer
#         averaged_predictions = {category: 0 for category in ['person', 'age', 'gender', 'body_part']}
#         for preds in self.prediction_buffer:
#             for category in preds:
#                 averaged_predictions[category] += preds[category]

#         # Determine the final prediction based on the highest average confidence
#         final_label = max(averaged_predictions, key=averaged_predictions.get)

#         # Return the final label, or 'unknown' if no confident prediction was made
#         if averaged_predictions[final_label] > 0:
#             return final_label
#         else:
#             return 'unknown'

#     def annotate_image(self, image, text, position=(10, 50), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, color=(255, 255, 255), thickness=2):
#         """
#         Annotates the image with the specified text.
        
#         Args:
#             image: The image frame to annotate.
#             text: The text to annotate on the image.
#             position: A tuple (x, y) for the position of the text.
#             font: The font type to use for the text.
#             font_scale: The scale of the text font.
#             color: The color of the text (B, G, R).
#             thickness: The thickness of the text.
#         """

#         cv2.putText(image, text, position, font, font_scale, color, thickness)
