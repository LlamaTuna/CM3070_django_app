# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Install system dependencies, including CMake and g++
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \ 
    cmake \  
    libasound2-dev \
    libportaudio2 \
    libportaudiocpp0 \
    portaudio19-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    alsa-utils \
    alsa-oss \
    pulseaudio \
    v4l-utils \
    libhdf5-dev \
    ffmpeg \  
    && rm -rf /var/lib/apt/lists/*

# Install TensorFlow
RUN pip install --no-cache-dir tensorflow tensorflow-hub

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
ADD . /app

# Copy the dlib shape predictor model file to the appropriate directory
COPY camera/models/shape_predictor_68_face_landmarks.dat /app/camera/models/

# Add MobileNetV3 model files to the appropriate directory
COPY camera/models/mobilenet/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt /app/camera/models/mobilenet/
COPY camera/models/mobilenet/frozen_inference_graph.pb /app/camera/models/mobilenet/

# Use a reliable PyPI mirror
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Create ALSA configuration files
RUN echo "pcm.!default {\n type hw\n card 0\n}\nctl.!default {\n type hw\n card 0\n}" > /etc/asound.conf
RUN echo "@hooks [\n{\n func load\n files [\n\"/etc/asound.conf\"\n ]\n errors false\n}\n]" > /usr/share/alsa/alsa.conf

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME=World

# Run app.py when the container launches
CMD ["python", "manage.py", "runserver", "0.0.0.0:80"]
