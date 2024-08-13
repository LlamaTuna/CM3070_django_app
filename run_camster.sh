
    #!/bin/bash

    # Start the base command
    cmd="docker run -d \
        --name camster \
        --device /dev/snd \
        --group-add audio \
        -e PULSE_SERVER=unix:${XDG_RUNTIME_DIR}/pulse/native \
        -v ${XDG_RUNTIME_DIR}/pulse/native:${XDG_RUNTIME_DIR}/pulse/native \
        -v ~/.config/pulse/cookie:/root/.config/pulse/cookie \
        -p 8000:80"

    # Check if /dev/video0 exists and add it to the command
    if [ -c /dev/video0 ]; then
        cmd="$cmd --device=/dev/video0"
    fi

    # Check if /dev/video1 exists and add it to the command
    if [ -c /dev/video1 ]; then
        cmd="$cmd --device=/dev/video1"
    fi

    # Check if /dev/video2 exists and add it to the command
    if [ -c /dev/video2 ]; then
        cmd="$cmd --device=/dev/video2"
    fi

    # Add the image name at the end of the command
    cmd="$cmd snarkvader/camster:latest"

    # Execute the final command
    eval $cmd