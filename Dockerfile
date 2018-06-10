FROM ubuntu:16.04

ENV DEBIAN_FRONTEND noninteractive

# Install apt dependencies
RUN apt-get update && apt-get install -y \
    locales \
    python3 \
    g++ \
    git \
    cmake \
    wget \
    fish \
    qt5-default \
    qtbase5-dev \
    && rm -rf /var/lib/apt/lists/*

# Install OpenCV
RUN cd /tmp \
    && git clone --branch 3.4.1 --depth=1 --single-branch https://github.com/opencv/opencv.git \
    && git clone --branch 3.4.1 --depth=1 --single-branch https://github.com/opencv/opencv_contrib.git \
    && cd /tmp/opencv \
    && mkdir release && cd release \
    && cmake -D CMAKE_BUILD_TYPE=RELEASE \
             -D CMAKE_INSTALL_PREFIX=/usr/local \
             -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
             -D WITH_QT=ON \
             -D WITH_GTK=OFF \
             -D WITH_JASPER=OFF \
             -D WITH_FFMPEG=ON \
             -D ENABLE_PRECOMPILED_HEADERS=OFF \
             -D WITH_TBB=ON \
             -D WITH_IPP=ON \
             -D WITH_CUDA=OFF \
             -D WITH_OPENCL=OFF \
             -D BUILD_PERF_TESTS=OFF \
             -D BUILD_TESTS=OFF \
             -D WITH_1394=OFF \
             -D BUILD_opencv_legacy=OFF \
             -D BUILD_opencv_dnn_modern=OFF \
             -D BUILD_opencv_dnn=ON \
             -D BUILD_opencv_apps=OFF \
             -D BUILD_opencv_java=OFF \
             -D BUILD_SHARED_LIBS=OFF \
             -D BUILD_DOCS=OFF \
             -D BUILD_TIFF=ON \
             -D BUILD_JPEG=ON \
             -D BUILD_PNG=ON \
             -D BUILD_OPENEXR=ON \
             -D BUILD_TBB=ON \
             -D ENABLE_CXX11=ON \
             -D CPU_BASELINE=AVX \
             -D CPU_DISPATCH=AVX2 \
             -D ENABLE_FAST_MATH=ON .. \
    && cd /tmp/opencv/release \
    && make -j $(nproc) && make install \
    && /bin/bash -c 'echo "/usr/local/lib" > /etc/ld.so.conf.d/opencv.conf' \
    && /bin/bash -c 'echo "/usr/local/share/OpenCV/3rdparty/lib" >> /etc/ld.so.conf.d/opencv.conf' \
    && ldconfig \
    && cd / \
    && rm -rf /tmp/opencv \
    && rm -rf /tmp/opencv_contrib

# Move source files
COPY app /app

WORKDIR /app
