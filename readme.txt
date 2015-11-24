--------------------------------------------
---              SVI_MAPPER              ---
--------------------------------------------
version: 1.0



--------------------------------------------
dependencies:

     OpenCV 2.4.11: https://github.com/Itseez/opencv/archive/2.4.11.zip
               g2o: https://github.com/schdomin/g2o
thin_visensor_node: https://github.com/grisetti/thin_drivers
        fps_mapper: https://www.google.ch (txt_io library)
               ROS: http://www.ros.org (dataset acquisition)



--------------------------------------------
build sequence example (shell in project root):

mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE -D OpenCV_DIR="/home/dom/libraries/cpp/opencv-2.4.11/release" -D G2O_ROOT="/home/dom/libraries/cpp/g2o" -D FPS_MAPPER_INCLUDE_DIR="/home/dom/workspace_catkin/src/fps_mapper/src" -D FPS_MAPPER_TXT_IO_LIBRARY="/home/dom/workspace_catkin/devel/lib/libtxt_io_library.so" ..
make



--------------------------------------------
output filepath:

bin

