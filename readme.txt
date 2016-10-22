
MMMMMMMMMMMMMMMMMMMMM.           .    .           .MMMMMMMMMMMMMMMMMMMMM
 `MMMMMMMMMMMMMMMMMMMM           M\  /M           MMMMMMMMMMMMMMMMMMMM'
   `MMMMMMMMMMMMMMMMMMM          MMMMMM          MMMMMMMMMMMMMMMMMMM'  
     MMMMMMMMMMMMMMMMMMM-_______MMMMMMMM_______-MMMMMMMMMMMMMMMMMMM    
      MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM    
      MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM    
      MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM    
     .MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM.    
    MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM  
                   `MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM'                
                          `MMMMMMMMMMMMMMMMMM'                    
                              `MMMMMMMMMM'                              
                                 MMMMMM             
                                  MMMM                                  
                                   MM 



------------------------------------------------------------------------
project: svi_mapper
version: 0.72
contact: https://github.com/schdomin



------------------------------------------------------------------------
dependencies:

      OpenCV: https://github.com/opencv/opencv (trunk)
         g2o: https://github.com/RainerKuemmerle/g2o (trunk)
thin_drivers: https://github.com/grisetti/thin_drivers (visensor/txt_io library)
         ROS: http://www.ros.org (dataset acquisition)



------------------------------------------------------------------------
build sequence EXAMPLE (shell in project root):

mkdir build
cd build
cmake -D OpenCV_DIR='/home/dom/libraries/cpp/opencv/release' \
-D G2O_ROOT='/home/dom/libraries/cpp/g2o' \
-D TXT_IO_INCLUDE_DIR='/home/dom/workspace_catkin/src/thin_drivers/thin_state_publisher/src' \
-D TXT_IO_LIBRARY='/home/dom/workspace_catkin/devel/lib/libthin_txt_io_library.so' \
-D DLIB_INCLUDE_DIR='/home/dom/libraries/cpp/DLib/include' \
-D DBOW2_INCLUDE_DIR='/home/dom/libraries/cpp/DBoW2/include/DBoW2' \
-D DLIB_LIBRARY='/home/dom/libraries/cpp/DLib/build/libDLib.so' \
-D DBOW2_LIBRARY='/home/dom/libraries/cpp/DBoW2/build/libDBoW2.so' \
-D CMAKE_BUILD_TYPE=Release ..
make -j666



------------------------------------------------------------------------
hardware requirements:

threads: 3
RAM: 4-32GB (depending on desired map scale)

