message("[ initialization ]-------------------------------------------------------------")
message(" ")

cmake_minimum_required(VERSION 2.8.3)
project(svi_mapper)

#ds modules
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/modules" ${CMAKE_MODULE_PATH})
message("CMake module path: ${CMAKE_MODULE_PATH}")

#ds default build type DEBUG
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE DEBUG)
endif(NOT CMAKE_BUILD_TYPE)

message("build type: ${CMAKE_BUILD_TYPE}")

message(" ")
message("[ modules search ]-------------------------------------------------------------")
message(" ")

#ds find packages
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(G2O REQUIRED)
find_package(Qt4 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(QGLViewer REQUIRED)
find_package(Cholmod REQUIRED)

message(" ")
message("[ modules versions ]-----------------------------------------------------------")
message(" ")

#ds log package information
#message("Eigen: ${EIGENVERSION}")
message("OpenCV: ${OpenCV_VERSION}")
message("Qt4: ${QTVERSION}")
#message("OpenGL: ${OPENGLVERSION}")
#message("GLUT: ${OpenCV_VERSION}")
#message("QGLViewer: ${QGLViewerVERSION}")
#message("Cholmod: ${CHOLMOD_VERSION}")

message(" ")
message("[ compilation ]----------------------------------------------------------------")
message(" ")

#ds output paths
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")
message("binaries: ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
message(" ")



#ds add c++11 support, no optimization, all warnings
add_definitions(-O3 -march=native -Wall -pedantic -std=c++11 -pthread)
#add_definitions(-O3 -march=native -Wall -pedantic -std=c++11 -pthread -DNDEBUG)



#ds include dirs
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${G2O_INCLUDE_DIR})
include_directories(${CHOLMOD_INCLUDE_DIR})
include_directories(${QT_INCLUDES})
include_directories(${OPENGL_INCLUDE_DIRS})
include_directories(${GLUT_INCLUDE_DIRS})
include_directories(${QGLVIEWER_INCLUDE_DIR})
include_directories(${FPS_MAPPER_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)



#ds source files
set(SOURCEFILES
src/types/CLandmark.cpp
src/types/CKeyFrame.cpp
src/utility/CLogger.cpp
src/utility/CIMUInterpolator.cpp
src/vision/CMiniVisionToolbox.cpp
src/core/CTriangulator.cpp
src/core/CFundamentalMatcher.cpp
src/core/CMapper.cpp
src/optimization/Cg2oOptimizer.cpp
src/optimization/closure_buffer.cpp
src/optimization/closure_checker.cpp
src/gui/CViewerScene.cpp
src/optimization/CSolverStereoPosit.cpp)



#ds libraries
set(LIBRARIES_GENERAL
${OpenCV_LIBRARIES}
${G2O_CORE_LIBRARY}
${G2O_TYPES_SLAM3D}
${G2O_SOLVER_CHOLMOD}
${CHOLMOD_LIBRARY}
${G2O_STUFF_LIBRARY}
${FPS_MAPPER_TXT_IO_LIBRARY}
-fopenmp
-pthread)

set(LIBRARIES_VISUALIZATION
${OPENGL_LIBRARIES}
${GLUT_LIBRARY}
${QGLVIEWER_LIBRARY}
${QT_QTXML_LIBRARY}
${QT_QTOPENGL_LIBRARY}
${QT_QTGUI_LIBRARY}
${QT_QTCORE_LIBRARY}
GLU)



#ds core binaries
add_executable(stereo_fps src/runnable/stereo_fps.cpp src/core/CTrackerStereo.cpp ${SOURCEFILES})
target_link_libraries(stereo_fps ${LIBRARIES_GENERAL} ${LIBRARIES_VISUALIZATION})

#ds io
add_executable(test_cloud_matching src/runnable/test_cloud_matching.cpp src/vision/CMiniVisionToolbox.cpp src/gui/CViewerCloud.cpp src/types/CKeyFrame.cpp)
target_link_libraries(test_cloud_matching ${OpenCV_LIBRARIES} ${LIBRARIES_VISUALIZATION})
add_executable(validate_dataset src/runnable/validate_dataset.cpp)
target_link_libraries(validate_dataset ${FPS_MAPPER_TXT_IO_LIBRARY})

#ds utilities
add_executable(interpolate_trajectory src/runnable/interpolate_trajectory.cpp src/utility/CLogger.cpp)
add_executable(compute_rotation_icp src/runnable/compute_rotation_icp.cpp src/utility/CLogger.cpp)
add_executable(triangulation_sampling src/runnable/triangulation_sampling.cpp src/vision/CMiniVisionToolbox.cpp)
target_link_libraries(triangulation_sampling ${OpenCV_LIBRARIES})

#ds list binaries UGLY GULY
message("stereo_fps")
message("test_cloud_matching")
message("validate_dataset")
message("interpolate_trajectory")
message("compute_rotation_icp")
message("triangulation_sampling")
message(" ")
message("-------------------------------------------------------------------------------")

