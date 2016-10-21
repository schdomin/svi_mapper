message("--------------------------------------------------------------------- initialization")

cmake_minimum_required(VERSION 2.8.3)
project(svi_mapper)

#ds modules
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/modules" ${CMAKE_MODULE_PATH})
message("CMake module path: ${CMAKE_MODULE_PATH}")

#ds default build type RELEASE
#if(NOT CMAKE_BUILD_TYPE)
#  set(CMAKE_BUILD_TYPE RELEASE)
#endif(NOT CMAKE_BUILD_TYPE)

set(CMAKE_BUILD_TYPE Debug)
message("build type: ${CMAKE_BUILD_TYPE}")

message("--------------------------------------------------------------------- modules")

#ds find packages
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(G2O REQUIRED)
find_package(Qt4 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(QGLViewer REQUIRED)
find_package(Cholmod REQUIRED)

#ds log package information
#message("Eigen: ${EIGENVERSION}")
message("OpenCV: ${OpenCV_VERSION}")
message("Qt4: ${QTVERSION}")
#message("OpenGL: ${OPENGLVERSION}")
#message("GLUT: ${OpenCV_VERSION}")
#message("QGLViewer: ${QGLViewerVERSION}")
#message("Cholmod: ${CHOLMOD_VERSION}")

message("--------------------------------------------------------------------- build")

#ds output paths
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "lib")
message("directory output binaries: ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
message("directory output libraries: ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")



#ds add c++11 support, no optimization, all warnings
add_definitions(-O3 -march=native -Wall -pedantic -std=c++11)
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
include_directories(${TXT_IO_INCLUDE_DIR})
include_directories(${DLIB_INCLUDE_DIR})
include_directories(${DBOW2_INCLUDE_DIR})
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
src/optimization/Cg2oOptimizer.cpp
src/optimization/edge_se3_linear_acceleration.cpp
src/optimization/closure_buffer.cpp
src/optimization/closure_checker.cpp
#src/gui/CViewerScene.cpp
src/gui/CViewerBTree.cpp
src/optimization/CSolverStereoPosit.cpp)



#ds libraries
set(LIBRARIES_GENERAL
${OpenCV_LIBRARIES}
${G2O_CORE_LIBRARY}
${G2O_TYPES_SLAM3D}
${G2O_SOLVER_CHOLMOD}
${CHOLMOD_LIBRARY}
${G2O_STUFF_LIBRARY}
${TXT_IO_LIBRARY}
${DLIB_LIBRARY}
${DBOW2_LIBRARY}
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



#ds g2o edges
add_library(G2oEdges SHARED src/optimization/G2oEdges.cpp src/optimization/edge_se3_linear_acceleration.cpp)
#target_link_libraries(G2oEdges)

#ds core binaries
#add_executable(tracker_svi src/runnable/tracker_svi.cpp src/core/CTrackerSVI.cpp ${SOURCEFILES})
#target_link_libraries(tracker_svi ${LIBRARIES_GENERAL} ${LIBRARIES_VISUALIZATION})
#add_executable(tracker_sv src/runnable/tracker_sv.cpp src/core/CTrackerSV.cpp ${SOURCEFILES})
#target_link_libraries(tracker_sv ${LIBRARIES_GENERAL} ${LIBRARIES_VISUALIZATION})
add_executable(tracker_gt src/runnable/tracker_gt.cpp src/core/CTrackerGT.cpp ${SOURCEFILES})
target_link_libraries(tracker_gt ${LIBRARIES_GENERAL} ${LIBRARIES_VISUALIZATION})



message("--------------------------------------------------------------------- completion")


