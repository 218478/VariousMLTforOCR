cmake_minimum_required(VERSION 3.0)
project("Kuczaj Kamil - Praca inzynierska")

# TODO: add library checking
find_package(OpenCV REQUIRED) # OpenCV 3
find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)

set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(DATASETS_DIR ${CMAKE_SOURCE_DIR}/datasets)

add_custom_target(Reader python2.7 ${SOURCE_DIR}/read.py ${DATASETS_DIR}/MNIST)
