cmake_minimum_required(VERSION 3.0)
project("Kuczaj Kamil - Praca inzynierska")

# TODO: add library checking
find_package(OpenCV REQUIRED) # OpenCV 3
find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)

set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(DATASETS_DIR ${CMAKE_SOURCE_DIR}/datasets)

add_custom_target(MNIST_OCR python2.7 ${SOURCE_DIR}/readMNIST.py
                  ${DATASETS_DIR}/MNIST
                  ${CMAKE_BINARY_DIR})

add_custom_target(CHARS74K_OCR python2.7 ${SOURCE_DIR}/readChars74K.py
                  ${DATASETS_DIR}/Chars74K
                  ${CMAKE_BINARY_DIR})

add_custom_target(APP python2.7 ${SOURCE_DIR}/qtApp.py
                  ${DATASETS_DIR}/models.png
                  ${CMAKE_BINARY_DIR})
