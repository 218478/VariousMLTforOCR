cmake_minimum_required(VERSION 3.5)
project("Kuczaj Kamil - Praca inzynierska")

# TODO: add library checking (cause the location for opencv varies)
find_package(OpenCV REQUIRED) # OpenCV 3

# set(Python_ADDITIONAL_VERSIONS 3.5)
include(GNUInstallDirs)
find_package(PythonInterp 3.5 REQUIRED)
# find_package(PythonLibs 3.5 REQUIRED) # problematic
# TODO: automatic proper python3 finder
set(PYTHON_INTERPRETER3 python3)
set(PYTHON_INTERPRETER2 python2)
message(STATUS "Set the python bin to: " ${PYTHON_INTERPRETER})

set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(GUI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/gui)
set(DATASETS_DIR ${CMAKE_SOURCE_DIR}/datasets)

# TODO: change target names. They are  misleading

add_custom_target(MNIST_OCR ${PYTHON_INTERPRETER3} ${SOURCE_DIR}/readMNIST.py
                  ${DATASETS_DIR}/MNIST
                  ${CMAKE_BINARY_DIR})

add_custom_target(CHARS74K_OCR ${PYTHON_INTERPRETER3} ${SOURCE_DIR}/readChars74K.py
                  ${DATASETS_DIR}/Chars74K/English/Fnt
                  ${CMAKE_BINARY_DIR})

add_custom_target(VISION ${PYTHON_INTERPRETER2} ${SOURCE_DIR}/opencv.py
                  ${DATASETS_DIR}/models.png
                  ${CMAKE_BINARY_DIR})

add_custom_target(APP ${PYTHON_INTERPRETER3} ${GUI_DIR}/main.py
                  ${CMAKE_BINARY_DIR})

add_custom_target(MAIN ${PYTHON_INTERPRETER3} ${SOURCE_DIR}/main.py
                  ${DATASETS_DIR}/Chars74K/English/Fnt
                  ${CMAKE_BINARY_DIR})