cmake_minimum_required(VERSION 3.5)
project("Kuczaj Kamil - Praca inzynierska")

find_package(OpenCV REQUIRED) # OpenCV 3
include(GNUInstallDirs)
find_package(PythonInterp 3.5 REQUIRED)
find_package(PythonLibs 3.5 REQUIRED)
# TODO: automatic proper python3 finder
set(PYTHON_INTERPRETER3 python3)
set(PYTHON_INTERPRETER2 python2)
message(STATUS "Set the python bin to: " ${PYTHON_INTERPRETER})

set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(DATASETS_DIR ${CMAKE_SOURCE_DIR}/datasets)
set(NN_MODELS_DIR ${CMAKE_SOURCE_DIR}/app_data/nn_models)
set(SEP_CHARS_DIR ${CMAKE_SOURCE_DIR}/app_data/separated_chars)
set(SEP_WORDS_DIR ${CMAKE_SOURCE_DIR}/app_data/separated_words)

add_custom_target(mnist_ocr ${PYTHON_INTERPRETER3} ${SOURCE_DIR}/readMNIST.py
                  ${DATASETS_DIR}/MNIST
                  ${CMAKE_BINARY_DIR})

add_custom_target(chars74k_ocr ${PYTHON_INTERPRETER3} ${SOURCE_DIR}/readChars74K.py
                  ${DATASETS_DIR}/Chars74K/English/Fnt
                  ${CMAKE_BINARY_DIR})

# TODO: switch opencv to python3
add_custom_target(vision ${PYTHON_INTERPRETER3} ${SOURCE_DIR}/opencv.py
                  ${DATASETS_DIR}/models.png
                  ${DATASETS_DIR}/textFragment.png
                  ${CMAKE_BINARY_DIR})

add_custom_target(main ${PYTHON_INTERPRETER3} ${SOURCE_DIR}/main.py
                  ${DATASETS_DIR}/Chars74K/English/Fnt
                  ${CMAKE_BINARY_DIR})

add_custom_target(app ${PYTHON_INTERPRETER3} ${SOURCE_DIR}/runGUI.py
                  ${NN_MODELS_DIR}
                  ${SEP_CHARS_DIR}
                  ${SEP_WORDS_DIR}
                  ${CMAKE_BINARY_DIR})