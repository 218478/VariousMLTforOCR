cmake_minimum_required(VERSION 3.5)
project("Kuczaj Kamil - Praca inzynierska")

find_program(PYTHON_EXEC python3 REQUIRED)
find_package(PythonInterp 3.5 REQUIRED)
find_package(PythonLibs 3.5 REQUIRED)
find_package(OpenCV 3.3 REQUIRED)
include(GNUInstallDirs)
include(find_python_module.cmake)

# Python modules
# TODO: convert it to requirements.txt list
find_python_module(PyQt5 REQUIRED)
find_python_module(tensorflow REQUIRED)
find_python_module(keras REQUIRED)
find_python_module(cv2 REQUIRED)
find_python_module(numpy REQUIRED)

set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(DATASETS_DIR ${CMAKE_SOURCE_DIR}/datasets)
set(EXAMPLE_IMAGES_DIR ${CMAKE_SOURCE_DIR}/testing/example_images)
set(NN_MODELS_DIR ${CMAKE_SOURCE_DIR}/app_data/nn_models)
set(UTILS_DIR ${CMAKE_SOURCE_DIR}/utils)

# TODO: obsolete need to upgrade
add_custom_target(mnist_ocr ${PYTHON_EXEC} ${SOURCE_DIR}/readMNIST.py
                  ${DATASETS_DIR}/MNIST
                  ${CMAKE_BINARY_DIR})

# reading Chars74K database
add_custom_target(chars74k_ocr ${PYTHON_EXEC} ${SOURCE_DIR}/readChars74K.py
                  # ${DATASETS_DIR}/Chars74K/English/Fnt
                  ${DATASETS_DIR}/dataset5
                  ${CMAKE_BINARY_DIR})

# separate and fragment text
add_custom_target(textExtractor ${PYTHON_EXEC} ${SOURCE_DIR}/textExtractor.py
                  ${EXAMPLE_IMAGES_DIR}/models.png
                  ${EXAMPLE_IMAGES_DIR}/textFragment.png
                  ${CMAKE_BINARY_DIR})

# teaching NN
add_custom_target(main_teach_nn ${PYTHON_EXEC} ${SOURCE_DIR}/main.py
                #   ${DATASETS_DIR}/Chars74K/English/Fnt
                  ${DATASETS_DIR}/dataset
                  ${CMAKE_BINARY_DIR})

add_subdirectory(gui)
add_subdirectory(uml)
add_subdirectory(testing)
add_subdirectory(utils)