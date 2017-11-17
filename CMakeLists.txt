cmake_minimum_required(VERSION 3.5)
project(VariousMLTforOCR)

find_program(PYTHON_EXEC python3 REQUIRED)
find_package(PythonInterp 3.5 REQUIRED)
find_package(PythonLibs 3.5 REQUIRED)
find_package(OpenCV 3.3 REQUIRED) # TODO: possibly migrate from opencv to PILLOW for easier mainteneance
include(GNUInstallDirs)

set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(DATASETS_DIR ${CMAKE_SOURCE_DIR}/datasets)
set(EXAMPLE_IMAGES_DIR ${CMAKE_SOURCE_DIR}/testing/example_images)
set(NN_MODELS_DIR ${CMAKE_SOURCE_DIR}/app_data/nn_models)
set(UTILS_DIR ${CMAKE_SOURCE_DIR}/utils)

add_subdirectory(gui)
add_subdirectory(src)
add_subdirectory(uml)
add_subdirectory(testing)
add_subdirectory(utils)
