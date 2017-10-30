cmake_minimum_required(VERSION 3.5)
project("Kuczaj Kamil - Praca inzynierska")

include(GNUInstallDirs)
include(find_python_module.cmake)
find_program(PLANTUML plantuml REQUIRED)
find_program(PYUIC pyuic5 REQUIRED)
find_program(PYTHON_EXEC python3 REQUIRED)
find_package(PythonInterp 3.5 REQUIRED)
find_package(PythonLibs 3.5 REQUIRED)
find_package(OpenCV REQUIRED) # OpenCV 3

#Python modules
find_python_module(PyQt4 REQUIRED)
find_python_module(tensorflow REQUIRED)
find_python_module(keras REQUIRED)
message(STATUS "Set the Python3 bin to: " ${PYTHON_EXEC})

set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(DATASETS_DIR ${CMAKE_SOURCE_DIR}/datasets)
set(NN_MODELS_DIR ${CMAKE_SOURCE_DIR}/app_data/nn_models)
set(GUI_DESIGN ${CMAKE_SOURCE_DIR}/gui/MainWindow.ui)
set(GUI_PYTHON ${SOURCE_DIR}/design.py)

add_custom_target(mnist_ocr ${PYTHON_EXEC} ${SOURCE_DIR}/readMNIST.py
                  ${DATASETS_DIR}/MNIST
                  ${CMAKE_BINARY_DIR})

# reading Chars74K database
add_custom_target(chars74k_ocr ${PYTHON_EXEC} ${SOURCE_DIR}/readChars74K.py
                  ${DATASETS_DIR}/Chars74K/English/Fnt
                  ${CMAKE_BINARY_DIR})

# separate and fragment text
add_custom_target(vision ${PYTHON_EXEC} ${SOURCE_DIR}/opencv.py
                  ${DATASETS_DIR}/models.png
                  ${DATASETS_DIR}/textFragment.png
                  ${CMAKE_BINARY_DIR})

# teaching NN
add_custom_target(main ${PYTHON_EXEC} ${SOURCE_DIR}/main.py
                  ${DATASETS_DIR}/Chars74K/English/Fnt
                  ${CMAKE_BINARY_DIR})

# GUI
add_custom_target(app ${PYTHON_EXEC} ${SOURCE_DIR}/runGUI.py
                  ${NN_MODELS_DIR}
                  ${CMAKE_BINARY_DIR})
# add_custom_command(TARGET app
#                    PRE_BUILD
#                    COMMAND ${PYUIC} "${GUI_DESIGN} > ${GUI_PYTHON}")

# UMLs for thesis
add_custom_target(umls echo "-- Umls generated")
file(GLOB UMLS ${CMAKE_SOURCE_DIR}/uml/*.plantuml)
foreach(UML ${UMLS})
    add_custom_command(TARGET umls
                       PRE_BUILD
                       COMMAND ${PLANTUML} ARGS ${UML} -output ${CMAKE_BINARY_DIR}/umls)
endforeach(UML)