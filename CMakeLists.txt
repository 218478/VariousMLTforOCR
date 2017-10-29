cmake_minimum_required(VERSION 3.5)
project("Kuczaj Kamil - Praca inzynierska")

find_package(OpenCV REQUIRED) # OpenCV 3
include(GNUInstallDirs)
find_package(PythonInterp 3.5 REQUIRED)
find_package(PythonLibs 3.5 REQUIRED)
find_program(PLANTUML plantuml REQUIRED)
find_program(PYTHON_INTERPRETER3 python3 REQUIRED)
message(STATUS "Set the Python3 bin to: " ${PYTHON_INTERPRETER3})

set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(DATASETS_DIR ${CMAKE_SOURCE_DIR}/datasets)
set(NN_MODELS_DIR ${CMAKE_SOURCE_DIR}/app_data/nn_models)
set(SEP_CHARS_DIR ${CMAKE_SOURCE_DIR}/app_data/separated_chars)
set(SEP_WORDS_DIR ${CMAKE_SOURCE_DIR}/app_data/separated_words)

add_custom_target(mnist_ocr ${PYTHON_INTERPRETER3} ${SOURCE_DIR}/readMNIST.py
                  ${DATASETS_DIR}/MNIST
                  ${CMAKE_BINARY_DIR})

# reading Chars74K database
add_custom_target(chars74k_ocr ${PYTHON_INTERPRETER3} ${SOURCE_DIR}/readChars74K.py
                  ${DATASETS_DIR}/Chars74K/English/Fnt
                  ${CMAKE_BINARY_DIR})

# separate and fragment text
add_custom_target(vision ${PYTHON_INTERPRETER3} ${SOURCE_DIR}/opencv.py
                  ${DATASETS_DIR}/models.png
                  ${DATASETS_DIR}/textFragment.png
                  ${CMAKE_BINARY_DIR})

# teaching NN
add_custom_target(main ${PYTHON_INTERPRETER3} ${SOURCE_DIR}/main.py
                  ${DATASETS_DIR}/Chars74K/English/Fnt
                  ${CMAKE_BINARY_DIR})

# GUI
add_custom_target(app ${PYTHON_INTERPRETER3} ${SOURCE_DIR}/runGUI.py
                  ${NN_MODELS_DIR}
                  ${SEP_CHARS_DIR}
                  ${SEP_WORDS_DIR}
                  ${CMAKE_BINARY_DIR})

# UMLs for thesis
add_custom_target(umls echo "-- Umls generated")
file(GLOB UMLS ${CMAKE_SOURCE_DIR}/uml/*.plantuml)
foreach(UML ${UMLS})
    add_custom_command(TARGET umls
                       PRE_BUILD
                       COMMAND ${PLANTUML} ARGS ${UML} -output ${CMAKE_BINARY_DIR}/umls)
endforeach(UML)