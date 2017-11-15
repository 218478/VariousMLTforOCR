# TODO: obsolete need to upgrade
add_custom_target(mnist_ocr ${PYTHON_EXEC} ${SOURCE_DIR}/readMNIST.py
            ${DATASETS_DIR}/MNIST
            ${CMAKE_BINARY_DIR})

# reading Chars74K database
add_custom_target(chars74k_ocr ${PYTHON_EXEC} ${SOURCE_DIR}/readChars74K.py
            ${DATASETS_DIR}/dataset_64x64
            ${CMAKE_BINARY_DIR})

# separate and fragment text
add_custom_target(textExtractor ${PYTHON_EXEC} ${SOURCE_DIR}/textExtractor.py
            ${EXAMPLE_IMAGES_DIR}/models.png
            ${EXAMPLE_IMAGES_DIR}/textFragment.png
            ${CMAKE_BINARY_DIR})

# teaching NN
add_custom_target(main_teach_nn ${PYTHON_EXEC} ${SOURCE_DIR}/main.py
        ${DATASETS_DIR}/dataset
        ${CMAKE_BINARY_DIR})
