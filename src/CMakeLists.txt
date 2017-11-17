# reading Chars74K database
add_custom_target(dataset_reader ${PYTHON_EXEC} ${SOURCE_DIR}/DatasetReader.py
            ${DATASETS_DIR}/dataset_64x64)

# separate and fragment text
add_custom_target(text_extractor ${PYTHON_EXEC} ${SOURCE_DIR}/TextExtractor.py
            ${EXAMPLE_IMAGES_DIR}/models.png
            ${EXAMPLE_IMAGES_DIR}/textFragment.png)
