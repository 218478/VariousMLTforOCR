set(UTILS_DIR ${CMAKE_SOURCE_DIR}/utils)
set(OUTPUT_UTILS ${CMAKE_BINARY_DIR}/dataset)

add_custom_target(dataset ${PYTHON_EXEC} ${UTILS_DIR}/create_my_dataset.py)

add_custom_target(generate_image ${PYTHON_EXEC} ${UTILS_DIR}/create_test_image.py)