find_python_module(imgaug REQUIRED)
find_python_module(PIL REQUIRED)
set(UTILS_DIR ${CMAKE_SOURCE_DIR}/utils)
set(OUTPUT_UTILS ${CMAKE_BINARY_DIR}/dataset)

add_custom_target(dataset ${PYTHON_EXEC} ${UTILS_DIR}/create_my_dataset.py
                  ${OUTPUT_UTILS})