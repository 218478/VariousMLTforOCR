find_program(PYTEST pytest REQUIRED)

set(OUTPUT_DIR ${CMAKE_BINARY_DIR}/testing)
set(SRC_FILES ${CMAKE_SOURCE_DIR}/src/*.py)
set(TEST_FILES ${CMAKE_SOURCE_DIR}/testing/*.py)
set(IMAGES ${CMAKE_SOURCE_DIR}/testing/example_images/*)

add_custom_target(copy_files
                  COMMAND cp ${SRC_FILES} ${TEST_FILES} ${IMAGES} ${OUTPUT_DIR} -R)

add_custom_target(testing ${PYTEST} ${CMAKE_BINARY_DIR}/testing DEPENDS copy_files)