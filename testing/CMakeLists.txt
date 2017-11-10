find_python_module(pytest REQUIRED)
find_program(PYTEST pytest REQUIRED)

set(OUTPUT_DIR ${CMAKE_BINARY_DIR}/testing)
set(SRC_FILES ${CMAKE_SOURCE_DIR}/src/*.py)
set(TEST_FILES ${CMAKE_SOURCE_DIR}/testing/*.py)

add_custom_target(copy_files
                  COMMAND cp ${SRC_FILES} ${TEST_FILES} ${OUTPUT_DIR} -R)

add_custom_target(testing ${PYTEST} . DEPENDS copy_files)