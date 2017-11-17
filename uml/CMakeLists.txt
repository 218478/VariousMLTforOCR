find_program(PLANTUML plantuml REQUIRED)
add_custom_target(uml echo "-- Umls generated")
file(GLOB UMLS ${CMAKE_SOURCE_DIR}/uml/*.plantuml)
foreach(UML ${UMLS})
    add_custom_command(TARGET uml
                       PRE_BUILD
                       COMMAND ${PLANTUML} -charset UTF-8 ARGS ${UML} -output ${CMAKE_BINARY_DIR}/uml)
    string(REPLACE .plantuml .png UML_FILENAME ${UML})
    string(REGEX REPLACE ^.*\\/\(.*.png\) \\1 UML_FILENAME ${UML_FILENAME})
    add_custom_command(TARGET uml
                       POST_BUILD
                       COMMAND eog ${UML_FILENAME})
endforeach(UML)

add_custom_target(python_class_diagram pyreverse3 ${CMAKE_SOURCE_DIR}/src/*)
add_custom_command(TARGET python_class_diagram
                   POST_BUILD
                   COMMAND dot -Tps classes_No_Name.dot -o outfile.ps)