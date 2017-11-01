find_program(PLANTUML plantuml REQUIRED)
add_custom_target(uml echo "-- Umls generated")
file(GLOB UMLS ${CMAKE_SOURCE_DIR}/uml/*.plantuml)
foreach(UML ${UMLS})
    add_custom_command(TARGET uml
                       PRE_BUILD
                       COMMAND ${PLANTUML} ARGS ${UML} -output ${CMAKE_BINARY_DIR}/uml)
endforeach(UML)