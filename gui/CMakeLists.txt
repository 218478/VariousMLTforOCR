# TODO: check it on cinnamon
if("$ENV{XDG_CURRENT_DESKTOP}" STREQUAL "GNOME" OR "$ENV{XDG_CURRENT_DESKTOP}" STREQUAL "X-Cinnamon")
message(WARNING "Experienced problems with Qt used on Gtk environment. Consider\
                 switching desktop environments in case of encountering critical Gtk errors")
endif()

find_program(PYUIC pyuic5 REQUIRED)

set(GUI_DESIGN ${CMAKE_SOURCE_DIR}/gui/MainWindow.ui)
set(GUI_PYTHON ${SOURCE_DIR}/design.py)

add_custom_target(ui_conversion ${PYUIC} ${GUI_DESIGN} > ${GUI_PYTHON})

add_custom_target(app ${PYTHON_EXEC} ${SOURCE_DIR}/MyGUI.py
                  ${NN_MODELS_DIR}
                  DEPENDS ui_conversion)