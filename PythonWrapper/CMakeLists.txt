cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

file (GLOB SOURCE_FILES "*.cpp")

find_package(pybind11 REQUIRED)
pybind11_add_module(PySpectralysis 
	${SOURCE_FILES}
)

target_link_libraries(PySpectralysis PUBLIC Engine)

install(TARGETS PySpectralysis DESTINATION ${CMAKE_BINARY_DIR}/outputs)
install(FILES ${CMAKE_CURRENT_LIST_DIR}/uiapp.py DESTINATION ${CMAKE_BINARY_DIR}/outputs)
install(FILES ${CMAKE_CURRENT_LIST_DIR}/icon_sm.png DESTINATION ${CMAKE_BINARY_DIR}/outputs)
