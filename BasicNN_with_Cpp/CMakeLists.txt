cmake_minimum_required(VERSION 3.16)

project(BasicNN_with_Cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(BasicNN_with_Cpp main.cpp)

include(GNUInstallDirs)
install(TARGETS BasicNN_with_Cpp
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
