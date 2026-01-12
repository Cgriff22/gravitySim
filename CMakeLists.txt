cmake_minimum_required(VERSION 3.16)
project(grav LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(grav
    main.cpp
    external/glad/src/glad.c
)

target_include_directories(grav PRIVATE
    external/glad/include
)

find_package(OpenGL REQUIRED)
find_package(glfw3 3.3 REQUIRED)

target_link_libraries(grav PRIVATE OpenGL::GL glfw)
