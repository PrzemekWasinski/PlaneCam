cmake_minimum_required(VERSION 3.16)
project(PlaneCam)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

add_compile_options(-O2 -Wall -Wextra)

find_package(Threads REQUIRED)
find_library(PIGPIO_LIB pigpio REQUIRED)
find_library(RT_LIB rt)

add_executable(tracker tracker.cpp)
target_link_libraries(tracker PRIVATE ${PIGPIO_LIB} Threads::Threads ${RT_LIB})

add_executable(camera_test tools/camera_test.cpp)
target_link_libraries(camera_test PRIVATE Threads::Threads)

add_executable(servo_test tools/servo_test.cpp)
target_link_libraries(servo_test PRIVATE ${PIGPIO_LIB} ${RT_LIB})

add_executable(tracker_mapping_test tools/tracker_mapping_test.cpp)
target_link_libraries(tracker_mapping_test PRIVATE ${PIGPIO_LIB} ${RT_LIB} Threads::Threads)
