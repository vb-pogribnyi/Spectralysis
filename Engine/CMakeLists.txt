cmake_minimum_required(VERSION 3.8.2) # 3.8.0 is minimum version that allow Linux remote building and debugging


# Remove trailing '\' and '/' at the end of the path, so can reliably compare
STRING(REGEX REPLACE "[/\\]+$" "" CMAKE_SOURCE_DIR ${CMAKE_SOURCE_DIR})
STRING(REGEX REPLACE "[/\\]+$" "" CMAKE_CURRENT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

message(STATUS "Attempting auto locate Vulkan using CMake......")

# Find Vulkan Path using CMake's Vulkan Module
# This will return Boolean 'Vulkan_FOUND' indicating the status of find as success(ON) or fail(OFF).
# Include directory path - 'Vulkan_INCLUDE_DIRS' and 'Vulkan_LIBRARY' with required libraries.
find_package(Vulkan)

# Try extracting VulkanSDK path from ${Vulkan_INCLUDE_DIRS}
if (NOT ${Vulkan_INCLUDE_DIRS} STREQUAL "")
	set(VULKAN_PATH ${Vulkan_INCLUDE_DIRS})
	STRING(REGEX REPLACE "/Include" "" VULKAN_PATH ${VULKAN_PATH})
endif()
	 
if(NOT Vulkan_FOUND)
	# CMake may fail to locate the libraries but could be able to 
	# provide some path in Vulkan SDK include directory variable
	# 'Vulkan_INCLUDE_DIRS', try to extract path from this.
	message(STATUS "Failed to locate Vulkan SDK, retrying again...")
	if(EXISTS "${VULKAN_PATH}")
		message(STATUS "Successfully located the Vulkan SDK: ${VULKAN_PATH}")
	else()
		message("Error: Unable to locate Vulkan SDK. Please turn off auto locate option by specifying 'AUTO_LOCATE_VULKAN' as 'OFF'")
		message("and specify manually path using 'VULKAN_SDK' and 'VULKAN_VERSION' variables in the CMakeLists.txt.")
		return()
	endif()
endif()

# Notification when too low version of C++ is set
if(NOT DEFINED CMAKE_CXX_STANDARD OR CMAKE_CXX_STANDARD LESS 20)
	message(FATAL_ERROR "ERROR: CMAKE_CXX_STANDARD is not set or it is too low. Minimum C++ v20 is required to compile \"Engine\" library.")
endif()

   
set(MODULE_FILES
	src/VulkanCommon.cpp
	src/SDFTFilter.cpp
	engine_wrapper.cpp
	engine_wrapper.h
	dlib_export.h
)

add_library(Engine SHARED ${MODULE_FILES})

target_include_directories(Engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(Engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/inc)
target_include_directories(Engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/lib/glm)

if(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
	# Include Vulkan header files from Vulkan SDK
	include_directories(AFTER ${VULKAN_PATH}/Include)

	# Link directory for vulkan-1
	link_directories(${VULKAN_PATH}/Bin;${VULKAN_PATH}/Lib;)
endif()
target_link_libraries( Engine "${VULKAN_PATH}/Lib/vulkan-1.lib" )

# List files that install and/or package should provide.
# Each library should know what it want to distribute,
# which files are internal or intermediate and which are public library export.
install(TARGETS Engine DESTINATION ${CMAKE_BINARY_DIR}/outputs)
INSTALL(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/Shaders/ DESTINATION ${CMAKE_BINARY_DIR}/outputs/Shaders
          FILES_MATCHING PATTERN "*.spv")

