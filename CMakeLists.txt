cmake_minimum_required(VERSION 3.20)
project(gsplat-raster-app VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(GSPLAT_USE_CUDA_TOOLCHAIN)
    include("${GSPLAT_SDK_ROOT}/cmake/cuda_compiler_cache.cmake")
endif()

option(GSPLAT_BUILD_RENDER_PLY "Build CLI render app" ON)
option(GSPLAT_BUILD_OSG_APP "Build OSG roaming app" ON)
set(GSPLAT_SDK_ROOT "${CMAKE_SOURCE_DIR}/../gsplat-raster-sdk" CACHE PATH "Path to gsplat-raster-sdk project")

if(NOT EXISTS "${GSPLAT_SDK_ROOT}/CMakeLists.txt")
    message(FATAL_ERROR "GSPLAT_SDK_ROOT not found: ${GSPLAT_SDK_ROOT}")
endif()

add_subdirectory("${GSPLAT_SDK_ROOT}" "${CMAKE_BINARY_DIR}/gsplat-raster-sdk")

if(GSPLAT_BUILD_RENDER_PLY)
    add_executable(gsplat_render_ply
        ${CMAKE_SOURCE_DIR}/src/render_ply_main.cpp
        ${CMAKE_SOURCE_DIR}/src/camera_truth.cpp
        ${CMAKE_SOURCE_DIR}/src/ply_loader_local.cpp
    )
    target_link_libraries(gsplat_render_ply PRIVATE gsplat_raster)
    if(WIN32 AND CUDAToolkit_BIN_DIR)
        file(GLOB _cudart_dll "${CUDAToolkit_BIN_DIR}/cudart64_*.dll")
        if(_cudart_dll)
            add_custom_command(TARGET gsplat_render_ply POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${_cudart_dll}" "$<TARGET_FILE_DIR:gsplat_render_ply>"
            )
        endif()
    endif()
endif()

if(GSPLAT_BUILD_OSG_APP)
    set(_gsplat_osg_found FALSE)
    set(_gsplat_osg_libs)
    set(_gsplat_osg_includes)

    find_package(unofficial-osg CONFIG QUIET)
    if(unofficial-osg_FOUND)
        set(_gsplat_osg_found TRUE)
        set(_gsplat_osg_libs
            unofficial::osg::osg
            unofficial::osg::osgDB
            unofficial::osg::osgGA
            unofficial::osg::osgUtil
            unofficial::osg::osgViewer
            unofficial::osg::OpenThreads
        )
        if(DEFINED VCPKG_INSTALLED_DIR AND DEFINED VCPKG_TARGET_TRIPLET)
            set(_osg_include "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/include")
            if(EXISTS "${_osg_include}/osg/Version")
                list(APPEND _gsplat_osg_includes "${_osg_include}")
            endif()
        endif()
    else()
        find_package(OpenSceneGraph QUIET COMPONENTS osg osgDB osgGA osgUtil osgViewer)
        if(OpenSceneGraph_FOUND)
            set(_gsplat_osg_found TRUE)
            set(_gsplat_osg_libs ${OPENSCENEGRAPH_LIBRARIES})
            set(_gsplat_osg_includes ${OPENSCENEGRAPH_INCLUDE_DIRS})
        endif()
    endif()

    if(_gsplat_osg_found)
        add_executable(gsplat_osg_app
            ${CMAKE_SOURCE_DIR}/src/osg_viewer_main.cpp
            ${CMAKE_SOURCE_DIR}/src/camera_truth.cpp
            ${CMAKE_SOURCE_DIR}/src/ply_loader_local.cpp
        )
        if(_gsplat_osg_includes)
            target_include_directories(gsplat_osg_app PRIVATE ${_gsplat_osg_includes})
        endif()
        target_link_libraries(gsplat_osg_app PRIVATE gsplat_raster ${_gsplat_osg_libs})
        if(WIN32 AND CUDAToolkit_BIN_DIR)
            file(GLOB _cudart_dll2 "${CUDAToolkit_BIN_DIR}/cudart64_*.dll")
            if(_cudart_dll2)
                add_custom_command(TARGET gsplat_osg_app POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${_cudart_dll2}" "$<TARGET_FILE_DIR:gsplat_osg_app>"
                )
            endif()
        endif()
    else()
        message(WARNING "OpenSceneGraph not found (unofficial-osg/OpenSceneGraph), skip gsplat_osg_app")
    endif()
endif()
