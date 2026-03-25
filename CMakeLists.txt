cmake_minimum_required(VERSION 3.21)
project(QtRtspViewer VERSION 2.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ── Qt6 ──────────────────────────────────────────────────────────────────────
find_package(Qt6 REQUIRED COMPONENTS
    Core Gui Widgets Multimedia Network
)

# ── OpenCV ───────────────────────────────────────────────────────────────────
find_package(OpenCV REQUIRED)

# ── FFmpeg (optional) ────────────────────────────────────────────────────────
# On Linux, pkg-config finds FFmpeg automatically.
# On Windows, pass -DFFMPEG_DIR=<path-to-ffmpeg> where the path contains
# include/ and lib/ subdirectories (e.g. a vcpkg or MSYS2 install, or an
# official FFmpeg shared/dev download).
set(FFMPEG_DIR "" CACHE PATH "Root directory of an FFmpeg installation (contains include/ and lib/)")

find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(FFMPEG libavcodec libavformat libavutil libswscale)
endif()

# Fallback: manual discovery via FFMPEG_DIR (useful on Windows)
if(NOT FFMPEG_FOUND AND FFMPEG_DIR)
    set(_ffmpeg_inc "${FFMPEG_DIR}/include")
    set(_ffmpeg_lib "${FFMPEG_DIR}/lib")

    if(EXISTS "${_ffmpeg_inc}/libavcodec/avcodec.h" AND EXISTS "${_ffmpeg_lib}")
        set(FFMPEG_FOUND TRUE)
        set(FFMPEG_INCLUDE_DIRS "${_ffmpeg_inc}")
        set(FFMPEG_LIBRARY_DIRS "${_ffmpeg_lib}")

        # Locate each required library
        set(FFMPEG_LIBRARIES "")
        foreach(_lib avcodec avformat avutil swscale)
            find_library(_ffmpeg_${_lib}
                NAMES ${_lib}
                PATHS "${_ffmpeg_lib}"
                NO_DEFAULT_PATH
            )
            if(_ffmpeg_${_lib})
                list(APPEND FFMPEG_LIBRARIES "${_ffmpeg_${_lib}}")
            else()
                message(WARNING "FFmpeg library '${_lib}' not found in ${_ffmpeg_lib}")
                set(FFMPEG_FOUND FALSE)
                break()
            endif()
        endforeach()
    else()
        message(WARNING "FFMPEG_DIR is set but ${_ffmpeg_inc}/libavcodec/avcodec.h or ${_ffmpeg_lib} not found")
    endif()
endif()

# ── Sources ──────────────────────────────────────────────────────────────────
set(SOURCES
    src/main.cpp
    src/streamstatemanager.cpp
    src/mainwindow.cpp
    src/streamtab.cpp
    src/videoplayer.cpp
    src/videoworker.cpp
    src/recordingworker.cpp
    src/effectssidebar.cpp
    src/opencvprocessor.cpp
    src/recorddialog.cpp
    src/onvifclient.cpp
    src/onvifsettingsdialog.cpp
    res/resources.qrc
)

set(HEADERS
    src/streamstate.h
    src/streamstatemanager.h
    src/mainwindow.h
    src/streamtab.h
    src/videoplayer.h
    src/videoworker.h
    src/recordingworker.h
    src/effectssidebar.h
    src/opencvprocessor.h
    src/recorddialog.h
    src/onvifclient.h
    src/onvifsettingsdialog.h
)

add_executable(${PROJECT_NAME} WIN32 ${SOURCES} ${HEADERS})

target_link_libraries(${PROJECT_NAME} PRIVATE
    Qt6::Core
    Qt6::Gui
    Qt6::Widgets
    Qt6::Multimedia
    Qt6::Network
    ${OpenCV_LIBS}
)

if(FFMPEG_FOUND)
    message(STATUS "FFmpeg found - recording support enabled")
    target_compile_definitions(${PROJECT_NAME} PRIVATE HAVE_FFMPEG)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${FFMPEG_LIBRARIES})
    target_include_directories(${PROJECT_NAME} PRIVATE ${FFMPEG_INCLUDE_DIRS})
    target_link_directories(${PROJECT_NAME} PRIVATE ${FFMPEG_LIBRARY_DIRS})
else()
    message(WARNING "FFmpeg NOT found - recording support disabled")
endif()

# ── Copy OpenCV data to build dir ────────────────────────────────────────────
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/opencv
        $<TARGET_FILE_DIR:${PROJECT_NAME}>/opencv
    COMMENT "Copying OpenCV data files to output directory"
)
