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
    Core Gui Widgets Multimedia MultimediaWidgets Network
)

# ── OpenCV ───────────────────────────────────────────────────────────────────
find_package(OpenCV REQUIRED)

# ── FFmpeg (optional) ────────────────────────────────────────────────────────
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(FFMPEG libavcodec libavformat libavutil libswscale)
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
)

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

target_link_libraries(${PROJECT_NAME} PRIVATE
    Qt6::Core
    Qt6::Gui
    Qt6::Widgets
    Qt6::Multimedia
    Qt6::MultimediaWidgets
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
