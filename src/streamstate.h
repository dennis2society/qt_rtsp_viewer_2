#pragma once

#include <QString>

/// Playback state for a single stream.
enum class PlaybackState {
    Stopped,
    Playing
};

/// All per-stream state.  Accessed from the UI thread (writes) and
/// from each stream's worker thread (reads).  Access is synchronised
/// through StreamStateManager's read/write-lock helpers.
struct StreamState {
    // -- identity ----------------------------------------------------
    int streamId = -1;
    QString rtspUrl;
    QString cameraName;

    // -- playback ----------------------------------------------------
    PlaybackState playbackState = PlaybackState::Stopped;

    // -- recording --------------------------------------------------
    bool isRecording = false;
    bool isAutoRecording = false;
    QString recordCodec = QStringLiteral("libx264"); // libx264 | libx265
    QString recordFormat = QStringLiteral("mp4"); // mp4 | mkv | avi
    double recordFps = 25.0;

    // -- effects -----------------------------------------------------
    int blurAmount = 0;
    bool grayscaleEnabled = false;
    int brightnessAmount = 0; // -100 ... 100
    int contrastAmount = 0; // -100 ... 100
    int colorTemperature = 0; // -100 ... 100

    bool motionDetectionEnabled = false;
    int motionSensitivity = 20; // 1 ... 100

    bool motionVectorsEnabled = false;
    int motionVectorsSensitivity = 50; // 1 ... 100
    bool motionTracesEnabled = false;
    int motionTraceDecay = 50; // 1 ... 100  (maps to 0.80 ... 0.99)

    bool motionGraphEnabled = false;
    int motionGraphSensitivity = 50; // 1 ... 100

    bool faceDetectionEnabled = false;

    bool overlayEnabled = true; // FPS / resolution / datetime

    // -- CSV motion logging -----------------------------------------
    bool motionCsvEnabled = false; // log motion data to CSV during recording
    bool recordCleanVideo = false; // record pre-overlay frame (overlays still visible on screen)

    // -- auto-record on motion --------------------------------------
    bool autoRecordEnabled = false;
    double autoRecordThreshold = 0.50; // 0.0 ... 1.0
    int autoRecordTimeout = 5; // seconds  (1 ... 120)
};
