#pragma once

#include <QDateTime>
#include <QImage>
#include <QMutex>
#include <QObject>
#include <QVideoFrame>
#include <atomic>

class OpenCVProcessor;
class QTimer;

/// Lives on a dedicated QThread.  Receives raw QVideoFrames, applies the
/// full effects pipeline, and emits composited QImages.  Recording is handled
/// by a separate RecordingWorker on its own thread — this class only forwards
/// frames via the frameForRecording signal.
///
/// Frame delivery uses a "latest frame" pattern: incoming frames are stored
/// atomically and only the most recent frame is processed, preventing queue
/// build-up and display latency.
class VideoWorker : public QObject
{
    Q_OBJECT

public:
    explicit VideoWorker(int streamId, QObject *parent = nullptr);
    ~VideoWorker() override;

public slots:
    /// Called from the multimedia thread (DirectConnection).  Stores the
    /// frame and returns immediately — no heavy processing here.
    void submitFrame(const QVideoFrame &frame);

    void setPaused(bool p);
    void setStreamActive(bool active);
    void resetStream(); // clears inter-frame state when URL changes

    // ── recording state (kept for auto-record logic) ────────────────
    void setRecording(bool on);

signals:
    void frameReady(const QImage &image);

    /// Emitted for every frame while recording is active.  Connected to
    /// RecordingWorker::enqueueFrame across threads (queued connection).
    void frameForRecording(const QImage &image);

    /// Auto-record requests — connected to RecordingWorker via VideoPlayer
    void startRecordingRequested(const QString &path, const QString &codec, double fps);
    void stopRecordingRequested();

    void autoRecordingStarted(const QString &path);
    void autoRecordingStopped(const QString &path);

private slots:
    /// Driven by QTimer(0) on the worker thread.  Grabs the latest
    /// submitted frame and runs the full effects pipeline.
    void processPendingFrame();

private:
    void processFrame(const QVideoFrame &frame);
    void paintFpsOverlay(QImage &img);
    void handleAutoRecord(double motionLevel);

    int m_streamId;
    OpenCVProcessor *m_processor = nullptr;

    bool m_paused = false;
    bool m_streamActive = false;

    // FPS counter
    int m_frameCount = 0;
    double m_fps = 0.0;
    QDateTime m_fpsTimer;

    // Clean-frame bookkeeping (for detection algorithms)
    QImage m_cleanPrevious;
    QImage m_frozenFrame;

    // ── Latest-frame storage (written by multimedia thread) ─────────
    QMutex m_frameMutex;
    QVideoFrame m_pendingFrame;
    std::atomic<bool> m_hasNewFrame{false};

    // ── Recording awareness (no FFmpeg) ─────────────────────────────
    bool m_recording = false;
    QString m_recPath;
    QString m_recCodec;
    double m_recFps = 25.0;

    // ── Auto-record ─────────────────────────────────────────────────
    bool m_autoRecording = false;
    QDateTime m_autoRecStartTime;
    qint64 m_lastMotionAboveMs = 0;
};
