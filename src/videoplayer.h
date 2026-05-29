#pragma once

#include <QImage>
#include <QMediaPlayer>
#include <QPointF>
#include <QSize>
#include <QWidget>

class QLabel;
class QVideoSink;
class QThread;
class QTimer;
class QResizeEvent;
class QAudioOutput;
class VideoWorker;
class RecordingWorker;
class RawStreamWorker;

/// Widget that wraps QMediaPlayer + QLabel display + worker thread.
/// One instance per stream tab.
class VideoPlayer : public QWidget
{
    Q_OBJECT

public:
    explicit VideoPlayer(int streamId, QWidget *parent = nullptr);
    ~VideoPlayer() override;

    int streamId() const
    {
        return m_streamId;
    }

    void play(const QString &url);
    void stop();
    void pause();
    void seekTo(qint64 ms);
    qint64 position() const;
    qint64 duration() const;
    QMediaPlayer::PlaybackState playbackState() const;

    // recording
    void startRecording(const QString &path, const QString &codec, double fps);
    void stopRecording();

    // audio
    void setMuted(bool muted);
    bool isMuted() const;
    void setVolume(float v); ///< 0.0 – 1.0
    float volume() const;
    void setAutoMuted(bool muted); ///< transient mute, does not affect isMuted()

    // snapshot
    void saveSnapshot();

    // full-screen overlay
    void toggleFullScreen();

signals:
    void errorOccurred(const QString &msg);
    void playbackStarted();
    void playbackStopped();

    void recordingStarted();
    void recordingFinished(const QString &path);
    void recordingError(const QString &msg);
    void autoRecordingStarted(const QString &path);
    void autoRecordingStopped(const QString &path);

    void snapshotSaved(const QString &path);
    void faceDetectionUnavailable();

    // File playback / overlay helpers
    void positionChanged(qint64 ms);
    void durationChanged(qint64 ms);
    void mediaPlaybackStateChanged(QMediaPlayer::PlaybackState state);
    void mouseMoved();

private slots:
    void displayFrame(const QImage &image);

private:
    void startWorker();
    void stopWorker();

    // Zoom & pan
    bool eventFilter(QObject *obj, QEvent *event) override;
    void resizeEvent(QResizeEvent *event) override;
    void updateDisplay();
    void clampPanOffset();
    QPointF labelToImageCoords(const QPointF &labelPos) const;
    void showZoomOverlay();
    void repositionZoomOverlay();
    void updateFullScreenLabel();

    int m_streamId;
    QLabel *m_displayLabel = nullptr;
    QMediaPlayer *m_player = nullptr;
    QVideoSink *m_captureSink = nullptr;
    QAudioOutput *m_audioOutput = nullptr;
    bool m_userMuted = false; ///< set by user (persisted in StreamState)
    bool m_autoMuted = false; ///< set by auto-mute logic (transient)

    // Video processing thread
    QThread *m_workerThread = nullptr;
    VideoWorker *m_worker = nullptr;

    // Recording thread (separate from video processing)
    QThread *m_recorderThread = nullptr;
    RecordingWorker *m_recorder = nullptr;

    // Raw-stream copy (QProcess-based, lives on UI thread – no extra thread needed)
    RawStreamWorker *m_rawWorker = nullptr;

    // Zoom & pan state
    double m_zoomFactor = 1.0;
    QPointF m_panOffset; // centre of viewed region in image pixels
    QImage m_lastImage;
    QSize m_lastImageSize; // detect resolution changes

    // Drag state
    bool m_isDragging = false;
    QPoint m_lastMousePos;

    // Whether the stream is currently playing (fast vs smooth scaling)
    bool m_streamPlaying = false;

    // Zoom overlay
    QLabel *m_zoomOverlay = nullptr;
    QTimer *m_zoomOverlayTimer = nullptr;

    // Full-screen overlay
    QWidget *m_fullScreenWindow = nullptr;
    QLabel *m_fullScreenLabel = nullptr;
};
