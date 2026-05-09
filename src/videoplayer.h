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
class VideoWorker;
class RecordingWorker;

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

signals:
    void errorOccurred(const QString &msg);
    void playbackStarted();
    void playbackStopped();

    void recordingStarted();
    void recordingFinished(const QString &path);
    void recordingError(const QString &msg);
    void autoRecordingStarted(const QString &path);
    void autoRecordingStopped(const QString &path);

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

    int m_streamId;
    QLabel *m_displayLabel = nullptr;
    QMediaPlayer *m_player = nullptr;
    QVideoSink *m_captureSink = nullptr;

    // Video processing thread
    QThread *m_workerThread = nullptr;
    VideoWorker *m_worker = nullptr;

    // Recording thread (separate from video processing)
    QThread *m_recorderThread = nullptr;
    RecordingWorker *m_recorder = nullptr;

    // Zoom & pan state
    double m_zoomFactor = 1.0;
    QPointF m_panOffset; // centre of viewed region in image pixels
    QImage m_lastImage;
    QSize m_lastImageSize; // detect resolution changes

    // Drag state
    bool m_isDragging = false;
    QPoint m_lastMousePos;

    // Zoom overlay
    QLabel *m_zoomOverlay = nullptr;
    QTimer *m_zoomOverlayTimer = nullptr;
};
