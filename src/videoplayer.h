#pragma once

#include <QWidget>

class QMediaPlayer;
class QVideoWidget;
class QVideoSink;
class QThread;
class VideoWorker;
class RecordingWorker;

/// Widget that wraps QMediaPlayer + QVideoWidget + worker thread.
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
    void setPaused(bool paused);

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

    // internal
    void pauseStateChanged(bool paused);

private slots:
    void displayFrame(const QImage &image);

private:
    void startWorker();
    void stopWorker();

    int m_streamId;
    QMediaPlayer *m_player = nullptr;
    QVideoWidget *m_videoWidget = nullptr;
    QVideoSink *m_captureSink = nullptr;

    // Video processing thread
    QThread *m_workerThread = nullptr;
    VideoWorker *m_worker = nullptr;

    // Recording thread (separate from video processing)
    QThread *m_recorderThread = nullptr;
    RecordingWorker *m_recorder = nullptr;
};
