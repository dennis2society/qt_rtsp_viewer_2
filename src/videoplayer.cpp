#include "videoplayer.h"
#include "videoworker.h"
#include "streamstatemanager.h"

#include <QMediaPlayer>
#include <QVideoWidget>
#include <QVideoSink>
#include <QVideoFrame>
#include <QVBoxLayout>
#include <QThread>
#include <QUrl>

// ─────────────────────────────────────────────────────────────────────────────
VideoPlayer::VideoPlayer(int streamId, QWidget *parent)
    : QWidget(parent), m_streamId(streamId)
{
    m_videoWidget  = new QVideoWidget(this);
    m_captureSink  = new QVideoSink(this);
    m_player       = new QMediaPlayer(this);

    // Player sends frames to our capture sink (not directly to the video widget)
    m_player->setVideoOutput(m_captureSink);

    auto *layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->addWidget(m_videoWidget);
    setLayout(layout);

    // Error forwarding
    connect(m_player, &QMediaPlayer::errorOccurred, this,
            [this](QMediaPlayer::Error, const QString &msg) {
                emit errorOccurred(msg);
            });

    startWorker();
}

VideoPlayer::~VideoPlayer()
{
    stop();
    stopWorker();
}

// ─────────────────────────────────────────────────────────────────────────────
// Worker thread management
// ─────────────────────────────────────────────────────────────────────────────
void VideoPlayer::startWorker()
{
    if (m_workerThread) return;

    m_workerThread = new QThread(this);
    m_worker       = new VideoWorker(m_streamId);    // no parent – moved to thread

    m_worker->moveToThread(m_workerThread);

    connect(m_workerThread, &QThread::finished, m_worker, &QObject::deleteLater);

    // Frame pipeline (cross-thread, queued)
    connect(m_captureSink, &QVideoSink::videoFrameChanged,
            m_worker, &VideoWorker::processFrame);
    connect(m_worker, &VideoWorker::frameReady,
            this, &VideoPlayer::displayFrame);
    connect(this, &VideoPlayer::pauseStateChanged,
            m_worker, &VideoWorker::setPaused);

    // Recording signals relay
    connect(m_worker, &VideoWorker::recordingStarted,
            this, &VideoPlayer::recordingStarted);
    connect(m_worker, &VideoWorker::recordingFinished,
            this, &VideoPlayer::recordingFinished);
    connect(m_worker, &VideoWorker::recordingError,
            this, &VideoPlayer::recordingError);
    connect(m_worker, &VideoWorker::autoRecordingStarted,
            this, &VideoPlayer::autoRecordingStarted);
    connect(m_worker, &VideoWorker::autoRecordingStopped,
            this, &VideoPlayer::autoRecordingStopped);

    m_workerThread->start();
}

void VideoPlayer::stopWorker()
{
    if (!m_workerThread) return;
    m_workerThread->quit();
    m_workerThread->wait();
    m_workerThread = nullptr;      // deleteLater handles worker
    m_worker = nullptr;
}

// ─────────────────────────────────────────────────────────────────────────────
// Playback control
// ─────────────────────────────────────────────────────────────────────────────
void VideoPlayer::play(const QString &url)
{
    m_player->setSource(QUrl(url));
    m_player->play();

    if (m_worker)
        QMetaObject::invokeMethod(m_worker, "setStreamActive",
                                  Qt::QueuedConnection, Q_ARG(bool, true));

    StreamStateManager::instance().modifyState(m_streamId, [](StreamState &s) {
        s.playbackState = PlaybackState::Playing;
    });
    emit playbackStarted();
}

void VideoPlayer::stop()
{
    if (m_worker)
        QMetaObject::invokeMethod(m_worker, "setStreamActive",
                                  Qt::QueuedConnection, Q_ARG(bool, false));

    m_player->stop();
    m_player->setSource(QUrl());

    StreamStateManager::instance().modifyState(m_streamId, [](StreamState &s) {
        s.playbackState = PlaybackState::Stopped;
    });
    emit playbackStopped();
}

void VideoPlayer::setPaused(bool paused)
{
    if (paused)
        m_player->pause();
    else
        m_player->play();

    emit pauseStateChanged(paused);

    StreamStateManager::instance().modifyState(m_streamId, [paused](StreamState &s) {
        s.playbackState = paused ? PlaybackState::Paused : PlaybackState::Playing;
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// Recording forwarding
// ─────────────────────────────────────────────────────────────────────────────
void VideoPlayer::startRecording(const QString &path, const QString &codec,
                                  double fps)
{
    StreamStateManager::instance().modifyState(m_streamId, [](StreamState &s) {
        s.isRecording = true;
    });
    if (m_worker)
        QMetaObject::invokeMethod(m_worker, "startRecording",
                                  Qt::QueuedConnection,
                                  Q_ARG(QString, path),
                                  Q_ARG(QString, codec),
                                  Q_ARG(double, fps));
}

void VideoPlayer::stopRecording()
{
    if (m_worker)
        QMetaObject::invokeMethod(m_worker, "stopRecording",
                                  Qt::QueuedConnection);
}

// ─────────────────────────────────────────────────────────────────────────────
// Display the composited frame on the video widget
// ─────────────────────────────────────────────────────────────────────────────
void VideoPlayer::displayFrame(const QImage &image)
{
    if (image.isNull() || !m_videoWidget->videoSink()) return;
    m_videoWidget->videoSink()->setVideoFrame(
        QVideoFrame(image));
}
