#include "videoplayer.h"
#include "recordingworker.h"
#include "streamstatemanager.h"
#include "videoworker.h"

#include <QMediaPlayer>
#include <QThread>
#include <QUrl>
#include <QVBoxLayout>
#include <QVideoFrame>
#include <QVideoSink>
#include <QVideoWidget>

// ─────────────────────────────────────────────────────────────────────────────
VideoPlayer::VideoPlayer(int streamId, QWidget *parent)
    : QWidget(parent)
    , m_streamId(streamId)
{
    m_videoWidget = new QVideoWidget(this);
    m_captureSink = new QVideoSink(this);
    m_player = new QMediaPlayer(this);

    // Player sends frames to our capture sink (not directly to the video widget)
    m_player->setVideoOutput(m_captureSink);

    auto *layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->addWidget(m_videoWidget);
    setLayout(layout);

    // Error forwarding
    connect(m_player, &QMediaPlayer::errorOccurred, this, [this](QMediaPlayer::Error, const QString &msg) {
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
// Worker + recorder thread management
// ─────────────────────────────────────────────────────────────────────────────
void VideoPlayer::startWorker()
{
    if (m_workerThread)
        return;

    // ── Recorder thread (must be created first so connections exist) ──
    m_recorderThread = new QThread(this);
    m_recorder = new RecordingWorker(); // no parent – moved to thread
    m_recorder->moveToThread(m_recorderThread);
    connect(m_recorderThread, &QThread::finished, m_recorder, &QObject::deleteLater);

    // Recording signals → VideoPlayer
    connect(m_recorder, &RecordingWorker::recordingStarted, this, &VideoPlayer::recordingStarted);
    connect(m_recorder, &RecordingWorker::recordingFinished, this, &VideoPlayer::recordingFinished);
    connect(m_recorder, &RecordingWorker::recordingError, this, &VideoPlayer::recordingError);

    m_recorderThread->start();

    // ── Video worker thread ──────────────────────────────────────────
    m_workerThread = new QThread(this);
    m_worker = new VideoWorker(m_streamId); // no parent – moved to thread
    m_worker->moveToThread(m_workerThread);
    connect(m_workerThread, &QThread::finished, m_worker, &QObject::deleteLater);

    // Frame submission: multimedia thread → worker (DirectConnection stores
    // latest frame atomically; worker's QTimer picks it up — drops stale frames)
    connect(m_captureSink, &QVideoSink::videoFrameChanged, m_worker, &VideoWorker::submitFrame, Qt::DirectConnection);
    connect(m_worker, &VideoWorker::frameReady, this, &VideoPlayer::displayFrame);
    connect(this, &VideoPlayer::pauseStateChanged, m_worker, &VideoWorker::setPaused);

    // Recording frame pipeline: VideoWorker → RecordingWorker (cross-thread)
    connect(m_worker, &VideoWorker::frameForRecording, m_recorder, &RecordingWorker::enqueueFrame);

    // Auto-record signals: VideoWorker → RecordingWorker
    connect(m_worker, &VideoWorker::startRecordingRequested, m_recorder, &RecordingWorker::startRecording);
    connect(m_worker, &VideoWorker::stopRecordingRequested, m_recorder, &RecordingWorker::stopRecording);

    // Auto-record UI signals → VideoPlayer
    connect(m_worker, &VideoWorker::autoRecordingStarted, this, &VideoPlayer::autoRecordingStarted);
    connect(m_worker, &VideoWorker::autoRecordingStopped, this, &VideoPlayer::autoRecordingStopped);

    m_workerThread->start();
}

void VideoPlayer::stopWorker()
{
    // ── Stop recorder thread first (may need to flush) ───────────────
    if (m_recorderThread) {
        if (m_recorder)
            m_recorder->requestInterrupt();
        m_recorderThread->quit();
        m_recorderThread->wait(3000); // max 3 seconds to drain
        m_recorderThread = nullptr;
        m_recorder = nullptr;
    }

    // ── Then stop the video worker ───────────────────────────────────
    if (m_workerThread) {
        m_workerThread->quit();
        m_workerThread->wait(2000);
        m_workerThread = nullptr;
        m_worker = nullptr;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Playback control
// ─────────────────────────────────────────────────────────────────────────────
void VideoPlayer::play(const QString &url)
{
    m_player->setSource(QUrl(url));
    m_player->play();

    if (m_worker)
        QMetaObject::invokeMethod(m_worker, "setStreamActive", Qt::QueuedConnection, Q_ARG(bool, true));

    StreamStateManager::instance().modifyState(m_streamId, [](StreamState &s) {
        s.playbackState = PlaybackState::Playing;
    });
    emit playbackStarted();
}

void VideoPlayer::stop()
{
    if (m_worker)
        QMetaObject::invokeMethod(m_worker, "setStreamActive", Qt::QueuedConnection, Q_ARG(bool, false));

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
// Recording forwarding  (GUI thread → recorder thread)
// ─────────────────────────────────────────────────────────────────────────────
void VideoPlayer::startRecording(const QString &path, const QString &codec, double fps)
{
    StreamStateManager::instance().modifyState(m_streamId, [](StreamState &s) {
        s.isRecording = true;
    });

    // Tell the video worker it should start sending frames for recording
    if (m_worker)
        QMetaObject::invokeMethod(m_worker, "setRecording", Qt::QueuedConnection, Q_ARG(bool, true));

    // Tell the recorder to open the file
    if (m_recorder)
        QMetaObject::invokeMethod(m_recorder, "startRecording", Qt::QueuedConnection, Q_ARG(QString, path), Q_ARG(QString, codec), Q_ARG(double, fps));
}

void VideoPlayer::stopRecording()
{
    // Tell the video worker to stop sending frames
    if (m_worker)
        QMetaObject::invokeMethod(m_worker, "setRecording", Qt::QueuedConnection, Q_ARG(bool, false));

    // Tell the recorder to flush + finalize
    if (m_recorder)
        QMetaObject::invokeMethod(m_recorder, "stopRecording", Qt::QueuedConnection);

    StreamStateManager::instance().modifyState(m_streamId, [](StreamState &s) {
        s.isRecording = false;
        s.isAutoRecording = false;
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// Display the composited frame on the video widget
// ─────────────────────────────────────────────────────────────────────────────
void VideoPlayer::displayFrame(const QImage &image)
{
    if (image.isNull() || !m_videoWidget->videoSink())
        return;
    m_videoWidget->videoSink()->setVideoFrame(QVideoFrame(image));
}
