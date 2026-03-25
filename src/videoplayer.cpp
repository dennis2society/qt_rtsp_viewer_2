#include "videoplayer.h"
#include "recordingworker.h"
#include "streamstatemanager.h"
#include "videoworker.h"

#include <QLabel>
#include <QMediaPlayer>
#include <QPixmap>
#include <QThread>
#include <QUrl>
#include <QVBoxLayout>
#include <QVideoFrame>
#include <QVideoSink>

// ─────────────────────────────────────────────────────────────────────────────
VideoPlayer::VideoPlayer(int streamId, QWidget *parent)
    : QWidget(parent)
    , m_streamId(streamId)
{
    m_displayLabel = new QLabel(this);
    m_displayLabel->setAlignment(Qt::AlignCenter);
    m_displayLabel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    m_displayLabel->setStyleSheet(QStringLiteral("background-color: black;"));
    m_displayLabel->setMinimumSize(1, 1);

    m_captureSink = new QVideoSink(this);
    m_player = new QMediaPlayer(this);

    // Player sends frames to our capture sink for processing
    m_player->setVideoOutput(m_captureSink);

    auto *layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->addWidget(m_displayLabel);
    setLayout(layout);

    // Error forwarding
    connect(m_player, &QMediaPlayer::errorOccurred, this, [this](QMediaPlayer::Error, const QString &msg) {
        emit errorOccurred(msg);
    });
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
    // ── Disconnect frame delivery first ──────────────────────────────
    // The multimedia thread delivers frames via DirectConnection to the
    // worker.  We must sever that link before tearing down threads,
    // otherwise the multimedia thread can call into a dying worker.
    if (m_worker)
        disconnect(m_captureSink, &QVideoSink::videoFrameChanged,
                   m_worker, &VideoWorker::submitFrame);

    // ── Stop recorder thread first (may need to flush) ───────────────
    if (m_recorderThread) {
        if (m_recorder) {
            disconnect(m_recorder, nullptr, nullptr, nullptr);
            m_recorder->requestInterrupt();
        }
        m_recorderThread->quit();
        m_recorderThread->wait(5000);
        m_recorderThread = nullptr;
        m_recorder = nullptr;
    }

    // ── Then stop the video worker ───────────────────────────────────
    if (m_workerThread) {
        if (m_worker)
            disconnect(m_worker, nullptr, nullptr, nullptr);
        m_workerThread->quit();
        m_workerThread->wait(5000);
        m_workerThread = nullptr;
        m_worker = nullptr;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Playback control
// ─────────────────────────────────────────────────────────────────────────────
void VideoPlayer::play(const QString &url)
{
    // Ensure worker threads are running (no-op if already alive)
    startWorker();

    if (m_worker)
        QMetaObject::invokeMethod(m_worker, "resetStream", Qt::QueuedConnection);

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
    // Deactivate the worker first (stops processing timer)
    if (m_worker)
        QMetaObject::invokeMethod(m_worker, "setStreamActive", Qt::QueuedConnection, Q_ARG(bool, false));

    m_player->stop();
    m_player->setSource(QUrl());

    // Don't tear down worker threads — they are reused on the next play().
    // A deactivated worker with a stopped QTimer consumes zero CPU.

    StreamStateManager::instance().modifyState(m_streamId, [](StreamState &s) {
        s.playbackState = PlaybackState::Stopped;
    });
    emit playbackStopped();
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
    if (image.isNull())
        return;
    m_displayLabel->setPixmap(QPixmap::fromImage(image).scaled(m_displayLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}
