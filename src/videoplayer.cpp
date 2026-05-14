#include "videoplayer.h"
#include "rawstreamworker.h"
#include "recordingworker.h"
#include "streamstatemanager.h"
#include "videoworker.h"

#include <QEvent>
#include <QLabel>
#include <QMediaPlayer>
#include <QMouseEvent>
#include <QPixmap>
#include <QResizeEvent>
#include <QThread>
#include <QTimer>
#include <QUrl>
#include <QVBoxLayout>
#include <QVideoFrame>
#include <QVideoSink>
#include <QWheelEvent>

// -----------------------------------------------------------------------------
VideoPlayer::VideoPlayer(int streamId, QWidget *parent)
    : QWidget(parent)
    , m_streamId(streamId)
{
    m_displayLabel = new QLabel(this);
    m_displayLabel->setAlignment(Qt::AlignCenter);
    m_displayLabel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    m_displayLabel->setStyleSheet(QStringLiteral("background-color: black;"));
    m_displayLabel->setMinimumSize(1, 1);
    m_displayLabel->setMouseTracking(true);
    m_displayLabel->installEventFilter(this);

    m_captureSink = new QVideoSink(this);
    m_player = new QMediaPlayer(this);

    // Player sends frames to our capture sink for processing
    m_player->setVideoOutput(m_captureSink);

    auto *layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->addWidget(m_displayLabel);
    setLayout(layout);

    // Zoom overlay (floats above the layout as a bare child widget)
    m_zoomOverlay = new QLabel(this);
    m_zoomOverlay->setStyleSheet(
        "QLabel {"
        "  background-color: rgba(0,0,0,160);"
        "  color: white;"
        "  font-size: 14px;"
        "  font-weight: bold;"
        "  border-radius: 4px;"
        "  padding: 4px 10px;"
        "}");
    m_zoomOverlay->setAlignment(Qt::AlignCenter);
    m_zoomOverlay->setAttribute(Qt::WA_TransparentForMouseEvents);
    m_zoomOverlay->hide();

    m_zoomOverlayTimer = new QTimer(this);
    m_zoomOverlayTimer->setSingleShot(true);
    connect(m_zoomOverlayTimer, &QTimer::timeout, m_zoomOverlay, &QLabel::hide);

    // Error forwarding
    connect(m_player, &QMediaPlayer::errorOccurred, this, [this](QMediaPlayer::Error, const QString &msg) {
        emit errorOccurred(msg);
    });

    // Forward position/duration/state for file player use
    connect(m_player, &QMediaPlayer::positionChanged, this, &VideoPlayer::positionChanged);
    connect(m_player, &QMediaPlayer::durationChanged, this, &VideoPlayer::durationChanged);
    connect(m_player, &QMediaPlayer::playbackStateChanged, this, &VideoPlayer::mediaPlaybackStateChanged);

    // Raw stream copy worker – lives on the UI thread (QProcess is event-loop driven).
    // Signals connected once here to avoid the duplicate-lambda accumulation problem.
    m_rawWorker = new RawStreamWorker(this);
    connect(m_rawWorker, &RawStreamWorker::copyStarted, this, &VideoPlayer::recordingStarted);
    connect(m_rawWorker, &RawStreamWorker::copyFinished, this, [this](const QString &p) {
        StreamStateManager::instance().modifyState(m_streamId, [](StreamState &s) {
            s.isRecording = false;
        });
        emit recordingFinished(p);
    });
    connect(m_rawWorker, &RawStreamWorker::copyError, this, [this](const QString &msg) {
        StreamStateManager::instance().modifyState(m_streamId, [](StreamState &s) {
            s.isRecording = false;
        });
        emit recordingError(msg);
    });
}

VideoPlayer::~VideoPlayer()
{
    stop();
    stopWorker();
}

// -----------------------------------------------------------------------------
// Worker + recorder thread management
// -----------------------------------------------------------------------------
void VideoPlayer::startWorker()
{
    if (m_workerThread)
        return;

    // -- Recorder thread (must be created first so connections exist) --
    m_recorderThread = new QThread(this);
    m_recorder = new RecordingWorker(); // no parent - moved to thread
    m_recorder->moveToThread(m_recorderThread);
    connect(m_recorderThread, &QThread::finished, m_recorder, &QObject::deleteLater);

    // Recording signals -> VideoPlayer
    connect(m_recorder, &RecordingWorker::recordingStarted, this, &VideoPlayer::recordingStarted);
    connect(m_recorder, &RecordingWorker::recordingFinished, this, &VideoPlayer::recordingFinished);
    connect(m_recorder, &RecordingWorker::recordingError, this, &VideoPlayer::recordingError);

    m_recorderThread->start();

    // -- Video worker thread ------------------------------------------
    m_workerThread = new QThread(this);
    m_worker = new VideoWorker(m_streamId); // no parent - moved to thread
    m_worker->moveToThread(m_workerThread);
    connect(m_workerThread, &QThread::finished, m_worker, &QObject::deleteLater);

    // Frame submission: multimedia thread -> worker (DirectConnection stores
    // latest frame atomically; worker's QTimer picks it up - drops stale frames)
    connect(m_captureSink, &QVideoSink::videoFrameChanged, m_worker, &VideoWorker::submitFrame, Qt::DirectConnection);
    connect(m_worker, &VideoWorker::frameReady, this, &VideoPlayer::displayFrame);

    // Recording frame pipeline: VideoWorker -> RecordingWorker (cross-thread)
    connect(m_worker, &VideoWorker::frameForRecording, m_recorder, &RecordingWorker::enqueueFrame);

    // Auto-record signals: VideoWorker -> RecordingWorker
    connect(m_worker, &VideoWorker::startRecordingRequested, m_recorder, &RecordingWorker::startRecording);
    connect(m_worker, &VideoWorker::stopRecordingRequested, m_recorder, &RecordingWorker::stopRecording);

    // Auto-record UI signals -> VideoPlayer
    connect(m_worker, &VideoWorker::autoRecordingStarted, this, &VideoPlayer::autoRecordingStarted);
    connect(m_worker, &VideoWorker::autoRecordingStopped, this, &VideoPlayer::autoRecordingStopped);

    m_workerThread->start();
}

void VideoPlayer::stopWorker()
{
    // -- Stop raw copy process immediately (destructor also handles it) ---
    if (m_rawWorker && m_rawWorker->isRunning())
        m_rawWorker->requestStop();

    // -- Disconnect frame delivery first ------------------------------
    // The multimedia thread delivers frames via DirectConnection to the
    // worker.  We must sever that link before tearing down threads,
    // otherwise the multimedia thread can call into a dying worker.
    if (m_worker)
        disconnect(m_captureSink, &QVideoSink::videoFrameChanged, m_worker, &VideoWorker::submitFrame);

    // -- Stop recorder thread first (may need to flush) ---------------
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

    // -- Then stop the video worker -----------------------------------
    if (m_workerThread) {
        if (m_worker)
            disconnect(m_worker, nullptr, nullptr, nullptr);
        m_workerThread->quit();
        m_workerThread->wait(5000);
        m_workerThread = nullptr;
        m_worker = nullptr;
    }
}

// -----------------------------------------------------------------------------
// Playback control
// -----------------------------------------------------------------------------
void VideoPlayer::play(const QString &url)
{
    // Ensure worker threads are running (no-op if already alive)
    startWorker();

    if (!url.isEmpty()) {
        // New source: reset inter-frame state and set the new URL
        if (m_worker)
            QMetaObject::invokeMethod(m_worker, "resetStream", Qt::QueuedConnection);
        m_player->setSource(QUrl(url));
    }
    // Empty url = resume current source (file player pause→play)

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

    // Don't tear down worker threads - they are reused on the next play().
    // A deactivated worker with a stopped QTimer consumes zero CPU.

    StreamStateManager::instance().modifyState(m_streamId, [](StreamState &s) {
        s.playbackState = PlaybackState::Stopped;
    });
    emit playbackStopped();
}

void VideoPlayer::pause()
{
    m_player->pause();
}

void VideoPlayer::seekTo(qint64 ms)
{
    m_player->setPosition(ms);
}

qint64 VideoPlayer::position() const
{
    return m_player->position();
}

qint64 VideoPlayer::duration() const
{
    return m_player->duration();
}

QMediaPlayer::PlaybackState VideoPlayer::playbackState() const
{
    return m_player->playbackState();
}

// -----------------------------------------------------------------------------
// Recording forwarding  (GUI thread -> recorder thread)
// -----------------------------------------------------------------------------
void VideoPlayer::startRecording(const QString &path, const QString &codec, double fps)
{
    // ── Raw stream copy (ffmpeg binary, no re-encode) ─────────────────────────
    if (codec == QLatin1String("raw_copy")) {
        QString url;
        StreamStateManager::instance().readState(m_streamId, [&](const StreamState &s) {
            url = s.rtspUrl;
        });
        if (url.isEmpty()) {
            emit recordingError(QStringLiteral("No stream URL set – start playback first"));
            return;
        }

        StreamStateManager::instance().modifyState(m_streamId, [](StreamState &s) {
            s.isRecording = true;
        });
        m_rawWorker->startCopy(url, path);
        return;
    }

    // ── Encoded recording (FFmpeg libraries via RecordingWorker) ──────────────
    StreamStateManager::instance().modifyState(m_streamId, [](StreamState &s) {
        s.isRecording = true;
    });

    // Tell the video worker it should start sending frames for recording
    if (m_worker)
        QMetaObject::invokeMethod(m_worker, "setRecording", Qt::QueuedConnection, Q_ARG(bool, true), Q_ARG(QString, path));

    // Tell the recorder to open the file
    if (m_recorder)
        QMetaObject::invokeMethod(m_recorder, "startRecording", Qt::QueuedConnection, Q_ARG(QString, path), Q_ARG(QString, codec), Q_ARG(double, fps));
}

void VideoPlayer::stopRecording()
{
    // ── Raw copy active? ──────────────────────────────────────────────────────
    if (m_rawWorker && m_rawWorker->isRunning()) {
        m_rawWorker->requestStop();
        // State + signals are handled by copyFinished / copyError slots above.
        return;
    }

    // ── Encoded recording ─────────────────────────────────────────────────────
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

// -----------------------------------------------------------------------------
// Display the composited frame on the video widget
// -----------------------------------------------------------------------------
void VideoPlayer::displayFrame(const QImage &image)
{
    if (image.isNull())
        return;

    // Reset zoom & pan whenever the stream resolution changes
    if (image.size() != m_lastImageSize) {
        m_zoomFactor = 1.0;
        m_lastImageSize = image.size();
        m_panOffset = QPointF(image.width() / 2.0, image.height() / 2.0);
        m_isDragging = false;
        m_displayLabel->setCursor(Qt::ArrowCursor);
    }

    m_lastImage = image;
    updateDisplay();
}

// -----------------------------------------------------------------------------
// Zoom / pan helpers
// -----------------------------------------------------------------------------
void VideoPlayer::updateDisplay()
{
    if (m_lastImage.isNull())
        return;

    QImage imgToShow;
    if (m_zoomFactor <= 1.0) {
        imgToShow = m_lastImage;
    } else {
        const QSizeF cropSize = QSizeF(m_lastImage.size()) / m_zoomFactor;
        QPointF cropTL = m_panOffset - QPointF(cropSize.width() / 2.0, cropSize.height() / 2.0);
        cropTL.setX(qBound(0.0, cropTL.x(), m_lastImage.width() - cropSize.width()));
        cropTL.setY(qBound(0.0, cropTL.y(), m_lastImage.height() - cropSize.height()));
        imgToShow = m_lastImage.copy(QRect(qRound(cropTL.x()), qRound(cropTL.y()), qRound(cropSize.width()), qRound(cropSize.height())));
    }

    m_displayLabel->setPixmap(QPixmap::fromImage(imgToShow).scaled(m_displayLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}

void VideoPlayer::clampPanOffset()
{
    if (m_lastImage.isNull() || m_zoomFactor <= 1.0)
        return;
    const QSizeF cropSize = QSizeF(m_lastImage.size()) / m_zoomFactor;
    m_panOffset.setX(qBound(cropSize.width() / 2.0, m_panOffset.x(), m_lastImage.width() - cropSize.width() / 2.0));
    m_panOffset.setY(qBound(cropSize.height() / 2.0, m_panOffset.y(), m_lastImage.height() - cropSize.height() / 2.0));
}

// Map a position in m_displayLabel's coordinate system to image pixel coordinates,
// accounting for the current zoom crop.
QPointF VideoPlayer::labelToImageCoords(const QPointF &labelPos) const
{
    if (m_lastImage.isNull())
        return {};
    const QSize labelSize = m_displayLabel->size();
    const QSizeF cropSize = QSizeF(m_lastImage.size()) / m_zoomFactor;
    const QSizeF scaledSize = cropSize.scaled(labelSize, Qt::KeepAspectRatio);
    const QPointF pixmapTL((labelSize.width() - scaledSize.width()) / 2.0, (labelSize.height() - scaledSize.height()) / 2.0);
    const QPointF inPixmap = labelPos - pixmapTL;
    const QPointF cropPos(inPixmap.x() / scaledSize.width() * cropSize.width(), inPixmap.y() / scaledSize.height() * cropSize.height());

    // Crop top-left in image space (mirroring updateDisplay clamping)
    QPointF cropTL = m_panOffset - QPointF(cropSize.width() / 2.0, cropSize.height() / 2.0);
    cropTL.setX(qBound(0.0, cropTL.x(), m_lastImage.width() - cropSize.width()));
    cropTL.setY(qBound(0.0, cropTL.y(), m_lastImage.height() - cropSize.height()));
    return cropTL + cropPos;
}

void VideoPlayer::showZoomOverlay()
{
    const QString text = (m_zoomFactor < 1.005) ? QStringLiteral("1\u00d7") : QString("%1\u00d7").arg(m_zoomFactor, 0, 'f', 1);
    m_zoomOverlay->setText(text);
    repositionZoomOverlay();
    m_zoomOverlay->show();
    m_zoomOverlay->raise();
    m_zoomOverlayTimer->start(1500);
}

void VideoPlayer::repositionZoomOverlay()
{
    if (!m_zoomOverlay)
        return;
    m_zoomOverlay->adjustSize();
    constexpr int margin = 8;
    m_zoomOverlay->move(width() - m_zoomOverlay->width() - margin, margin);
}

// -----------------------------------------------------------------------------
// Event filter — handles wheel zoom and left-button pan on m_displayLabel
// -----------------------------------------------------------------------------
bool VideoPlayer::eventFilter(QObject *obj, QEvent *event)
{
    if (obj != m_displayLabel)
        return QObject::eventFilter(obj, event);

    switch (event->type()) {
    case QEvent::Wheel: {
        auto *we = static_cast<QWheelEvent *>(event);
        if (m_lastImage.isNull())
            break;

        // Image coordinates under the cursor before the zoom change
        const QPointF imagePoint = labelToImageCoords(we->position());

        const double zoomStep = 1.15;
        if (we->angleDelta().y() > 0)
            m_zoomFactor *= zoomStep;
        else
            m_zoomFactor /= zoomStep;
        m_zoomFactor = qBound(1.0, m_zoomFactor, 16.0);

        if (m_zoomFactor <= 1.0) {
            m_zoomFactor = 1.0;
            m_panOffset = QPointF(m_lastImage.width() / 2.0, m_lastImage.height() / 2.0);
            m_displayLabel->setCursor(Qt::ArrowCursor);
        } else {
            // Recompute pan so the image point under the cursor stays fixed
            const QSize labelSize = m_displayLabel->size();
            const QSizeF cropSize = QSizeF(m_lastImage.size()) / m_zoomFactor;
            const QSizeF scaledSize = cropSize.scaled(labelSize, Qt::KeepAspectRatio);
            const QPointF pixmapTL((labelSize.width() - scaledSize.width()) / 2.0, (labelSize.height() - scaledSize.height()) / 2.0);
            const QPointF cursorInPixmap = we->position() - pixmapTL;
            const QPointF frac(qBound(0.0, cursorInPixmap.x() / scaledSize.width(), 1.0), qBound(0.0, cursorInPixmap.y() / scaledSize.height(), 1.0));

            m_panOffset =
                imagePoint - QPointF(frac.x() * cropSize.width(), frac.y() * cropSize.height()) + QPointF(cropSize.width() / 2.0, cropSize.height() / 2.0);
            clampPanOffset();
            m_displayLabel->setCursor(Qt::OpenHandCursor);
        }

        updateDisplay();
        showZoomOverlay();
        return true;
    }

    case QEvent::MouseButtonPress: {
        auto *me = static_cast<QMouseEvent *>(event);
        if (me->button() == Qt::LeftButton && m_zoomFactor > 1.0) {
            m_isDragging = true;
            m_lastMousePos = me->pos();
            m_displayLabel->setCursor(Qt::ClosedHandCursor);
            return true;
        }
        break;
    }

    case QEvent::MouseMove: {
        auto *me = static_cast<QMouseEvent *>(event);
        emit mouseMoved();
        if (m_isDragging && !m_lastImage.isNull()) {
            const QPoint delta = me->pos() - m_lastMousePos;
            m_lastMousePos = me->pos();
            const QSize labelSize = m_displayLabel->size();
            const QSizeF cropSize = QSizeF(m_lastImage.size()) / m_zoomFactor;
            const QSizeF scaledSize = cropSize.scaled(labelSize, Qt::KeepAspectRatio);
            // Dragging right pulls the view left → pan offset increases
            m_panOffset += QPointF(-delta.x() / scaledSize.width() * cropSize.width(), -delta.y() / scaledSize.height() * cropSize.height());
            clampPanOffset();
            updateDisplay();
            return true;
        }
        break;
    }

    case QEvent::MouseButtonRelease: {
        auto *me = static_cast<QMouseEvent *>(event);
        if (me->button() == Qt::LeftButton && m_isDragging) {
            m_isDragging = false;
            m_displayLabel->setCursor(m_zoomFactor > 1.0 ? Qt::OpenHandCursor : Qt::ArrowCursor);
            return true;
        }
        break;
    }

    default:
        break;
    }

    return QObject::eventFilter(obj, event);
}

void VideoPlayer::resizeEvent(QResizeEvent *event)
{
    QWidget::resizeEvent(event);
    updateDisplay();
    repositionZoomOverlay();
}
