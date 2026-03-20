#include "videoworker.h"
#include "opencvprocessor.h"
#include "streamstatemanager.h"

#include <QDateTime>
#include <QDir>
#include <QPainter>
#include <QRegularExpression>
#include <QTimer>
#include <QVideoFrame>

#include <opencv2/imgproc.hpp>

// ─────────────────────────────────────────────────────────────────────────────
VideoWorker::VideoWorker(int streamId, QObject *parent)
    : QObject(parent)
    , m_streamId(streamId)
{
    m_processor = new OpenCVProcessor();
    m_fpsTimer = QDateTime::currentDateTime();

    // Zero-interval timer drives frame processing on this thread's event loop.
    // It fires whenever the loop is idle, which naturally rate-limits processing
    // to the speed the CPU can handle while still servicing other slots.
    // Started/stopped by setStreamActive() to avoid CPU usage when idle.
    m_processTimer = new QTimer(this);
    connect(m_processTimer, &QTimer::timeout, this, &VideoWorker::processPendingFrame);
}

VideoWorker::~VideoWorker()
{
    delete m_processor;
}

// ─────────────────────────────────────────────────────────────────────────────
// Slots
// ─────────────────────────────────────────────────────────────────────────────
void VideoWorker::setStreamActive(bool a)
{
    m_streamActive = a;
    if (a) {
        if (!m_processTimer->isActive())
            m_processTimer->start(0);
    } else {
        m_processTimer->stop();
    }
}
void VideoWorker::setRecording(bool on)
{
    m_recording = on;
}
void VideoWorker::resetStream()
{
    m_cleanPreviousGray = cv::Mat{};
    m_processor->reset();
}

// ─────────────────────────────────────────────────────────────────────────────
// Frame submission (runs on multimedia thread — must be fast)
// ─────────────────────────────────────────────────────────────────────────────
void VideoWorker::submitFrame(const QVideoFrame &frame)
{
    // Atomically store the latest frame; previous unprocessed frame is discarded.
    QMutexLocker lk(&m_frameMutex);
    m_pendingFrame = frame;
    m_hasNewFrame.store(true, std::memory_order_release);
}

// ─────────────────────────────────────────────────────────────────────────────
// Timer-driven: grab latest frame and process it
// ─────────────────────────────────────────────────────────────────────────────
void VideoWorker::processPendingFrame()
{
    if (!m_hasNewFrame.load(std::memory_order_acquire))
        return;

    QVideoFrame frame;
    {
        QMutexLocker lk(&m_frameMutex);
        frame = m_pendingFrame;
        m_pendingFrame = QVideoFrame();
        m_hasNewFrame.store(false, std::memory_order_release);
    }

    processFrame(frame);
}

// ─────────────────────────────────────────────────────────────────────────────
// Core frame pipeline
// ─────────────────────────────────────────────────────────────────────────────
void VideoWorker::processFrame(const QVideoFrame &frame)
{
    if (!m_streamActive)
        return;

    // FPS bookkeeping
    ++m_frameCount;
    qint64 elapsed = m_fpsTimer.msecsTo(QDateTime::currentDateTime());
    if (elapsed >= 1000) {
        m_fps = m_frameCount * 1000.0 / elapsed;
        m_frameCount = 0;
        m_fpsTimer = QDateTime::currentDateTime();
    }

    // Convert QVideoFrame → QImage → BGR cv::Mat (once)
    QVideoFrame f(frame);
    f.map(QVideoFrame::ReadOnly);
    QImage rawImage = f.toImage();
    f.unmap();
    if (rawImage.isNull())
        return;

    // ── Read effect state (thread-safe snapshot) ────────────────────
    StreamState st;
    StreamStateManager::instance().readState(m_streamId, [&](const StreamState &s) {
        st = s;
    });

    cv::Mat bgr = m_processor->qImageToBGR(rawImage);

    bool needsBrightContrast = (st.brightnessAmount != 0 || st.contrastAmount != 0);
    bool needsColorTemp = (st.colorTemperature != 0);
    bool needsBlur = (st.blurAmount > 0);
    bool needsAdjust = needsBrightContrast || needsColorTemp || st.grayscaleEnabled || needsBlur;

    if (m_processor->haveOpenCL() && needsAdjust) {
        // Single upload to GPU
        cv::UMat uBgr;
        bgr.copyTo(uBgr);

        if (needsBrightContrast)
            m_processor->applyBrightnessContrast(uBgr, st.brightnessAmount, st.contrastAmount);
        if (needsColorTemp)
            m_processor->applyColorTemperature(uBgr, st.colorTemperature);
        if (st.grayscaleEnabled) {
            cv::UMat uGray;
            cv::cvtColor(uBgr, uGray, cv::COLOR_BGR2GRAY);
            cv::cvtColor(uGray, uBgr, cv::COLOR_GRAY2BGR);
        }
        if (needsBlur)
            m_processor->applyGaussBlur(uBgr, st.blurAmount);

        // Single download from GPU
        uBgr.copyTo(bgr);
    } else {
        // CPU path
        if (needsBrightContrast)
            m_processor->applyBrightnessContrast(bgr, st.brightnessAmount, st.contrastAmount);
        if (needsColorTemp)
            m_processor->applyColorTemperature(bgr, st.colorTemperature);
        if (st.grayscaleEnabled) {
            cv::Mat gray;
            cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
            cv::cvtColor(gray, bgr, cv::COLOR_GRAY2BGR);
        }
        if (needsBlur)
            m_processor->applyGaussBlur(bgr, st.blurAmount);
    }

    // 5. Prepare for detection — compute gray/BGR snapshots only if needed
    bool needsMotion = st.motionDetectionEnabled || st.motionVectorsEnabled || st.motionGraphEnabled || st.autoRecordEnabled;
    bool needsFace = st.faceDetectionEnabled;

    cv::Mat cleanBGR, cleanGray;
    if (needsFace)
        cleanBGR = bgr.clone();
    if (needsMotion)
        cv::cvtColor(bgr, cleanGray, cv::COLOR_BGR2GRAY);

    // 6. Convert BGR → QImage (once)
    QImage image = m_processor->bgrToQImage(bgr);

    // 7. Spike detection (once for all motion functions)
    bool isSpike = false;
    double motionLevel = 0.0;
    if (needsMotion && !m_cleanPreviousGray.empty())
        isSpike = m_processor->isSpikeFrame(cleanGray, m_cleanPreviousGray);

    if (isSpike) {
        motionLevel = m_processor->decayMotionLevels();
    } else {
        // 8. Motion detection overlay
        if (st.motionDetectionEnabled && !m_cleanPreviousGray.empty())
            m_processor->applyMotionDetectionOverlay(image, cleanGray, m_cleanPreviousGray, st.motionSensitivity);

        // 9. Motion vectors overlay
        if (st.motionVectorsEnabled && !m_cleanPreviousGray.empty())
            m_processor->applyMotionVectorsOverlay(image, cleanGray, m_cleanPreviousGray);

        // 10. Motion level (for graph + auto-record)
        if ((st.motionGraphEnabled || st.autoRecordEnabled) && !m_cleanPreviousGray.empty())
            motionLevel = m_processor->computeMotionLevel(cleanGray, m_cleanPreviousGray, st.motionGraphSensitivity);
    }

    // 11. Face detection (independent of motion spike)
    if (needsFace)
        m_processor->applyFaceDetection(image, cleanBGR);

    // 12. Grid motion overlay
    if (st.motionGraphEnabled)
        m_processor->applyGridMotionOverlay(image, st.motionGraphSensitivity);

    // 13. Motion graph overlay
    if (st.motionGraphEnabled)
        m_processor->applyMotionGraphOverlay(image, motionLevel);

    // 14. Save clean gray for next iteration
    if (needsMotion)
        m_cleanPreviousGray = cleanGray;

    // 15. FPS / resolution / datetime overlay
    if (st.overlayEnabled)
        paintFpsOverlay(image);

    // 16. Send frame to recording thread (non-blocking)
    if (m_recording && m_streamActive)
        emit frameForRecording(image);

    // 17. Auto-record logic
    if (st.autoRecordEnabled)
        handleAutoRecord(motionLevel);

    emit frameReady(image);
}

// ─────────────────────────────────────────────────────────────────────────────
// FPS / resolution / datetime overlay
// ─────────────────────────────────────────────────────────────────────────────
void VideoWorker::paintFpsOverlay(QImage &img)
{
    QPainter p(&img);
    p.setPen(QPen(QColor(0, 255, 0), 2)); // Bright green with 2px width for better visibility
    p.setFont(QFont(QStringLiteral("Monospace"), 13, QFont::Bold));

    QString fps = QStringLiteral("FPS: %1").arg(m_fps, 0, 'f', 1);
    QString res = QStringLiteral("Res: %1×%2").arg(img.width()).arg(img.height());
    QString dt = QDateTime::currentDateTime().toString(QStringLiteral("yyyy-MM-dd hh:mm:ss"));

    int x = img.width() - 240, y = 20;
    if (x < 10)
        x = 10;

    // Background box
    p.fillRect(x - 4, y - 14, 234, 52, QColor(0, 0, 0, 160));
    p.drawText(x, y, fps);
    p.drawText(x, y + 16, res);
    p.drawText(x, y + 32, dt);
    p.end();
}

// ─────────────────────────────────────────────────────────────────────────────
// Auto-record on motion
// ─────────────────────────────────────────────────────────────────────────────
void VideoWorker::handleAutoRecord(double motionLevel)
{
    StreamState st;
    StreamStateManager::instance().readState(m_streamId, [&](const StreamState &s) {
        st = s;
    });

    qint64 nowMs = QDateTime::currentMSecsSinceEpoch();

    if (motionLevel >= st.autoRecordThreshold) {
        // Reset the timeout countdown – keeps recording alive while motion
        // continues to exceed the threshold (ensures continuous recording).
        m_lastMotionAboveMs = nowMs;

        if (!m_autoRecording && !m_recording) {
            // Start auto-recording
            QString folder = StreamStateManager::instance().outputFolder();
            if (folder.isEmpty())
                return;

            QString ts = QDateTime::currentDateTime().toString(QStringLiteral("yyyy-MM-dd_HH-mm-ss"));
            QString cam = st.cameraName;
            cam.replace(QRegularExpression(QStringLiteral("[^a-zA-Z0-9_-]")), QStringLiteral("_"));
            QString ext = st.recordFormat;
            QString path = QStringLiteral("%1/%2_%3_motion.%4").arg(folder, ts, cam, ext);

            m_recPath = path;
            m_recCodec = st.recordCodec;
            m_recFps = st.recordFps;
            m_recording = true;
            m_autoRecording = true;
            m_autoRecStartTime = QDateTime::currentDateTime();

            StreamStateManager::instance().modifyState(m_streamId, [](StreamState &s) {
                s.isAutoRecording = true;
                s.isRecording = true;
            });

            // Tell RecordingWorker to open a file
            emit startRecordingRequested(path, m_recCodec, m_recFps);
            emit autoRecordingStarted(path);
        }
    }

    if (m_autoRecording) {
        int timeoutMs = st.autoRecordTimeout * 1000;
        if (nowMs - m_lastMotionAboveMs > timeoutMs) {
            // Motion below threshold for the full timeout → stop
            QString path = m_recPath;
            m_recording = false;
            m_autoRecording = false;

            StreamStateManager::instance().modifyState(m_streamId, [](StreamState &s) {
                s.isRecording = false;
                s.isAutoRecording = false;
            });

            // Tell RecordingWorker to finalize the file
            emit stopRecordingRequested();
            emit autoRecordingStopped(path);
        }
    }
}

// end of file
