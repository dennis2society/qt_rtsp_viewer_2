#include "videoworker.h"
#include "opencvprocessor.h"
#include "streamstatemanager.h"

#include <QVideoFrame>
#include <QPainter>
#include <QDir>
#include <QDateTime>
#include <QRegularExpression>
#include <QTimer>

// ─────────────────────────────────────────────────────────────────────────────
VideoWorker::VideoWorker(int streamId, QObject *parent)
    : QObject(parent), m_streamId(streamId)
{
    m_processor = new OpenCVProcessor();
    m_fpsTimer  = QDateTime::currentDateTime();

    // Zero-interval timer drives frame processing on this thread's event loop.
    // It fires whenever the loop is idle, which naturally rate-limits processing
    // to the speed the CPU can handle while still servicing other slots.
    auto *timer = new QTimer(this);
    connect(timer, &QTimer::timeout, this, &VideoWorker::processPendingFrame);
    timer->start(0);
}

VideoWorker::~VideoWorker()
{
    delete m_processor;
}

// ─────────────────────────────────────────────────────────────────────────────
// Slots
// ─────────────────────────────────────────────────────────────────────────────
void VideoWorker::setPaused(bool p)   { m_paused = p; }
void VideoWorker::setStreamActive(bool a) { m_streamActive = a; }
void VideoWorker::setRecording(bool on)  { m_recording = on; }

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
    if (!m_hasNewFrame.load(std::memory_order_acquire)) return;

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
    if (!m_streamActive) return;

    // FPS bookkeeping
    ++m_frameCount;
    qint64 elapsed = m_fpsTimer.msecsTo(QDateTime::currentDateTime());
    if (elapsed >= 1000) {
        m_fps = m_frameCount * 1000.0 / elapsed;
        m_frameCount = 0;
        m_fpsTimer = QDateTime::currentDateTime();
    }

    if (m_paused) {
        if (!m_frozenFrame.isNull()) {
            QImage out = m_frozenFrame.copy();
            paintFpsOverlay(out);
            emit frameReady(out);
        }
        return;
    }

    // Convert QVideoFrame → QImage
    QVideoFrame f(frame);
    f.map(QVideoFrame::ReadOnly);
    QImage image = f.toImage();
    f.unmap();
    if (image.isNull()) return;

    // ── Read effect state (thread-safe snapshot) ────────────────────
    StreamState st;
    StreamStateManager::instance().readState(m_streamId, [&](const StreamState &s) {
        st = s;
    });

    // 1. Brightness / Contrast
    if (st.brightnessAmount != 0 || st.contrastAmount != 0)
        image = m_processor->applyBrightnessContrast(image, st.brightnessAmount,
                                                      st.contrastAmount);

    // 2. Colour temperature
    if (st.colorTemperature != 0)
        image = m_processor->applyColorTemperature(image, st.colorTemperature);

    // 3. Grayscale
    if (st.grayscaleEnabled)
        image = image.convertToFormat(QImage::Format_Grayscale8)
                      .convertToFormat(QImage::Format_RGB888);

    // 4. Blur
    if (st.blurAmount > 0)
        image = m_processor->applyGaussBlur(image, st.blurAmount);

    // 5. Clean snapshot
    QImage cleanImage = image.copy();

    // 6. Motion detection overlay
    if (st.motionDetectionEnabled)
        image = m_processor->applyMotionDetectionOverlay(
            image, cleanImage, m_cleanPrevious, st.motionSensitivity);

    // 7. Motion vectors overlay
    if (st.motionVectorsEnabled)
        image = m_processor->applyMotionVectorsOverlay(
            image, cleanImage, m_cleanPrevious);

    // 8. Face detection overlay
    if (st.faceDetectionEnabled)
        image = m_processor->applyFaceDetection(image, cleanImage);

    // 9. Motion level (for graph + auto-record)
    double motionLevel = 0.0;
    if (st.motionGraphEnabled || st.autoRecordEnabled) {
        motionLevel = m_processor->computeMotionLevel(
            cleanImage, m_cleanPrevious, st.motionGraphSensitivity);
    }

    // 10. Grid motion overlay
    if (st.motionGraphEnabled)
        image = m_processor->applyGridMotionOverlay(
            image, cleanImage, m_cleanPrevious, st.motionGraphSensitivity);

    // 11. Motion graph overlay
    if (st.motionGraphEnabled)
        image = m_processor->applyMotionGraphOverlay(image, motionLevel);

    // 12. Save clean frame for next iteration
    m_cleanPrevious = cleanImage;
    m_frozenFrame   = image;

    // 13. FPS / resolution / datetime overlay
    if (st.overlayEnabled)
        paintFpsOverlay(image);

    // 14. Send frame to recording thread (non-blocking)
    if (m_recording && m_streamActive)
        emit frameForRecording(image);

    // 15. Auto-record logic
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
    p.setPen(QPen(QColor(0, 255, 0), 2));  // Bright green with 2px width for better visibility
    p.setFont(QFont(QStringLiteral("Monospace"), 13, QFont::Bold));

    QString fps = QStringLiteral("FPS: %1").arg(m_fps, 0, 'f', 1);
    QString res = QStringLiteral("Res: %1×%2").arg(img.width()).arg(img.height());
    QString dt  = QDateTime::currentDateTime().toString(QStringLiteral("yyyy-MM-dd hh:mm:ss"));

    int x = img.width() - 240, y = 20;
    if (x < 10) x = 10;

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
            if (folder.isEmpty()) return;

            QString ts = QDateTime::currentDateTime().toString(
                QStringLiteral("yyyy-MM-dd_HH-mm-ss"));
            QString cam = st.cameraName;
            cam.replace(QRegularExpression(QStringLiteral("[^a-zA-Z0-9_-]")),
                        QStringLiteral("_"));
            QString ext = st.recordFormat;
            QString path = QStringLiteral("%1/%2_%3_motion.%4")
                               .arg(folder, ts, cam, ext);

            m_recPath    = path;
            m_recCodec   = st.recordCodec;
            m_recFps     = st.recordFps;
            m_recording  = true;
            m_autoRecording   = true;
            m_autoRecStartTime = QDateTime::currentDateTime();

            StreamStateManager::instance().modifyState(m_streamId, [](StreamState &s) {
                s.isAutoRecording = true;
                s.isRecording     = true;
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
            m_recording     = false;
            m_autoRecording = false;

            StreamStateManager::instance().modifyState(m_streamId, [](StreamState &s) {
                s.isRecording     = false;
                s.isAutoRecording = false;
            });

            // Tell RecordingWorker to finalize the file
            emit stopRecordingRequested();
            emit autoRecordingStopped(path);
        }
    }
}

// end of file
