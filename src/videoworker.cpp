#include "videoworker.h"
#include "opencvprocessor.h"
#include "streamstatemanager.h"

#include <QVideoFrame>
#include <QPainter>
#include <QDir>
#include <QDateTime>
#include <QRegularExpression>

#include <string>

// ─────────────────────────────────────────────────────────────────────────────
VideoWorker::VideoWorker(int streamId, QObject *parent)
    : QObject(parent), m_streamId(streamId)
{
    m_processor = new OpenCVProcessor();
    m_fpsTimer  = QDateTime::currentDateTime();
}

VideoWorker::~VideoWorker()
{
#ifdef HAVE_FFMPEG
    if (m_recOpen)
        closeRecorder();
#endif
    delete m_processor;
}

// ─────────────────────────────────────────────────────────────────────────────
// Slots
// ─────────────────────────────────────────────────────────────────────────────
void VideoWorker::setPaused(bool p)   { m_paused = p; }
void VideoWorker::setStreamActive(bool a) { m_streamActive = a; }

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

    // 14. Recording
#ifdef HAVE_FFMPEG
    if (m_recording && m_streamActive)
        writeRecordingFrame(image);
#endif

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
#ifdef HAVE_FFMPEG
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
            m_recOpen    = false;
            m_recFrameIndex = 0;
            m_autoRecording   = true;
            m_autoRecStartTime = QDateTime::currentDateTime();

            emit autoRecordingStarted(path);
        }
    }

    if (m_autoRecording) {
        int timeoutMs = st.autoRecordTimeout * 1000;
        if (nowMs - m_lastMotionAboveMs > timeoutMs) {
            // Motion below threshold for the full timeout → stop
            QString path = m_recPath;
            stopRecording();
            m_autoRecording = false;
            StreamStateManager::instance().modifyState(m_streamId, [](StreamState &s) {
                s.isAutoRecording = false;
            });
            emit autoRecordingStopped(path);
        }
    }
#else
    Q_UNUSED(motionLevel);
#endif
}

// ─────────────────────────────────────────────────────────────────────────────
// Manual recording control
// ─────────────────────────────────────────────────────────────────────────────
void VideoWorker::startRecording(const QString &path, const QString &codec,
                                  double fps)
{
#ifdef HAVE_FFMPEG
    if (m_recording) return;
    m_recPath       = path;
    m_recCodec      = codec;
    m_recFps        = fps;
    m_recording     = true;
    m_recOpen       = false;
    m_recFrameIndex = 0;
    emit recordingStarted();
#else
    Q_UNUSED(path); Q_UNUSED(codec); Q_UNUSED(fps);
    emit recordingError(QStringLiteral("FFmpeg support not compiled in"));
#endif
}

void VideoWorker::stopRecording()
{
#ifdef HAVE_FFMPEG
    if (!m_recording) return;
    m_recording = false;
    QString path = m_recPath;
    closeRecorder();
    StreamStateManager::instance().modifyState(m_streamId, [](StreamState &s) {
        s.isRecording     = false;
        s.isAutoRecording = false;
    });
    emit recordingFinished(path);
#endif
}

// ─────────────────────────────────────────────────────────────────────────────
// FFmpeg recording internals
// ─────────────────────────────────────────────────────────────────────────────
#ifdef HAVE_FFMPEG

bool VideoWorker::openRecorder(int w, int h)
{
    int ret = avformat_alloc_output_context2(&m_fmtCtx, nullptr, nullptr,
                                              m_recPath.toUtf8().constData());
    if (ret < 0 || !m_fmtCtx) return false;

    const AVCodec *codec = avcodec_find_encoder_by_name(m_recCodec.toUtf8().constData());
    if (!codec)
        codec = avcodec_find_encoder_by_name("libx264");
    if (!codec) { avformat_free_context(m_fmtCtx); m_fmtCtx = nullptr; return false; }

    m_stream = avformat_new_stream(m_fmtCtx, codec);
    if (!m_stream) { avformat_free_context(m_fmtCtx); m_fmtCtx = nullptr; return false; }

    m_codecCtx = avcodec_alloc_context3(codec);
    m_codecCtx->width      = w;
    m_codecCtx->height     = h;
    m_codecCtx->time_base  = {1, static_cast<int>(m_recFps)};
    m_codecCtx->framerate  = {static_cast<int>(m_recFps), 1};
    m_codecCtx->pix_fmt    = AV_PIX_FMT_YUV420P;
    m_codecCtx->bit_rate   = 4'000'000;

    av_opt_set(m_codecCtx->priv_data, "preset", "fast", 0);
    av_opt_set(m_codecCtx->priv_data, "crf",    "23",   0);

    if (m_fmtCtx->oformat->flags & AVFMT_GLOBALHEADER)
        m_codecCtx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

    ret = avcodec_open2(m_codecCtx, codec, nullptr);
    if (ret < 0) { avcodec_free_context(&m_codecCtx); avformat_free_context(m_fmtCtx); m_fmtCtx = nullptr; return false; }

    avcodec_parameters_from_context(m_stream->codecpar, m_codecCtx);
    m_stream->time_base = m_codecCtx->time_base;

    if (!(m_fmtCtx->oformat->flags & AVFMT_NOFILE)) {
        ret = avio_open(&m_fmtCtx->pb, m_recPath.toUtf8().constData(), AVIO_FLAG_WRITE);
        if (ret < 0) { avcodec_free_context(&m_codecCtx); avformat_free_context(m_fmtCtx); m_fmtCtx = nullptr; return false; }
    }

    // For MP4/MOV containers, use fragmented mode so the file is
    // continuously playable even if the recorder is not cleanly closed.
    AVDictionary *muxOpts = nullptr;
    if (m_fmtCtx->oformat && m_fmtCtx->oformat->name) {
        std::string fmt(m_fmtCtx->oformat->name);
        if (fmt.find("mp4") != std::string::npos ||
            fmt.find("mov") != std::string::npos) {
            av_dict_set(&muxOpts, "movflags",
                        "frag_keyframe+empty_moov+default_base_moof", 0);
        }
    }

    ret = avformat_write_header(m_fmtCtx, &muxOpts);
    av_dict_free(&muxOpts);
    if (ret < 0) { avio_closep(&m_fmtCtx->pb); avcodec_free_context(&m_codecCtx); avformat_free_context(m_fmtCtx); m_fmtCtx = nullptr; return false; }

    m_avFrame = av_frame_alloc();
    m_avFrame->format = AV_PIX_FMT_YUV420P;
    m_avFrame->width  = w;
    m_avFrame->height = h;
    av_frame_get_buffer(m_avFrame, 0);

    m_swsCtx = sws_getContext(w, h, AV_PIX_FMT_RGB24,
                               w, h, AV_PIX_FMT_YUV420P,
                               SWS_FAST_BILINEAR, nullptr, nullptr, nullptr);

    m_recOpen = true;
    m_recFrameIndex = 0;
    return true;
}

void VideoWorker::writeRecordingFrame(const QImage &img)
{
    QImage rgb = img.convertToFormat(QImage::Format_RGB888);

    if (!m_recOpen) {
        if (!openRecorder(rgb.width(), rgb.height())) {
            m_recording = false;
            emit recordingError(QStringLiteral("Failed to open recorder"));
            return;
        }
    }

    const uint8_t *srcSlice[1] = { rgb.constBits() };
    int srcStride[1] = { static_cast<int>(rgb.bytesPerLine()) };

    av_frame_make_writable(m_avFrame);
    sws_scale(m_swsCtx, srcSlice, srcStride, 0, rgb.height(),
              m_avFrame->data, m_avFrame->linesize);

    m_avFrame->pts = m_recFrameIndex++;
    encodeAndWrite(m_avFrame);
}

void VideoWorker::encodeAndWrite(AVFrame *frame)
{
    int ret = avcodec_send_frame(m_codecCtx, frame);
    if (ret < 0) return;

    AVPacket *pkt = av_packet_alloc();
    while (ret >= 0) {
        ret = avcodec_receive_packet(m_codecCtx, pkt);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
        if (ret < 0) break;
        av_packet_rescale_ts(pkt, m_codecCtx->time_base, m_stream->time_base);
        pkt->stream_index = m_stream->index;
        av_interleaved_write_frame(m_fmtCtx, pkt);
        av_packet_unref(pkt);
    }
    av_packet_free(&pkt);

    // Flush I/O buffer so data is continuously written to disk
    if (m_fmtCtx && m_fmtCtx->pb)
        avio_flush(m_fmtCtx->pb);
}

void VideoWorker::closeRecorder()
{
    if (!m_recOpen) return;

    // Flush
    encodeAndWrite(nullptr);
    av_write_trailer(m_fmtCtx);

    if (m_swsCtx)   { sws_freeContext(m_swsCtx);  m_swsCtx  = nullptr; }
    if (m_avFrame)   { av_frame_free(&m_avFrame);              }
    if (m_codecCtx)  { avcodec_free_context(&m_codecCtx);      }
    if (m_fmtCtx) {
        if (!(m_fmtCtx->oformat->flags & AVFMT_NOFILE))
            avio_closep(&m_fmtCtx->pb);
        avformat_free_context(m_fmtCtx);
        m_fmtCtx = nullptr;
    }
    m_recOpen = false;
}

#endif // HAVE_FFMPEG
