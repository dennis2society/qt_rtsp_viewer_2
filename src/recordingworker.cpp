#include "recordingworker.h"

#include <QTimer>
#include <QDebug>
#include <string>

// ─────────────────────────────────────────────────────────────────────────────
RecordingWorker::RecordingWorker(QObject *parent)
    : QObject(parent)
{
    // A zero-interval timer drives the encode loop on this thread's event loop.
    // It fires whenever the event loop is idle, so it doesn't starve signals
    // such as startRecording / stopRecording.
    auto *timer = new QTimer(this);
    connect(timer, &QTimer::timeout, this, &RecordingWorker::processQueue);
    timer->start(0);
}

RecordingWorker::~RecordingWorker()
{
#ifdef HAVE_FFMPEG
    if (m_recOpen)
        closeRecorder();
#endif
}

// ─────────────────────────────────────────────────────────────────────────────
// Called cross-thread (from VideoWorker) — must be fast
// ─────────────────────────────────────────────────────────────────────────────
void RecordingWorker::enqueueFrame(const QImage &image)
{
    if (!m_recording) return;

    QMutexLocker lk(&m_queueMutex);
    if (m_queue.size() >= kMaxQueueSize) {
        m_queue.dequeue();   // drop oldest to keep memory bounded
    }
    m_queue.enqueue(image.copy());   // deep copy — source may be transient
}

// ─────────────────────────────────────────────────────────────────────────────
void RecordingWorker::startRecording(const QString &path, const QString &codec,
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
    m_flushCounter  = 0;
    m_interrupt.store(false);

    emit recordingStarted();
#else
    Q_UNUSED(path); Q_UNUSED(codec); Q_UNUSED(fps);
    emit recordingError(QStringLiteral("FFmpeg support not compiled in"));
#endif
}

// ─────────────────────────────────────────────────────────────────────────────
void RecordingWorker::stopRecording()
{
#ifdef HAVE_FFMPEG
    if (!m_recording) return;
    m_recording = false;

    // Drain remaining frames (with a hard cap to avoid hanging forever)
    int remaining = 0;
    {
        QMutexLocker lk(&m_queueMutex);
        remaining = m_queue.size();
    }

    constexpr int kMaxDrainFrames = 120; // don't drain more than ~4 seconds worth
    int drained = 0;

    while (drained < remaining && drained < kMaxDrainFrames &&
           !m_interrupt.load(std::memory_order_relaxed)) {
        QImage img;
        {
            QMutexLocker lk(&m_queueMutex);
            if (m_queue.isEmpty()) break;
            img = m_queue.dequeue();
        }
        writeFrame(img);
        ++drained;
    }

    // Discard anything left after max-drain
    {
        QMutexLocker lk(&m_queueMutex);
        m_queue.clear();
    }

    QString path = m_recPath;
    closeRecorder();
    emit recordingFinished(path);
#endif
}

// ─────────────────────────────────────────────────────────────────────────────
// Timer-driven encode loop – runs on the recorder thread
// ─────────────────────────────────────────────────────────────────────────────
void RecordingWorker::processQueue()
{
    if (!m_recording) return;
    if (m_interrupt.load(std::memory_order_relaxed)) return;

    // Process up to 4 frames per timer tick to keep up without starving the
    // event loop (so that stopRecording / startRecording signals get through).
    for (int i = 0; i < 4; ++i) {
        QImage img;
        {
            QMutexLocker lk(&m_queueMutex);
            if (m_queue.isEmpty()) return;
            img = m_queue.dequeue();
        }
        writeFrame(img);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FFmpeg internals
// ─────────────────────────────────────────────────────────────────────────────
#ifdef HAVE_FFMPEG

bool RecordingWorker::openRecorder(int w, int h)
{
    int ret = avformat_alloc_output_context2(&m_fmtCtx, nullptr, nullptr,
                                              m_recPath.toUtf8().constData());
    if (ret < 0 || !m_fmtCtx) return false;

    // ── Codec selection: try HW encoders first, fall back to software ──
    const AVCodec *codec = nullptr;

    // Determine whether the user picked h264 or hevc family
    QString lower = m_recCodec.toLower();
    bool isHevc = lower.contains(QStringLiteral("265")) ||
                  lower.contains(QStringLiteral("hevc"));

    // Try the user-requested codec first
    codec = avcodec_find_encoder_by_name(m_recCodec.toUtf8().constData());

    // If that didn't work, cascade through HW → SW encoders
    if (!codec) {
        const char *candidates[] = {nullptr, nullptr, nullptr, nullptr, nullptr};
        if (isHevc) {
            candidates[0] = "hevc_nvenc";
            candidates[1] = "hevc_vaapi";
            candidates[2] = "hevc_qsv";
            candidates[3] = "libx265";
        } else {
            candidates[0] = "h264_nvenc";
            candidates[1] = "h264_vaapi";
            candidates[2] = "h264_qsv";
            candidates[3] = "libx264";
        }
        for (auto *name : candidates) {
            if (!name) break;
            codec = avcodec_find_encoder_by_name(name);
            if (codec) {
                qDebug() << "[RecordingWorker] Selected encoder:" << name;
                break;
            }
        }
    }

    if (!codec) {
        avformat_free_context(m_fmtCtx);
        m_fmtCtx = nullptr;
        return false;
    }

    qDebug() << "[RecordingWorker] Using encoder:" << codec->name;

    m_stream = avformat_new_stream(m_fmtCtx, codec);
    if (!m_stream) { avformat_free_context(m_fmtCtx); m_fmtCtx = nullptr; return false; }

    m_codecCtx = avcodec_alloc_context3(codec);
    m_codecCtx->width      = w;
    m_codecCtx->height     = h;
    m_codecCtx->time_base  = {1, static_cast<int>(m_recFps)};
    m_codecCtx->framerate  = {static_cast<int>(m_recFps), 1};
    m_codecCtx->pix_fmt    = AV_PIX_FMT_YUV420P;

    // ── Encoder-specific tuning ────────────────────────────────────
    std::string codecName(codec->name);
    bool isNvenc  = codecName.find("nvenc") != std::string::npos;
    bool isVaapi  = codecName.find("vaapi") != std::string::npos;
    bool isQsv    = codecName.find("qsv")   != std::string::npos;

    if (isNvenc) {
        // NVENC: use constant-quality VBR, low-latency preset
        av_opt_set(m_codecCtx->priv_data, "preset",  "p4",  0);  // balanced
        av_opt_set(m_codecCtx->priv_data, "tune",    "ll",  0);  // low-latency
        av_opt_set(m_codecCtx->priv_data, "rc",      "vbr", 0);  // variable bitrate
        av_opt_set(m_codecCtx->priv_data, "cq",      "23",  0);  // quality level
        m_codecCtx->bit_rate     = 0;  // let CQ drive quality
        m_codecCtx->gop_size     = static_cast<int>(m_recFps);    // keyframe every ~1 sec
    } else if (isVaapi) {
        m_codecCtx->bit_rate = 6'000'000;
        m_codecCtx->gop_size = static_cast<int>(m_recFps);
    } else if (isQsv) {
        m_codecCtx->bit_rate = 6'000'000;
        m_codecCtx->gop_size = static_cast<int>(m_recFps);
    } else {
        // Software (libx264 / libx265)
        av_opt_set(m_codecCtx->priv_data, "preset", "fast",     0);
        av_opt_set(m_codecCtx->priv_data, "crf",    "23",       0);
        m_codecCtx->bit_rate = 0;
        m_codecCtx->gop_size = static_cast<int>(m_recFps);
    }

    if (m_fmtCtx->oformat->flags & AVFMT_GLOBALHEADER)
        m_codecCtx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

    ret = avcodec_open2(m_codecCtx, codec, nullptr);
    if (ret < 0) {
        qWarning() << "[RecordingWorker] Failed to open encoder:" << codec->name
                    << " – error" << ret;
        // If a HW encoder failed, try libx264 as last resort
        if (isNvenc || isVaapi || isQsv) {
            avcodec_free_context(&m_codecCtx);
            avformat_free_context(m_fmtCtx);
            m_fmtCtx = nullptr;

            // Retry with software
            m_recCodec = isHevc ? QStringLiteral("libx265")
                                : QStringLiteral("libx264");
            qDebug() << "[RecordingWorker] Falling back to" << m_recCodec;
            return openRecorder(w, h);
        }
        avcodec_free_context(&m_codecCtx);
        avformat_free_context(m_fmtCtx);
        m_fmtCtx = nullptr;
        return false;
    }

    avcodec_parameters_from_context(m_stream->codecpar, m_codecCtx);
    m_stream->time_base = m_codecCtx->time_base;

    if (!(m_fmtCtx->oformat->flags & AVFMT_NOFILE)) {
        ret = avio_open(&m_fmtCtx->pb, m_recPath.toUtf8().constData(),
                        AVIO_FLAG_WRITE);
        if (ret < 0) {
            avcodec_free_context(&m_codecCtx);
            avformat_free_context(m_fmtCtx);
            m_fmtCtx = nullptr;
            return false;
        }
    }

    // For MP4/MOV containers, use fragmented mode for resilience
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
    if (ret < 0) {
        avio_closep(&m_fmtCtx->pb);
        avcodec_free_context(&m_codecCtx);
        avformat_free_context(m_fmtCtx);
        m_fmtCtx = nullptr;
        return false;
    }

    m_avFrame = av_frame_alloc();
    m_avFrame->format = AV_PIX_FMT_YUV420P;
    m_avFrame->width  = w;
    m_avFrame->height = h;
    av_frame_get_buffer(m_avFrame, 0);

    m_swsCtx = sws_getContext(w, h, AV_PIX_FMT_RGB24,
                               w, h, AV_PIX_FMT_YUV420P,
                               SWS_FAST_BILINEAR, nullptr, nullptr, nullptr);

    m_recOpen       = true;
    m_recFrameIndex = 0;
    m_flushCounter  = 0;
    return true;
}

void RecordingWorker::writeFrame(const QImage &img)
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

void RecordingWorker::encodeAndWrite(AVFrame *frame)
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

    // Flush I/O every ~30 frames (~1 second) instead of every frame
    if (frame && ++m_flushCounter >= 30) {
        m_flushCounter = 0;
        if (m_fmtCtx && m_fmtCtx->pb)
            avio_flush(m_fmtCtx->pb);
    }
}

void RecordingWorker::closeRecorder()
{
    if (!m_recOpen) return;

    // Flush encoder (drain buffered frames)
    encodeAndWrite(nullptr);
    av_write_trailer(m_fmtCtx);

    // Final I/O flush
    if (m_fmtCtx && m_fmtCtx->pb)
        avio_flush(m_fmtCtx->pb);

    if (m_swsCtx)  { sws_freeContext(m_swsCtx); m_swsCtx = nullptr; }
    if (m_avFrame)  { av_frame_free(&m_avFrame);                     }
    if (m_codecCtx) { avcodec_free_context(&m_codecCtx);             }
    if (m_fmtCtx) {
        if (!(m_fmtCtx->oformat->flags & AVFMT_NOFILE))
            avio_closep(&m_fmtCtx->pb);
        avformat_free_context(m_fmtCtx);
        m_fmtCtx = nullptr;
    }
    m_recOpen = false;
}

#endif // HAVE_FFMPEG
