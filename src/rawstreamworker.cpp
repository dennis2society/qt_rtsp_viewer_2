#include "rawstreamworker.h"

#include <QCoreApplication>
#include <QDebug>
#include <QFile>
#include <QTimer>

#ifdef HAVE_FFMPEG
#include <QMetaObject>

#include <string>
#include <vector>

extern "C" {
#include <libavformat/avformat.h>
#include <libavutil/error.h>
#include <libavutil/time.h>
}
#endif

#ifndef HAVE_FFMPEG
// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
QString RawStreamWorker::findFfmpegBinary()
{
    // Prefer a binary shipped alongside the application (easy Windows deploy).
    const QString appDir = QCoreApplication::applicationDirPath();
#ifdef Q_OS_WIN
    const QString local = appDir + QStringLiteral("/ffmpeg.exe");
#else
    const QString local = appDir + QStringLiteral("/ffmpeg");
#endif
    if (QFile::exists(local))
        return local;

    // Fall back to whatever is in PATH.
    return QStringLiteral("ffmpeg");
}
#endif // !HAVE_FFMPEG

// ---------------------------------------------------------------------------
RawStreamWorker::RawStreamWorker(QObject *parent)
    : QObject(parent)
{
}

#ifdef HAVE_FFMPEG
RawStreamWorker::~RawStreamWorker()
{
    // Interrupt callback makes blocking network I/O return promptly.
    m_stopFlag = true;
    if (m_thread.joinable())
        m_thread.join();
}
#else
RawStreamWorker::~RawStreamWorker()
{
    // Destructor may run during app shutdown; kill immediately.
    if (m_process && m_process->state() != QProcess::NotRunning) {
        m_process->kill();
        m_process->waitForFinished(2000);
    }
}
#endif

#ifndef HAVE_FFMPEG
// ---------------------------------------------------------------------------
void RawStreamWorker::startCopy(const QString &rtspUrl, const QString &outputPath)
{
    if (m_process) {
        emit copyError(QStringLiteral("Raw copy already in progress"));
        return;
    }

    m_outputPath = outputPath;
    m_stopping = false;

    const QString ffmpeg = findFfmpegBinary();

    // Arguments for stream-copy recording:
    //   -rtsp_transport tcp   – more reliable over Wi-Fi / lossy links
    //   -i <url>              – input (RTSP or any URL)
    //   -c copy               – passthrough: no decode, no re-encode
    //   -y                    – overwrite output if it already exists
    const QStringList args = {QStringLiteral("-rtsp_transport"),
                              QStringLiteral("tcp"),
                              QStringLiteral("-i"),
                              rtspUrl,
                              QStringLiteral("-c"),
                              QStringLiteral("copy"),
                              QStringLiteral("-y"),
                              outputPath};

    m_process = new QProcess(this);
    // Merge stdout + stderr so we can forward ffmpeg's log if needed.
    m_process->setProcessChannelMode(QProcess::MergedChannels);

    connect(m_process, &QProcess::started, this, [this]() {
        qDebug() << "[RawStreamWorker] ffmpeg started";
        emit copyStarted();
    });

    connect(m_process, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this, [this](int exitCode, QProcess::ExitStatus) {
        qDebug() << "[RawStreamWorker] ffmpeg finished, exit:" << exitCode;
        const QString path = m_outputPath;
        cleanup();
        emit copyFinished(path);
    });

    connect(m_process, &QProcess::errorOccurred, this, [this](QProcess::ProcessError err) {
        if (err == QProcess::FailedToStart) {
            cleanup();
            emit copyError(
                QStringLiteral("ffmpeg binary not found. "
                               "Install ffmpeg and ensure it is in your PATH, "
                               "or place ffmpeg next to the application executable."));
        } else if (err == QProcess::Crashed && !m_stopping) {
            // Unexpected crash (m_stopping suppresses deliberate kills).
            cleanup();
            emit copyError(QStringLiteral("ffmpeg process crashed unexpectedly"));
        }
        // QProcess::Killed after requestStop() → finished() will fire → handled there.
    });

    qDebug() << "[RawStreamWorker] Launching:" << ffmpeg << args;
    m_process->start(ffmpeg, args);
}

// ---------------------------------------------------------------------------
void RawStreamWorker::requestStop()
{
    if (!m_process || m_process->state() == QProcess::NotRunning)
        return;

    m_stopping = true;
    qDebug() << "[RawStreamWorker] Requesting graceful stop";

    // Send 'q' to ffmpeg's stdin – it flushes the muxer and writes the
    // container trailer before exiting (important for MP4/MKV integrity).
    m_process->write("q");
    m_process->closeWriteChannel();

    // Force-kill if ffmpeg hasn't exited within 4 seconds.
    QTimer::singleShot(4000, this, [this]() {
        if (m_process && m_process->state() != QProcess::NotRunning) {
            qDebug() << "[RawStreamWorker] Force-killing ffmpeg";
            m_process->kill();
        }
    });
}

// ---------------------------------------------------------------------------
bool RawStreamWorker::isRunning() const
{
    return m_process && m_process->state() != QProcess::NotRunning;
}

// ---------------------------------------------------------------------------
void RawStreamWorker::cleanup()
{
    if (m_process) {
        m_process->disconnect(this);
        m_process->deleteLater();
        m_process = nullptr;
    }
}
#endif // !HAVE_FFMPEG

#ifdef HAVE_FFMPEG
// ---------------------------------------------------------------------------
// libavformat packet-copy implementation
// ---------------------------------------------------------------------------
namespace
{

constexpr int64_t kStallTimeoutUs = 10LL * 1000 * 1000;

struct InterruptState {
    const std::atomic<bool> *stop;
    int64_t lastActivityUs;
};

int interruptCallback(void *opaque)
{
    auto *s = static_cast<InterruptState *>(opaque);
    return s->stop->load() || (av_gettime_relative() - s->lastActivityUs) > kStallTimeoutUs;
}

QString avErrorString(int err)
{
    char buf[AV_ERROR_MAX_STRING_SIZE] = {};
    av_strerror(err, buf, sizeof(buf));
    return QString::fromUtf8(buf);
}

} // namespace

void RawStreamWorker::startCopy(const QString &rtspUrl, const QString &outputPath)
{
    if (m_running) {
        emit copyError(QStringLiteral("Raw copy already in progress"));
        return;
    }
    if (m_thread.joinable())
        m_thread.join();

    m_outputPath = outputPath;
    m_stopFlag = false;
    m_running = true;

    m_thread = std::thread([this, rtspUrl, outputPath]() {
        const QString error = copyStream(rtspUrl, outputPath);
        QMetaObject::invokeMethod(
            this,
            [this, error]() {
                onThreadDone(error);
            },
            Qt::QueuedConnection);
    });
}

void RawStreamWorker::requestStop()
{
    m_stopFlag = true;
}

bool RawStreamWorker::isRunning() const
{
    return m_running;
}

void RawStreamWorker::onThreadDone(const QString &error)
{
    if (m_thread.joinable())
        m_thread.join();
    m_running = false;

    if (error.isEmpty())
        emit copyFinished(m_outputPath);
    else
        emit copyError(error);
}

QString RawStreamWorker::copyStream(const QString &url, const QString &outputPath)
{
    QString error;
    AVFormatContext *in = nullptr;
    AVFormatContext *out = nullptr;
    AVPacket *pkt = av_packet_alloc();
    bool headerWritten = false;
    std::vector<int> streamMap;

    InterruptState irq{&m_stopFlag, av_gettime_relative()};

    do {
        in = avformat_alloc_context();
        if (!in || !pkt) {
            error = QStringLiteral("Out of memory");
            break;
        }
        in->interrupt_callback.callback = interruptCallback;
        in->interrupt_callback.opaque = &irq;

        AVDictionary *inOpts = nullptr;
        av_dict_set(&inOpts, "rtsp_transport", "tcp", 0); // more reliable over Wi-Fi / lossy links
        int ret = avformat_open_input(&in, url.toUtf8().constData(), nullptr, &inOpts);
        av_dict_free(&inOpts);
        if (ret < 0) {
            error = QStringLiteral("Cannot open stream: %1").arg(avErrorString(ret));
            break;
        }

        irq.lastActivityUs = av_gettime_relative();
        ret = avformat_find_stream_info(in, nullptr);
        if (ret < 0) {
            error = QStringLiteral("Cannot read stream info: %1").arg(avErrorString(ret));
            break;
        }

        const std::string path = outputPath.toUtf8().toStdString();
        ret = avformat_alloc_output_context2(&out, nullptr, nullptr, path.c_str());
        if (ret < 0 || !out) {
            error = QStringLiteral("Cannot create output container: %1").arg(avErrorString(ret));
            break;
        }

        streamMap.assign(in->nb_streams, -1);
        int outIdx = 0;
        for (unsigned i = 0; i < in->nb_streams; ++i) {
            const AVCodecParameters *par = in->streams[i]->codecpar;
            if (par->codec_type != AVMEDIA_TYPE_VIDEO && par->codec_type != AVMEDIA_TYPE_AUDIO)
                continue;
            AVStream *st = avformat_new_stream(out, nullptr);
            if (!st || avcodec_parameters_copy(st->codecpar, par) < 0) {
                error = QStringLiteral("Cannot set up output stream");
                break;
            }
            st->codecpar->codec_tag = 0;
            streamMap[i] = outIdx++;
        }
        if (!error.isEmpty())
            break;
        if (outIdx == 0) {
            error = QStringLiteral("Stream has no audio/video tracks");
            break;
        }

        if (!(out->oformat->flags & AVFMT_NOFILE)) {
            ret = avio_open(&out->pb, path.c_str(), AVIO_FLAG_WRITE);
            if (ret < 0) {
                error = QStringLiteral("Cannot open output file: %1").arg(avErrorString(ret));
                break;
            }
        }

        // Fragmented MP4 stays playable if the app dies mid-recording.
        AVDictionary *muxOpts = nullptr;
        const std::string fmt = out->oformat->name ? out->oformat->name : "";
        if (fmt.find("mp4") != std::string::npos || fmt.find("mov") != std::string::npos)
            av_dict_set(&muxOpts, "movflags", "frag_keyframe+empty_moov+default_base_moof", 0);
        ret = avformat_write_header(out, &muxOpts);
        av_dict_free(&muxOpts);
        if (ret < 0) {
            error = QStringLiteral("Cannot write header (codec not supported by this container?): %1").arg(avErrorString(ret));
            break;
        }
        headerWritten = true;

        QMetaObject::invokeMethod(
            this,
            [this]() {
                emit copyStarted();
            },
            Qt::QueuedConnection);

        while (!m_stopFlag) {
            irq.lastActivityUs = av_gettime_relative();
            ret = av_read_frame(in, pkt);
            if (ret < 0) {
                if (!m_stopFlag && ret != AVERROR_EOF)
                    error = QStringLiteral("Stream interrupted: %1").arg(avErrorString(ret));
                break;
            }

            const int outStream = pkt->stream_index < static_cast<int>(streamMap.size()) ? streamMap[pkt->stream_index] : -1;
            if (outStream < 0) {
                av_packet_unref(pkt);
                continue;
            }

            av_packet_rescale_ts(pkt, in->streams[pkt->stream_index]->time_base, out->streams[outStream]->time_base);
            pkt->stream_index = outStream;
            pkt->pos = -1;
            ret = av_interleaved_write_frame(out, pkt);
            av_packet_unref(pkt);
            if (ret < 0) {
                error = QStringLiteral("Write failed: %1").arg(avErrorString(ret));
                break;
            }
        }
    } while (false);

    // Always finalise the container so the partial file stays playable.
    if (out) {
        if (headerWritten)
            av_write_trailer(out);
        if (!(out->oformat->flags & AVFMT_NOFILE))
            avio_closep(&out->pb);
        avformat_free_context(out);
    }
    if (in)
        avformat_close_input(&in);
    av_packet_free(&pkt);

    return error;
}
#endif // HAVE_FFMPEG
