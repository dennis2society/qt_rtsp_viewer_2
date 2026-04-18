#pragma once

#include <QImage>
#include <QMutex>
#include <QObject>
#include <QQueue>
#include <QWaitCondition>
#include <atomic>

#ifdef HAVE_FFMPEG
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
}
#endif

/// Runs on its own QThread.  Receives QImages from VideoWorker via a lock-free
/// queue and encodes / muxes them to disk without blocking the video pipeline.
class RecordingWorker : public QObject
{
    Q_OBJECT

public:
    explicit RecordingWorker(QObject *parent = nullptr);
    ~RecordingWorker() override;

    /// Ask the encoding loop to exit ASAP (checked between frames).
    void requestInterrupt()
    {
        m_interrupt.store(true, std::memory_order_relaxed);
    }

public slots:
    /// Append an image to the encode queue (called cross-thread from VideoWorker).
    void enqueueFrame(const QImage &image);

    /// Prepare for recording; actual file is opened on first frame.
    void startRecording(const QString &path, const QString &codec, double fps);

    /// Signal stop: drain remaining queue (with timeout), finalize file.
    void stopRecording();

    /// Called periodically by a QTimer on *this* thread to drain the queue.
    void processQueue();

signals:
    void recordingStarted();
    void recordingFinished(const QString &path);
    void recordingError(const QString &msg);

private:
#ifdef HAVE_FFMPEG
    bool openRecorder(int w, int h);
    void writeFrame(const QImage &img);
    void encodeAndWrite(AVFrame *frame);
    void closeRecorder();
#endif

    // -- queue --------------------------------------------------------
    static constexpr int kMaxQueueSize = 60; // cap memory (~60 frames)

    QMutex m_queueMutex;
    QQueue<QImage> m_queue;

    // -- recording state ---------------------------------------------
    std::atomic<bool> m_interrupt{false};
    bool m_recording = false;
    bool m_recOpen = false;
    QString m_recPath;
    QString m_recCodec;
    double m_recFps = 25.0;
    int64_t m_recFrameIndex = 0;
    int m_flushCounter = 0;

#ifdef HAVE_FFMPEG
    AVFormatContext *m_fmtCtx = nullptr;
    AVCodecContext *m_codecCtx = nullptr;
    AVStream *m_stream = nullptr;
    AVFrame *m_avFrame = nullptr;
    SwsContext *m_swsCtx = nullptr;
#endif
};
