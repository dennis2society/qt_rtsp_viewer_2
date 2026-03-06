#pragma once

#include <QObject>
#include <QImage>
#include <QDateTime>
#include <QVideoFrame>

#ifdef HAVE_FFMPEG
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
}
#endif

class OpenCVProcessor;

/// Lives on a dedicated QThread.  Receives raw QVideoFrames, applies the
/// full effects pipeline, optionally records, and emits composited QImages.
class VideoWorker : public QObject {
    Q_OBJECT

public:
    explicit VideoWorker(int streamId, QObject *parent = nullptr);
    ~VideoWorker() override;

public slots:
    // ── frame pipeline ──────────────────────────────────────────────
    void processFrame(const QVideoFrame &frame);
    void setPaused(bool p);
    void setStreamActive(bool active);

    // ── recording ───────────────────────────────────────────────────
    void startRecording(const QString &path, const QString &codec, double fps);
    void stopRecording();

signals:
    void frameReady(const QImage &image);
    void recordingStarted();
    void recordingFinished(const QString &path);
    void recordingError(const QString &msg);
    void autoRecordingStarted(const QString &path);
    void autoRecordingStopped(const QString &path);

private:
    void paintFpsOverlay(QImage &img);
    void handleAutoRecord(double motionLevel);

#ifdef HAVE_FFMPEG
    bool openRecorder(int w, int h);
    void writeRecordingFrame(const QImage &img);
    void encodeAndWrite(AVFrame *frame);
    void closeRecorder();
#endif

    int                  m_streamId;
    OpenCVProcessor     *m_processor = nullptr;

    bool                 m_paused       = false;
    bool                 m_streamActive = false;

    // FPS counter
    int                  m_frameCount = 0;
    double               m_fps        = 0.0;
    QDateTime            m_fpsTimer;

    // Clean-frame bookkeeping (for detection algorithms)
    QImage               m_cleanPrevious;
    QImage               m_frozenFrame;

    // ── Recording state ─────────────────────────────────────────────
    bool    m_recording      = false;
    bool    m_recOpen        = false;
    QString m_recPath;
    QString m_recCodec;
    double  m_recFps         = 25.0;
    int64_t m_recFrameIndex  = 0;

    // ── Auto-record ─────────────────────────────────────────────────
    bool      m_autoRecording    = false;
    QDateTime m_autoRecStartTime;
    qint64    m_lastMotionAboveMs = 0;

#ifdef HAVE_FFMPEG
    AVFormatContext *m_fmtCtx   = nullptr;
    AVCodecContext  *m_codecCtx = nullptr;
    AVStream        *m_stream   = nullptr;
    AVFrame         *m_avFrame  = nullptr;
    SwsContext      *m_swsCtx   = nullptr;
#endif
};
