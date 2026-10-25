#pragma once

#include <QObject>
#include <QString>

#ifdef HAVE_FFMPEG
#include <atomic>
#include <thread>
#else
#include <QProcess>
#endif

/// Records an RTSP stream without re-encoding (packet copy).
///
/// With the FFmpeg libraries (HAVE_FFMPEG) the stream is remuxed in-process via
/// libavformat on a worker thread; no \c ffmpeg executable is needed.
/// Without them it falls back to running the \c ffmpeg binary (in PATH or next
/// to the application) with <tt>-c copy</tt>.
///
/// Public API and signals are used from the UI thread only.
class RawStreamWorker : public QObject
{
    Q_OBJECT

public:
    explicit RawStreamWorker(QObject *parent = nullptr);
    ~RawStreamWorker() override;

    /// Start copying \a rtspUrl → \a outputPath.  Call from the UI thread.
    void startCopy(const QString &rtspUrl, const QString &outputPath);

    /// Ask the copy to finish gracefully (the container trailer is written).
    void requestStop();

    bool isRunning() const;

signals:
    void copyStarted();
    void copyFinished(const QString &path);
    void copyError(const QString &msg);

private:
    QString m_outputPath;

#ifdef HAVE_FFMPEG
    /// Runs on the worker thread; returns an empty string on success.
    QString copyStream(const QString &url, const QString &outputPath);
    void onThreadDone(const QString &error);

    std::thread m_thread;
    std::atomic<bool> m_stopFlag{false};
    bool m_running = false;
#else
    void cleanup();

    /// Locate the ffmpeg binary: checks next to the application first, then
    /// falls back to PATH.
    static QString findFfmpegBinary();

    QProcess *m_process = nullptr;
    bool m_stopping = false;
#endif
};
