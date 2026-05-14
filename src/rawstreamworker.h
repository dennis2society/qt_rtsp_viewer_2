#pragma once

#include <QObject>
#include <QProcess>
#include <QString>

/// Records an RTSP stream by invoking the system \c ffmpeg binary as a child
/// process with <tt>-c copy</tt> (no re-encoding).  No FFmpeg development
/// libraries are required at compile time – only the \c ffmpeg executable at
/// runtime (in PATH or next to the application binary).  This makes Windows
/// builds trivial: ship \c ffmpeg.exe alongside the application.
///
/// Lives on the UI thread; QProcess is event-loop driven so no extra thread
/// is needed.
class RawStreamWorker : public QObject
{
    Q_OBJECT

public:
    explicit RawStreamWorker(QObject *parent = nullptr);
    ~RawStreamWorker() override;

    /// Start copying \a rtspUrl → \a outputPath.  Call from any thread that
    /// has an event loop (UI thread is fine).
    void startCopy(const QString &rtspUrl, const QString &outputPath);

    /// Ask ffmpeg to quit gracefully; force-kills after 4 s if it refuses.
    void requestStop();

    bool isRunning() const;

signals:
    void copyStarted();
    void copyFinished(const QString &path);
    void copyError(const QString &msg);

private:
    void cleanup();

    /// Locate the ffmpeg binary: checks next to the application first, then
    /// falls back to PATH.
    static QString findFfmpegBinary();

    QProcess *m_process = nullptr;
    QString m_outputPath;
    bool m_stopping = false;
};
