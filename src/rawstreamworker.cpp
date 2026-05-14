#include "rawstreamworker.h"

#include <QCoreApplication>
#include <QDebug>
#include <QFile>
#include <QTimer>

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

// ---------------------------------------------------------------------------
RawStreamWorker::RawStreamWorker(QObject *parent)
    : QObject(parent)
{
}

RawStreamWorker::~RawStreamWorker()
{
    // Destructor may run during app shutdown; kill immediately.
    if (m_process && m_process->state() != QProcess::NotRunning) {
        m_process->kill();
        m_process->waitForFinished(2000);
    }
}

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
