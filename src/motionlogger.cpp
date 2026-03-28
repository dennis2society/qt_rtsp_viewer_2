#include "motionlogger.h"

#include <QDateTime>

MotionLogger::~MotionLogger()
{
    close();
}

bool MotionLogger::open(const QString &csvPath)
{
    close();
    m_file.setFileName(csvPath);
    if (!m_file.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text))
        return false;
    m_stream.setDevice(&m_file);

    // Write CSV header
    m_stream << "frame,timestamp_sec,unix_timestamp_ms,fps,type,track_id,"
                "bbox_x,bbox_y,bbox_w,bbox_h,cog_x,cog_y,magnitude\n";
    m_stream.flush();
    return true;
}

void MotionLogger::close()
{
    if (m_file.isOpen()) {
        m_stream.flush();
        m_file.close();
    }
}

void MotionLogger::logFrame(qint64 frameNumber, double timestampSec, double fps, const QVector<DetectionBlob> &detections, const QVector<VectorBlob> &vectors)
{
    if (!m_file.isOpen())
        return;

    const qint64 unixMs = QDateTime::currentMSecsSinceEpoch();

    for (const auto &d : detections) {
        m_stream << frameNumber << ',' << QString::number(timestampSec, 'f', 4) << ',' << unixMs << ',' << QString::number(fps, 'f', 1) << ',' << "detection"
                 << ',' << d.trackId << ',' << d.bbox.x() << ',' << d.bbox.y() << ',' << d.bbox.width() << ',' << d.bbox.height() << ','
                 << ",," // no CoG or magnitude for detection
                 << '\n';
    }

    for (const auto &v : vectors) {
        m_stream << frameNumber << ',' << QString::number(timestampSec, 'f', 4) << ',' << unixMs << ',' << QString::number(fps, 'f', 1) << ',' << "vector_blob"
                 << ',' << v.trackId << ',' << v.bbox.x() << ',' << v.bbox.y() << ',' << v.bbox.width() << ',' << v.bbox.height() << ','
                 << QString::number(v.cog.x(), 'f', 1) << ',' << QString::number(v.cog.y(), 'f', 1) << ',' << QString::number(v.magnitude, 'f', 3) << '\n';
    }

    // Flush periodically (every call — recording is not performance-critical for CSV)
    m_stream.flush();
}
