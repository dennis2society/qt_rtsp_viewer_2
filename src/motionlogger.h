#pragma once

#include <QFile>
#include <QPointF>
#include <QRectF>
#include <QString>
#include <QTextStream>
#include <QVector>

/// Lightweight CSV logger for motion data. One instance per VideoWorker.
/// Only writes when explicitly opened (during recording).
class MotionLogger
{
public:
    MotionLogger() = default;
    ~MotionLogger();

    /// Open the CSV file for writing. Returns true on success.
    bool open(const QString &csvPath);

    /// Close the file and flush.
    void close();

    bool isOpen() const
    {
        return m_file.isOpen();
    }

    /// Bounding-box blob from motion detection overlay.
    struct DetectionBlob {
        QRect bbox; // bounding rect (x,y,w,h)
        QPointF cog; // centre-of-gravity (pixel-count weighted)
        double magnitude = 0.0; // active-pixel density [0..1]
        int trackId = -1; // assigned by tracker (-1 = untracked)
    };

    /// Flow blob from motion vectors overlay.
    struct VectorBlob {
        QRect bbox; // bounding rect of the flow region
        QPointF cog; // centre-of-gravity (weighted by magnitude)
        double magnitude; // average magnitude in the blob region
        int trackId = -1; // assigned by tracker (-1 = untracked)
    };

    /// Log one frame's worth of motion data.
    /// Multiple detection blobs and vector blobs may be present per frame.
    void logFrame(qint64 frameNumber, double timestampSec, double fps, const QVector<DetectionBlob> &detections, const QVector<VectorBlob> &vectors);

private:
    QFile m_file;
    QTextStream m_stream;
};
