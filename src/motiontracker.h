#pragma once

#include <QPointF>
#include <QRect>
#include <QVector>
#include <deque>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>

/// Kalman-filter-based multi-object tracker that assigns persistent IDs
/// to detection blobs and flow-vector blobs across frames.
/// Smooths bounding boxes and provides centre-of-gravity tracking.
class MotionTracker
{
public:
    MotionTracker() = default;

    /// Reset all tracks (e.g. on stream change).
    void reset();

    /// A single tracked object.
    struct TrackedObject {
        int id; // persistent track ID
        QRect smoothedBox; // Kalman-smoothed bounding box
        QPointF cog; // smoothed centre-of-gravity
        int age; // frames since first seen
        int framesLost; // consecutive frames without a match
    };

    /// Update the tracker with new detections for this frame.
    /// Returns the current tracked objects (matched + predicted).
    QVector<TrackedObject> update(const QVector<QRect> &detections);

    /// Maximum frames a track can be lost before removal.
    static constexpr int kMaxLostFrames = 15;
    /// IoU threshold for matching detections to tracks.
    static constexpr double kIoUThreshold = 0.15;

private:
    struct Track {
        int id;
        cv::KalmanFilter kf; // state: [cx, cy, w, h, dcx, dcy, dw, dh]
        int age = 0;
        int framesLost = 0;

        QRect predictedBox() const;
        void correct(const QRect &measurement);
        QRect predict();
    };

    static Track createTrack(int id, const QRect &bbox);
    static double iou(const QRect &a, const QRect &b);

    std::vector<Track> m_tracks;
    int m_nextId = 1;
};
