#include "motiontracker.h"

#include <algorithm>
#include <cmath>
#include <limits>

// ─────────────────────────────────────────────────────────────────────────────
void MotionTracker::reset()
{
    m_tracks.clear();
    m_nextId = 1;
}

// ─────────────────────────────────────────────────────────────────────────────
// Kalman filter helpers
// ─────────────────────────────────────────────────────────────────────────────
MotionTracker::Track MotionTracker::createTrack(int id, const QRect &bbox)
{
    Track t;
    t.id = id;

    // State: [cx, cy, w, h, dcx, dcy, dw, dh]  (8-dim)
    // Measurement: [cx, cy, w, h]               (4-dim)
    t.kf.init(8, 4, 0, CV_32F);

    // Transition matrix (constant-velocity model)
    cv::setIdentity(t.kf.transitionMatrix);
    // velocity components feed into position
    t.kf.transitionMatrix.at<float>(0, 4) = 1.0f; // cx += dcx
    t.kf.transitionMatrix.at<float>(1, 5) = 1.0f; // cy += dcy
    t.kf.transitionMatrix.at<float>(2, 6) = 1.0f; // w  += dw
    t.kf.transitionMatrix.at<float>(3, 7) = 1.0f; // h  += dh

    // Measurement matrix (observe cx, cy, w, h)
    t.kf.measurementMatrix = cv::Mat::zeros(4, 8, CV_32F);
    t.kf.measurementMatrix.at<float>(0, 0) = 1.0f;
    t.kf.measurementMatrix.at<float>(1, 1) = 1.0f;
    t.kf.measurementMatrix.at<float>(2, 2) = 1.0f;
    t.kf.measurementMatrix.at<float>(3, 3) = 1.0f;

    // Process noise — moderate, allows tracking through brief jumps
    cv::setIdentity(t.kf.processNoiseCov, cv::Scalar(1e-1));
    // Measurement noise — moderate, trusts measurements reasonably
    cv::setIdentity(t.kf.measurementNoiseCov, cv::Scalar(5e-1));
    // Initial error covariance
    cv::setIdentity(t.kf.errorCovPost, cv::Scalar(1.0));

    // Initial state
    float cx = bbox.x() + bbox.width() * 0.5f;
    float cy = bbox.y() + bbox.height() * 0.5f;
    t.kf.statePost.at<float>(0) = cx;
    t.kf.statePost.at<float>(1) = cy;
    t.kf.statePost.at<float>(2) = static_cast<float>(bbox.width());
    t.kf.statePost.at<float>(3) = static_cast<float>(bbox.height());
    // velocities start at zero
    t.kf.statePost.at<float>(4) = 0.0f;
    t.kf.statePost.at<float>(5) = 0.0f;
    t.kf.statePost.at<float>(6) = 0.0f;
    t.kf.statePost.at<float>(7) = 0.0f;

    return t;
}

QRect MotionTracker::Track::predictedBox() const
{
    float cx = kf.statePost.at<float>(0);
    float cy = kf.statePost.at<float>(1);
    float w = std::max(kf.statePost.at<float>(2), 1.0f);
    float h = std::max(kf.statePost.at<float>(3), 1.0f);
    return QRect(static_cast<int>(cx - w * 0.5f), static_cast<int>(cy - h * 0.5f), static_cast<int>(w), static_cast<int>(h));
}

QRect MotionTracker::Track::predict()
{
    kf.predict();
    float cx = kf.statePre.at<float>(0);
    float cy = kf.statePre.at<float>(1);
    float w = std::max(kf.statePre.at<float>(2), 1.0f);
    float h = std::max(kf.statePre.at<float>(3), 1.0f);
    return QRect(static_cast<int>(cx - w * 0.5f), static_cast<int>(cy - h * 0.5f), static_cast<int>(w), static_cast<int>(h));
}

void MotionTracker::Track::correct(const QRect &meas)
{
    cv::Mat m(4, 1, CV_32F);
    m.at<float>(0) = meas.x() + meas.width() * 0.5f;
    m.at<float>(1) = meas.y() + meas.height() * 0.5f;
    m.at<float>(2) = static_cast<float>(meas.width());
    m.at<float>(3) = static_cast<float>(meas.height());
    kf.correct(m);
}

// ─────────────────────────────────────────────────────────────────────────────
double MotionTracker::iou(const QRect &a, const QRect &b)
{
    int x1 = std::max(a.left(), b.left());
    int y1 = std::max(a.top(), b.top());
    int x2 = std::min(a.right(), b.right());
    int y2 = std::min(a.bottom(), b.bottom());
    if (x2 < x1 || y2 < y1)
        return 0.0;
    double inter = static_cast<double>(x2 - x1 + 1) * (y2 - y1 + 1);
    double areaA = static_cast<double>(a.width()) * a.height();
    double areaB = static_cast<double>(b.width()) * b.height();
    return inter / (areaA + areaB - inter);
}

// ─────────────────────────────────────────────────────────────────────────────
// Main update: predict → match → correct → create/remove
// ─────────────────────────────────────────────────────────────────────────────
QVector<MotionTracker::TrackedObject> MotionTracker::update(const QVector<QRect> &detections)
{
    // 1. Predict all existing tracks
    std::vector<QRect> predicted;
    predicted.reserve(m_tracks.size());
    for (auto &t : m_tracks)
        predicted.push_back(t.predict());

    // 2. Compute IoU cost matrix
    int nTracks = static_cast<int>(m_tracks.size());
    int nDets = detections.size();

    // Greedy assignment by best IoU (simple and fast for small counts)
    std::vector<bool> trackMatched(nTracks, false);
    std::vector<bool> detMatched(nDets, false);
    std::vector<std::pair<int, int>> matches; // (trackIdx, detIdx)

    // Build all valid pairs sorted by IoU descending
    struct Pair {
        int t;
        int d;
        double score;
    };
    std::vector<Pair> pairs;
    for (int ti = 0; ti < nTracks; ++ti)
        for (int di = 0; di < nDets; ++di) {
            double s = iou(predicted[ti], detections[di]);
            if (s >= kIoUThreshold)
                pairs.push_back({ti, di, s});
        }
    std::sort(pairs.begin(), pairs.end(), [](const Pair &a, const Pair &b) {
        return a.score > b.score;
    });

    for (const auto &p : pairs) {
        if (trackMatched[p.t] || detMatched[p.d])
            continue;
        trackMatched[p.t] = true;
        detMatched[p.d] = true;
        matches.push_back({p.t, p.d});
    }

    // 3. Correct matched tracks
    for (const auto &[ti, di] : matches) {
        m_tracks[ti].correct(detections[di]);
        m_tracks[ti].framesLost = 0;
        ++m_tracks[ti].age;
    }

    // 4. Mark unmatched tracks as lost
    for (int ti = 0; ti < nTracks; ++ti) {
        if (!trackMatched[ti]) {
            ++m_tracks[ti].framesLost;
            ++m_tracks[ti].age;
        }
    }

    // 5. Create new tracks for unmatched detections
    for (int di = 0; di < nDets; ++di) {
        if (!detMatched[di]) {
            m_tracks.push_back(createTrack(m_nextId++, detections[di]));
        }
    }

    // 6. Remove dead tracks
    m_tracks.erase(std::remove_if(m_tracks.begin(),
                                  m_tracks.end(),
                                  [](const Track &t) {
                                      return t.framesLost > kMaxLostFrames;
                                  }),
                   m_tracks.end());

    // 7. Build output
    QVector<TrackedObject> result;
    result.reserve(static_cast<int>(m_tracks.size()));
    for (const auto &t : m_tracks) {
        QRect box = t.predictedBox();
        QPointF cog(box.x() + box.width() * 0.5, box.y() + box.height() * 0.5);
        result.append({t.id, box, cog, t.age, t.framesLost});
    }
    return result;
}
