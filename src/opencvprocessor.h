#pragma once

#include <QFile>
#include <QImage>
#include <QPointF>
#include <QRect>
#include <QVector>
#include <deque>

#include <opencv2/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/objdetect.hpp>

#include "motionlogger.h"

/// Pure image-processing utility – no QObject, no threading awareness.
/// One instance per VideoWorker (i.e. per stream).
class OpenCVProcessor
{
public:
    OpenCVProcessor();

    void reset(); // call when stream source changes

    // ── conversion ──────────────────────────────────────────────────
    cv::Mat qImageToBGR(const QImage &img);
    QImage bgrToQImage(const cv::Mat &bgr);

    // ── image adjustments (in-place on BGR cv::Mat – CPU path) ─────
    void applyGaussBlur(cv::Mat &bgr, int amount);
    void applyBrightnessContrast(cv::Mat &bgr, int brightness, int contrast);
    void applyColorTemperature(cv::Mat &bgr, int temperature);

    // ── image adjustments (in-place on BGR cv::UMat – GPU path) ──
    void applyGaussBlur(cv::UMat &bgr, int amount);
    void applyBrightnessContrast(cv::UMat &bgr, int brightness, int contrast);
    void applyColorTemperature(cv::UMat &bgr, int temperature);

    bool haveOpenCL() const
    {
        return m_haveOpenCL;
    }

    // ── spike detection / reference frame ───────────────────────
    /// Push the current gray frame into the history buffer.
    /// Returns true if the frame is usable (not a spike).
    /// After calling, use referenceGray() to get the best comparison frame.
    bool pushGrayFrame(const cv::Mat &grayCur);
    /// The best stable reference frame from recent history.
    /// Empty until at least 2 non-spike frames have been pushed.
    const cv::Mat &referenceGray() const
    {
        return m_referenceGray;
    }
    double decayMotionLevels();

    // ── detection overlays (paint directly on QImage) ───────────────
    void applyMotionDetectionOverlay(QImage &image,
                                     const cv::Mat &grayCur,
                                     const cv::Mat &grayPrev,
                                     int sensitivity,
                                     bool showTraces = false,
                                     int traceDecay = 50,
                                     bool drawOverlay = true,
                                     QVector<MotionLogger::DetectionBlob> *outBlobs = nullptr);
    void applyMotionVectorsOverlay(QImage &image,
                                   const cv::Mat &grayCur,
                                   const cv::Mat &grayPrev,
                                   int sensitivity = 50,
                                   bool showTraces = false,
                                   int traceDecay = 50,
                                   bool drawOverlay = true,
                                   QVector<MotionLogger::VectorBlob> *outBlobs = nullptr);
    void applyFaceDetection(QImage &image, const cv::Mat &bgrClean);

    // ── motion analysis ─────────────────────────────────────────────
    double computeMotionLevel(const cv::Mat &grayCur, const cv::Mat &grayPrev, int sensitivity);
    void applyGridMotionOverlay(QImage &image, int sensitivity);
    void applyMotionGraphOverlay(QImage &image, double motionLevel);

private:
    // reusable buffers
    cv::Mat m_srcMat, m_work1, m_work2, m_work3, m_rgbMat;

    // face detection
    cv::CascadeClassifier m_faceCascade;
    bool m_faceCascadeLoaded = false;

    // motion graph history
    std::deque<double> m_graphHistory; // aggregate level per frame
    static constexpr int kGraphHistoryLen = 120;

    // grid motion (6 cols × 4 rows)
    static constexpr int kGridCols = 6;
    static constexpr int kGridRows = 4;
    std::vector<double> m_cellLevels; // smoothed EMA per cell

    // per-cell history for the stacked bar chart
    std::vector<std::deque<double>> m_cellHistory;

    // Codec-artifact (I-frame) spike rejection — frame history
    struct FrameRecord {
        cv::Mat gray;
        double diff = 0.0; // diff vs. previous record
        bool spike = false;
    };
    std::deque<FrameRecord> m_frameHistory;
    static constexpr int kFrameHistoryLen = 5;
    cv::Mat m_referenceGray; // best stable reference
    double m_globalDiffEma = -1.0;
    static constexpr double kGlobalDiffEmaAlpha = 0.15;
    static constexpr double kSpikeMultiplier = 3.5;

    // OpenCL availability (checked once at construction)
    bool m_haveOpenCL = false;

    // Motion traces (decaying centroid trails)
    struct TracePoint {
        QPointF pos;
        double opacity;
    };
    std::deque<TracePoint> m_motionTraces; // for motion vectors
    std::deque<TracePoint> m_detectionTraces; // for motion detection
    static constexpr int kMaxTracePoints = 200;
};
