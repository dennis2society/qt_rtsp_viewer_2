#pragma once

#include <QFile>
#include <QImage>
#include <deque>

#include <opencv2/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/objdetect.hpp>

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

    // ── image adjustments (in-place on BGR cv::Mat) ─────────────────
    void applyGaussBlur(cv::Mat &bgr, int amount);
    void applyBrightnessContrast(cv::Mat &bgr, int brightness, int contrast);
    void applyColorTemperature(cv::Mat &bgr, int temperature);

    // ── spike detection ─────────────────────────────────────────────
    bool isSpikeFrame(const cv::Mat &grayCur, const cv::Mat &grayPrev);
    double decayMotionLevels();

    // ── detection overlays (paint directly on QImage) ───────────────
    void applyMotionDetectionOverlay(QImage &image, const cv::Mat &grayCur, const cv::Mat &grayPrev, int sensitivity);
    void applyMotionVectorsOverlay(QImage &image, const cv::Mat &grayCur, const cv::Mat &grayPrev);
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

    // Codec-artifact (I-frame) spike rejection
    double m_globalDiffEma = -1.0;
    static constexpr double kGlobalDiffEmaAlpha = 0.15;
    static constexpr double kSpikeMultiplier = 4.0;

    // OpenCL availability (checked once at construction)
    bool m_haveOpenCL = false;
};
