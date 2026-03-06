#pragma once

#include <QImage>
#include <QFile>
#include <deque>

#include <opencv2/core.hpp>
#include <opencv2/objdetect.hpp>

/// Pure image-processing utility – no QObject, no threading awareness.
/// One instance per VideoWorker (i.e. per stream).
class OpenCVProcessor {
public:
    OpenCVProcessor();

    // ── image adjustments ───────────────────────────────────────────
    QImage applyGaussBlur(const QImage &src, int amount);
    QImage applyBrightnessContrast(const QImage &src, int brightness, int contrast);
    QImage applyColorTemperature(const QImage &src, int temperature);

    // ── detection overlays ──────────────────────────────────────────
    QImage applyMotionDetectionOverlay(const QImage &drawTarget,
                                       const QImage &cleanCurrent,
                                       const QImage &cleanPrevious,
                                       int sensitivity);

    QImage applyMotionVectorsOverlay(const QImage &drawTarget,
                                      const QImage &cleanCurrent,
                                      const QImage &cleanPrevious);

    QImage applyFaceDetection(const QImage &drawTarget,
                               const QImage &cleanCurrent);

    // ── motion analysis ─────────────────────────────────────────────
    double computeMotionLevel(const QImage &cleanCurrent,
                              const QImage &cleanPrevious,
                              int sensitivity);

    QImage applyGridMotionOverlay(const QImage &drawTarget,
                                   const QImage &cleanCurrent,
                                   const QImage &cleanPrevious,
                                   int sensitivity);

    QImage applyMotionGraphOverlay(const QImage &drawTarget,
                                    double motionLevel);

private:
    // helpers
    cv::Mat qImageToMat(const QImage &img);
    QImage  matToQImage(const cv::Mat &mat, QImage::Format fmt);

    // reusable buffers
    cv::Mat m_srcMat, m_work1, m_work2, m_work3, m_rgbMat;
    QImage  m_resultImage;

    // face detection
    cv::CascadeClassifier m_faceCascade;
    bool                  m_faceCascadeLoaded = false;

    // motion graph history
    std::deque<double>         m_graphHistory;       // aggregate level per frame
    static constexpr int       kGraphHistoryLen = 120;

    // grid motion (6 cols × 4 rows)
    static constexpr int kGridCols = 6;
    static constexpr int kGridRows = 4;
    std::vector<double>  m_cellLevels;               // smoothed EMA per cell
    std::deque<double>   m_medianHistory;             // for spike rejection
    static constexpr int kMedianWindowLen = 30;

    // per-cell history for the stacked bar chart
    std::vector<std::deque<double>> m_cellHistory;
};
