#include "opencvprocessor.h"

#include <QCoreApplication>
#include <QPainter>

#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>

// ─────────────────────────────────────────────────────────────────────────────
OpenCVProcessor::OpenCVProcessor()
{
    m_cellLevels.resize(kGridCols * kGridRows, 0.0);
    m_cellHistory.resize(kGridRows);
    for (auto &dq : m_cellHistory)
        dq.resize(kGraphHistoryLen, 0.0);

    m_haveOpenCL = cv::ocl::haveOpenCL();
    if (m_haveOpenCL)
        cv::ocl::setUseOpenCL(true);
}

void OpenCVProcessor::reset()
{
    m_globalDiffEma = -1.0;
    m_frameHistory.clear();
    m_referenceGray = cv::Mat{};
    std::fill(m_cellLevels.begin(), m_cellLevels.end(), 0.0);
    for (auto &dq : m_cellHistory)
        std::fill(dq.begin(), dq.end(), 0.0);
    m_graphHistory.clear();
    m_motionTraces.clear();
    m_detectionTraces.clear();
}

// ─────────────────────────────────────────────────────────────────────────────
// Conversion helpers
// ─────────────────────────────────────────────────────────────────────────────
cv::Mat OpenCVProcessor::qImageToBGR(const QImage &img)
{
    QImage tmp = img;
    if (tmp.format() != QImage::Format_RGB888 && tmp.format() != QImage::Format_Grayscale8)
        tmp = tmp.convertToFormat(QImage::Format_RGB888);

    cv::Mat mat(tmp.height(),
                tmp.width(),
                tmp.format() == QImage::Format_Grayscale8 ? CV_8UC1 : CV_8UC3,
                const_cast<uchar *>(tmp.constBits()),
                static_cast<size_t>(tmp.bytesPerLine()));
    mat.copyTo(m_srcMat); // deep copy – detaches from QImage
    if (m_srcMat.channels() == 3)
        cv::cvtColor(m_srcMat, m_srcMat, cv::COLOR_RGB2BGR);
    return m_srcMat;
}

QImage OpenCVProcessor::bgrToQImage(const cv::Mat &bgr)
{
    if (bgr.channels() == 3) {
        cv::cvtColor(bgr, m_rgbMat, cv::COLOR_BGR2RGB);
        return QImage(m_rgbMat.data, m_rgbMat.cols, m_rgbMat.rows, static_cast<int>(m_rgbMat.step), QImage::Format_RGB888).copy();
    }
    return QImage(bgr.data, bgr.cols, bgr.rows, static_cast<int>(bgr.step), QImage::Format_Grayscale8).copy();
}

// ─────────────────────────────────────────────────────────────────────────────
// Blur (in-place on BGR cv::Mat)
// ─────────────────────────────────────────────────────────────────────────────
void OpenCVProcessor::applyGaussBlur(cv::Mat &bgr, int amount)
{
    if (amount <= 0)
        return;
    int ks = amount * 2 + 1;
    double sigma = amount * 0.5;
    cv::GaussianBlur(bgr, m_work1, cv::Size(ks, ks), sigma);
    cv::swap(bgr, m_work1);
}

void OpenCVProcessor::applyGaussBlur(cv::UMat &bgr, int amount)
{
    if (amount <= 0)
        return;
    int ks = amount * 2 + 1;
    double sigma = amount * 0.5;
    cv::UMat tmp;
    cv::GaussianBlur(bgr, tmp, cv::Size(ks, ks), sigma);
    cv::swap(bgr, tmp);
}

// ─────────────────────────────────────────────────────────────────────────────
// Brightness / Contrast (in-place on BGR cv::Mat)
// ─────────────────────────────────────────────────────────────────────────────
void OpenCVProcessor::applyBrightnessContrast(cv::Mat &bgr, int brightness, int contrast)
{
    if (brightness == 0 && contrast == 0)
        return;

    double alpha = 1.0 + contrast / 100.0;
    double beta = brightness;

    cv::Mat lut(1, 256, CV_8U);
    uchar *p = lut.ptr();
    for (int i = 0; i < 256; ++i)
        p[i] = cv::saturate_cast<uchar>(alpha * i + beta);

    std::vector<cv::Mat> chs;
    cv::split(bgr, chs);
    for (auto &ch : chs)
        cv::LUT(ch, lut, ch);
    cv::merge(chs, m_work1);
    cv::swap(bgr, m_work1);
}

void OpenCVProcessor::applyBrightnessContrast(cv::UMat &bgr, int brightness, int contrast)
{
    if (brightness == 0 && contrast == 0)
        return;

    double alpha = 1.0 + contrast / 100.0;
    double beta = brightness;
    bgr.convertTo(bgr, -1, alpha, beta);
}

// ─────────────────────────────────────────────────────────────────────────────
// Colour temperature (in-place on BGR cv::Mat)
// ─────────────────────────────────────────────────────────────────────────────
void OpenCVProcessor::applyColorTemperature(cv::Mat &bgr, int temperature)
{
    if (temperature == 0)
        return;

    double t = temperature / 100.0;
    double rScale = 1.0 - t * 0.30;
    double bScale = 1.0 + t * 0.30;

    cv::Mat lutAll(1, 256, CV_8UC3);
    auto *q = lutAll.ptr<cv::Vec3b>();
    for (int i = 0; i < 256; ++i) {
        q[i][0] = cv::saturate_cast<uchar>(i * bScale); // B
        q[i][1] = cv::saturate_cast<uchar>(i); // G
        q[i][2] = cv::saturate_cast<uchar>(i * rScale); // R
    }
    cv::LUT(bgr, lutAll, m_work1);
    cv::swap(bgr, m_work1);
}

void OpenCVProcessor::applyColorTemperature(cv::UMat &bgr, int temperature)
{
    if (temperature == 0)
        return;

    double t = temperature / 100.0;
    double rScale = 1.0 - t * 0.30;
    double bScale = 1.0 + t * 0.30;

    std::vector<cv::UMat> chs;
    cv::split(bgr, chs);
    chs[0].convertTo(chs[0], -1, bScale, 0); // B
    chs[2].convertTo(chs[2], -1, rScale, 0); // R
    cv::merge(chs, bgr);
}

// ─────────────────────────────────────────────────────────────────────────────
// Spike detection with frame-history reference selection
// ─────────────────────────────────────────────────────────────────────────────
bool OpenCVProcessor::pushGrayFrame(const cv::Mat &grayCur)
{
    FrameRecord rec;
    // Light blur to suppress compression noise before any motion analysis
    cv::GaussianBlur(grayCur, rec.gray, cv::Size(5, 5), 1.0);

    // Compute diff against the most recent frame in history
    if (!m_frameHistory.empty()) {
        const cv::Mat &prev = m_frameHistory.back().gray;
        if (grayCur.size() != prev.size()) {
            // Resolution change — reset history
            m_frameHistory.clear();
            m_globalDiffEma = -1.0;
            m_referenceGray = cv::Mat{};
            rec.diff = 0.0;
            rec.spike = false;
            m_frameHistory.push_back(std::move(rec));
            return false;
        }

        cv::Mat diff;
        cv::absdiff(grayCur, prev, diff);
        double globalDiff = cv::mean(diff)[0] / 255.0;
        rec.diff = globalDiff;

        if (m_globalDiffEma < 0.0) {
            m_globalDiffEma = globalDiff;
        }

        // A frame is a spike if its diff is much larger than the running average
        rec.spike = (globalDiff > kSpikeMultiplier * m_globalDiffEma) && (m_globalDiffEma > 0.002);

        // Also check against the median diff of recent history for extra robustness:
        // if this frame's diff is > 3× the median of non-spike history diffs, flag it
        if (!rec.spike && m_frameHistory.size() >= 3) {
            std::vector<double> recentDiffs;
            for (const auto &fr : m_frameHistory)
                if (!fr.spike)
                    recentDiffs.push_back(fr.diff);
            if (recentDiffs.size() >= 2) {
                std::sort(recentDiffs.begin(), recentDiffs.end());
                double median = recentDiffs[recentDiffs.size() / 2];
                if (median > 0.001 && globalDiff > kSpikeMultiplier * median)
                    rec.spike = true;
            }
        }

        // Only update the EMA with non-spike diffs
        if (!rec.spike)
            m_globalDiffEma = kGlobalDiffEmaAlpha * globalDiff + (1.0 - kGlobalDiffEmaAlpha) * m_globalDiffEma;
    } else {
        rec.diff = 0.0;
        rec.spike = false;
    }

    m_frameHistory.push_back(std::move(rec));
    while (static_cast<int>(m_frameHistory.size()) > kFrameHistoryLen)
        m_frameHistory.pop_front();

    // Select the best reference frame: the most recent non-spike frame
    // that is at least 1 frame old (so we compare against a stable frame)
    m_referenceGray = cv::Mat{};
    for (int i = static_cast<int>(m_frameHistory.size()) - 2; i >= 0; --i) {
        if (!m_frameHistory[static_cast<size_t>(i)].spike) {
            m_referenceGray = m_frameHistory[static_cast<size_t>(i)].gray;
            break;
        }
    }

    // If all history frames are spikes, fall back to the oldest frame
    if (m_referenceGray.empty() && m_frameHistory.size() >= 2)
        m_referenceGray = m_frameHistory.front().gray;

    return !m_frameHistory.back().spike;
}

double OpenCVProcessor::decayMotionLevels()
{
    constexpr double decayEma = 0.85;
    double maxCell = 0.0, sumCell = 0.0;
    for (int i = 0; i < kGridCols * kGridRows; ++i) {
        m_cellLevels[i] *= decayEma;
        maxCell = std::max(maxCell, m_cellLevels[i]);
        sumCell += m_cellLevels[i];
    }
    return std::min(0.6 * maxCell + 0.4 * (sumCell / (kGridCols * kGridRows)), 1.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// Motion detection overlay (contour-based, clustered bounding boxes)
// ─────────────────────────────────────────────────────────────────────────────
void OpenCVProcessor::applyMotionDetectionOverlay(QImage &image,
                                                  const cv::Mat &grayCur,
                                                  const cv::Mat &grayPrev,
                                                  int sensitivity,
                                                  bool showTraces,
                                                  int traceDecay)
{
    int thresh = std::max(1, 100 - sensitivity);

    if (m_haveOpenCL) {
        cv::UMat uCur, uPrev, uDiff, uThresh;
        grayCur.copyTo(uCur);
        grayPrev.copyTo(uPrev);
        cv::absdiff(uCur, uPrev, uDiff);
        cv::threshold(uDiff, uThresh, thresh, 255, cv::THRESH_BINARY);
        uThresh.copyTo(m_work3);
    } else {
        cv::absdiff(grayCur, grayPrev, m_work2);
        cv::threshold(m_work2, m_work3, thresh, 255, cv::THRESH_BINARY);
    }

    // Morphological cleanup: dilate to bridge nearby regions, then erode to remove specks
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::dilate(m_work3, m_work3, kernel, cv::Point(-1, -1), 1);
    cv::erode(m_work3, m_work3, kernel, cv::Point(-1, -1), 1);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(m_work3, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Collect significant bounding boxes
    std::vector<cv::Rect> boxes;
    for (const auto &c : contours) {
        if (cv::contourArea(c) > 1000)
            boxes.push_back(cv::boundingRect(c));
    }

    // Cluster nearby boxes by merging overlapping/touching rectangles
    // Expand each box slightly for proximity merging, then iteratively merge
    if (!boxes.empty()) {
        constexpr int kMergeMargin = 30; // pixels
        bool merged = true;
        while (merged) {
            merged = false;
            for (size_t i = 0; i < boxes.size(); ++i) {
                cv::Rect expanded_i(boxes[i].x - kMergeMargin,
                                    boxes[i].y - kMergeMargin,
                                    boxes[i].width + 2 * kMergeMargin,
                                    boxes[i].height + 2 * kMergeMargin);
                for (size_t j = i + 1; j < boxes.size();) {
                    cv::Rect expanded_j(boxes[j].x - kMergeMargin,
                                        boxes[j].y - kMergeMargin,
                                        boxes[j].width + 2 * kMergeMargin,
                                        boxes[j].height + 2 * kMergeMargin);
                    if ((expanded_i & expanded_j).area() > 0) {
                        boxes[i] = boxes[i] | boxes[j]; // union
                        boxes.erase(boxes.begin() + static_cast<long>(j));
                        // Re-expand after merge
                        expanded_i = cv::Rect(boxes[i].x - kMergeMargin,
                                              boxes[i].y - kMergeMargin,
                                              boxes[i].width + 2 * kMergeMargin,
                                              boxes[i].height + 2 * kMergeMargin);
                        merged = true;
                    } else {
                        ++j;
                    }
                }
            }
        }
    }

    QPainter p(&image);
    p.setPen(QPen(Qt::red, 2));
    for (const auto &r : boxes)
        p.drawRect(r.x, r.y, r.width, r.height);

    // Motion traces from detection centroids
    if (showTraces && !boxes.empty()) {
        double decayFactor = 0.80 + (traceDecay - 1) * (0.19 / 99.0);
        for (auto &t : m_detectionTraces)
            t.opacity *= decayFactor;
        while (!m_detectionTraces.empty() && m_detectionTraces.front().opacity < 0.03)
            m_detectionTraces.pop_front();

        for (const auto &r : boxes) {
            double cx = r.x + r.width * 0.5;
            double cy = r.y + r.height * 0.5;
            m_detectionTraces.push_back({QPointF(cx, cy), 1.0});
        }

        while (static_cast<int>(m_detectionTraces.size()) > kMaxTracePoints)
            m_detectionTraces.pop_front();

        p.setPen(Qt::NoPen);
        for (const auto &t : m_detectionTraces) {
            int alpha = static_cast<int>(t.opacity * 200);
            double radius = 3.0 + t.opacity * 5.0;
            p.setBrush(QColor(255, 80, 80, alpha));
            p.drawEllipse(t.pos, radius, radius);
        }
    } else if (showTraces) {
        // Decay traces even when no boxes this frame
        double decayFactor = 0.80 + (traceDecay - 1) * (0.19 / 99.0);
        for (auto &t : m_detectionTraces)
            t.opacity *= decayFactor;
        while (!m_detectionTraces.empty() && m_detectionTraces.front().opacity < 0.03)
            m_detectionTraces.pop_front();

        if (!m_detectionTraces.empty()) {
            p.setPen(Qt::NoPen);
            for (const auto &t : m_detectionTraces) {
                int alpha = static_cast<int>(t.opacity * 200);
                double radius = 3.0 + t.opacity * 5.0;
                p.setBrush(QColor(255, 80, 80, alpha));
                p.drawEllipse(t.pos, radius, radius);
            }
        }
    }

    p.end();
}

// ─────────────────────────────────────────────────────────────────────────────
// Motion vectors overlay (optical flow)
// ─────────────────────────────────────────────────────────────────────────────
void OpenCVProcessor::applyMotionVectorsOverlay(QImage &image,
                                                const cv::Mat &grayCur,
                                                const cv::Mat &grayPrev,
                                                int sensitivity,
                                                bool showTraces,
                                                int traceDecay)
{
    cv::Mat flow;

    if (m_haveOpenCL) {
        cv::UMat uCur, uPrev, uSmallCur, uSmallPrev, uFlow;
        grayCur.copyTo(uCur);
        grayPrev.copyTo(uPrev);
        cv::resize(uCur, uSmallCur, cv::Size(), 0.1, 0.1);
        cv::resize(uPrev, uSmallPrev, cv::Size(), 0.1, 0.1);
        cv::calcOpticalFlowFarneback(uSmallPrev, uSmallCur, uFlow, 0.5, 2, 9, 2, 5, 1.1, 0);
        uFlow.copyTo(flow);
    } else {
        cv::Mat smallCur, smallPrev;
        cv::resize(grayCur, smallCur, cv::Size(), 0.1, 0.1);
        cv::resize(grayPrev, smallPrev, cv::Size(), 0.1, 0.1);
        cv::calcOpticalFlowFarneback(smallPrev, smallCur, flow, 0.5, 2, 9, 2, 5, 1.1, 0);
    }

    const double scaleX = static_cast<double>(image.width()) / (grayCur.cols * 0.1);
    const double scaleY = static_cast<double>(image.height()) / (grayCur.rows * 0.1);

    // ── Batch vectors by colour bucket for efficient drawing ────────
    static constexpr int kBuckets = 8;
    static const QColor bucketColors[kBuckets] = {
        QColor(0, 0, 255),
        QColor(36, 0, 219),
        QColor(73, 0, 182),
        QColor(109, 0, 146),
        QColor(146, 0, 109),
        QColor(182, 0, 73),
        QColor(219, 0, 36),
        QColor(255, 0, 0),
    };
    std::vector<QLineF> bucketLines[kBuckets];

    for (int y = 0; y < flow.rows; y += 2) {
        for (int x = 0; x < flow.cols; x += 2) {
            const cv::Point2f &fxy = flow.at<cv::Point2f>(y, x);
            double mag = std::sqrt(fxy.x * fxy.x + fxy.y * fxy.y);
            // Magnitude threshold: sensitivity 1→thresh 2.0, sensitivity 100→thresh 0.2
            double magThresh = 2.0 - (sensitivity - 1) * (1.8 / 99.0);
            if (mag < magThresh)
                continue;

            double normMag = std::min(mag / 5.0, 1.0);
            int bucket = std::min(static_cast<int>(normMag * (kBuckets - 1)), kBuckets - 1);

            double px = x * scaleX;
            double py = y * scaleY;
            double ex = px + fxy.x * scaleX * 6;
            double ey = py + fxy.y * scaleY * 6;
            bucketLines[bucket].emplace_back(px, py, ex, ey);

            // Arrowhead
            double angle = std::atan2(ey - py, ex - px);
            constexpr double arrowLen = 6.0;
            bucketLines[bucket].emplace_back(ex, ey, ex - arrowLen * std::cos(angle - 0.4), ey - arrowLen * std::sin(angle - 0.4));
            bucketLines[bucket].emplace_back(ex, ey, ex - arrowLen * std::cos(angle + 0.4), ey - arrowLen * std::sin(angle + 0.4));
        }
    }

    QPainter p(&image);
    p.setRenderHint(QPainter::Antialiasing);
    for (int b = 0; b < kBuckets; ++b) {
        if (bucketLines[b].empty())
            continue;
        p.setPen(QPen(bucketColors[b], 2));
        p.drawLines(bucketLines[b].data(), static_cast<int>(bucketLines[b].size()));
    }

    // ── Motion traces (decaying centroid trails) ────────────────────
    if (showTraces) {
        // Decay existing traces  (slider 1–100 → factor 0.80–0.99)
        double decayFactor = 0.80 + (traceDecay - 1) * (0.19 / 99.0);
        for (auto &t : m_motionTraces)
            t.opacity *= decayFactor;
        while (!m_motionTraces.empty() && m_motionTraces.front().opacity < 0.03)
            m_motionTraces.pop_front();

        // Find high-motion centroids in a coarse grid
        static constexpr int kTCols = 6;
        static constexpr int kTRows = 4;
        int cellW = flow.cols / kTCols;
        int cellH = flow.rows / kTRows;
        if (cellW > 0 && cellH > 0) {
            for (int row = 0; row < kTRows; ++row) {
                for (int col = 0; col < kTCols; ++col) {
                    double sumMag = 0, sumX = 0, sumY = 0;
                    int count = 0;
                    int yEnd = std::min((row + 1) * cellH, flow.rows);
                    int xEnd = std::min((col + 1) * cellW, flow.cols);
                    for (int y = row * cellH; y < yEnd; ++y) {
                        for (int x = col * cellW; x < xEnd; ++x) {
                            const auto &f = flow.at<cv::Point2f>(y, x);
                            double mag = std::sqrt(f.x * f.x + f.y * f.y);
                            if (mag > 1.0) {
                                sumMag += mag;
                                sumX += x * mag;
                                sumY += y * mag;
                                ++count;
                            }
                        }
                    }
                    if (count > cellW * cellH / 10) {
                        double cx = (sumX / sumMag) * scaleX;
                        double cy = (sumY / sumMag) * scaleY;
                        m_motionTraces.push_back({QPointF(cx, cy), 1.0});
                    }
                }
            }
        }

        // Cap trace buffer
        while (static_cast<int>(m_motionTraces.size()) > kMaxTracePoints)
            m_motionTraces.pop_front();

        // Draw traces as fading orange circles
        p.setPen(Qt::NoPen);
        for (const auto &t : m_motionTraces) {
            int alpha = static_cast<int>(t.opacity * 200);
            double radius = 3.0 + t.opacity * 5.0;
            p.setBrush(QColor(255, 165, 0, alpha));
            p.drawEllipse(t.pos, radius, radius);
        }
    }

    p.end();
}

// ─────────────────────────────────────────────────────────────────────────────
// Face detection
// ─────────────────────────────────────────────────────────────────────────────
void OpenCVProcessor::applyFaceDetection(QImage &image, const cv::Mat &bgrClean)
{
    if (!m_faceCascadeLoaded) {
        QString path = QCoreApplication::applicationDirPath() + QStringLiteral("/opencv/haarcascade_frontalface_default.xml");
        m_faceCascadeLoaded = m_faceCascade.load(path.toStdString());
        if (!m_faceCascadeLoaded)
            return;
    }

    cv::Mat small, gray;
    cv::resize(bgrClean, small, cv::Size(), 0.5, 0.5);
    cv::cvtColor(small, gray, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(gray, gray);

    std::vector<cv::Rect> faces;
    m_faceCascade.detectMultiScale(gray, faces, 1.1, 4, 0, cv::Size(30, 30));

    QPainter p(&image);
    p.setPen(QPen(Qt::green, 2));
    for (const auto &f : faces)
        p.drawRect(f.x * 2, f.y * 2, f.width * 2, f.height * 2);
    p.end();
}

// ─────────────────────────────────────────────────────────────────────────────
// Motion level (grid-based, EMA smoothed)
// ─────────────────────────────────────────────────────────────────────────────
double OpenCVProcessor::computeMotionLevel(const cv::Mat &grayCur, const cv::Mat &grayPrev, int sensitivity)
{
    if (grayPrev.empty())
        return 0.0;

    int cellW = grayCur.cols / kGridCols;
    int cellH = grayCur.rows / kGridRows;
    if (cellW < 1 || cellH < 1)
        return 0.0;

    double maxCell = 0.0, sumCell = 0.0;
    constexpr double ema = 0.55;

    for (int row = 0; row < kGridRows; ++row) {
        for (int col = 0; col < kGridCols; ++col) {
            cv::Rect roi(col * cellW, row * cellH, cellW, cellH);
            cv::Mat diff;
            cv::absdiff(grayCur(roi), grayPrev(roi), diff);
            double raw = cv::mean(diff)[0] / 255.0;

            // Noise floor: ignore small diffs (compression artifacts)
            if (raw < 0.008)
                raw = 0.0;

            int idx = row * kGridCols + col;
            m_cellLevels[idx] = ema * m_cellLevels[idx] + (1.0 - ema) * raw;

            // Scale by sensitivity: 1 → 0.5×, 50 → 2.0×, 100 → 4.0×
            double scale = 0.5 + (sensitivity - 1) * (3.5 / 99.0);
            double lv = m_cellLevels[idx] * scale;
            lv = std::sqrt(std::min(lv, 1.0));
            maxCell = std::max(maxCell, lv);
            sumCell += lv;
        }
    }

    double aggregate = 0.6 * maxCell + 0.4 * (sumCell / (kGridCols * kGridRows));
    return std::min(aggregate, 1.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// Grid motion overlay (coloured rectangles per cell)
// ─────────────────────────────────────────────────────────────────────────────
void OpenCVProcessor::applyGridMotionOverlay(QImage &image, int sensitivity)
{
    QPainter p(&image);

    int cellW = image.width() / kGridCols;
    int cellH = image.height() / kGridRows;

    for (int row = 0; row < kGridRows; ++row) {
        for (int col = 0; col < kGridCols; ++col) {
            int idx = row * kGridCols + col;
            double lv = m_cellLevels[idx] * (sensitivity / 25.0);
            lv = std::sqrt(std::min(lv, 1.0));

            QColor c;
            if (lv < 0.4)
                c = QColor(0, 255, 0);
            else if (lv < 0.7)
                c = QColor(255, 255, 0);
            else
                c = QColor(255, 0, 0);

            int alpha = static_cast<int>(lv * 180);
            if (alpha < 8)
                alpha = 0;
            c.setAlpha(alpha);
            p.fillRect(col * cellW, row * cellH, cellW, cellH, c);

            p.setPen(QPen(QColor(255, 255, 255, 40), 1));
            p.drawRect(col * cellW, row * cellH, cellW, cellH);
        }
    }
    p.end();
}

// ─────────────────────────────────────────────────────────────────────────────
// Motion graph overlay (stacked bar chart, sliding window)
// ─────────────────────────────────────────────────────────────────────────────
void OpenCVProcessor::applyMotionGraphOverlay(QImage &image, double motionLevel)
{
    m_graphHistory.push_back(motionLevel);
    if (static_cast<int>(m_graphHistory.size()) > kGraphHistoryLen)
        m_graphHistory.pop_front();

    for (int row = 0; row < kGridRows; ++row) {
        double rowMax = 0.0;
        for (int col = 0; col < kGridCols; ++col)
            rowMax = std::max(rowMax, m_cellLevels[row * kGridCols + col]);
        m_cellHistory[row].push_back(rowMax);
        if (static_cast<int>(m_cellHistory[row].size()) > kGraphHistoryLen)
            m_cellHistory[row].pop_front();
    }

    QPainter p(&image);

    const int gW = 320, gH = 120, margin = 10;
    int gX = margin;
    int gY = image.height() - gH - margin;
    if (gY < 0)
        gY = 0;

    p.fillRect(gX, gY, gW, gH, QColor(0, 0, 0, 160));

    static const QColor rowColors[4] = {
        QColor(66, 133, 244),
        QColor(0, 188, 212),
        QColor(255, 193, 7),
        QColor(244, 67, 54),
    };

    int n = static_cast<int>(m_graphHistory.size());
    double barW = static_cast<double>(gW) / kGraphHistoryLen;

    for (int i = 0; i < n; ++i) {
        double x = gX + i * barW;
        double yBottom = gY + gH;

        for (int row = kGridRows - 1; row >= 0; --row) {
            double val = 0.0;
            if (i < static_cast<int>(m_cellHistory[row].size()))
                val = m_cellHistory[row][static_cast<size_t>(i)];
            double h = val * gH;
            if (h < 1)
                continue;

            p.fillRect(QRectF(x, yBottom - h, barW + 0.5, h), rowColors[row]);
            yBottom -= h;
        }
    }

    p.setPen(Qt::white);
    p.setFont(QFont(QStringLiteral("Monospace"), 9));
    QString pct = QStringLiteral("Motion: %1 %").arg(static_cast<int>(motionLevel * 100));
    p.drawText(gX + 4, gY + 14, pct);

    int lx = gX + gW - 90;
    for (int row = 0; row < kGridRows; ++row) {
        p.fillRect(lx, gY + 4 + row * 14, 10, 10, rowColors[row]);
        p.drawText(lx + 14, gY + 13 + row * 14, QStringLiteral("Row %1").arg(row + 1));
    }

    p.end();
}
