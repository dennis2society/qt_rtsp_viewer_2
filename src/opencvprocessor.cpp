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
    std::fill(m_cellLevels.begin(), m_cellLevels.end(), 0.0);
    for (auto &dq : m_cellHistory)
        std::fill(dq.begin(), dq.end(), 0.0);
    m_graphHistory.clear();
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────
cv::Mat OpenCVProcessor::qImageToMat(const QImage &img)
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

QImage OpenCVProcessor::matToQImage(const cv::Mat &mat, QImage::Format fmt)
{
    if (mat.channels() == 3) {
        cv::cvtColor(mat, m_rgbMat, cv::COLOR_BGR2RGB);
        return QImage(m_rgbMat.data, m_rgbMat.cols, m_rgbMat.rows, static_cast<int>(m_rgbMat.step), fmt).copy();
    }
    return QImage(mat.data, mat.cols, mat.rows, static_cast<int>(mat.step), QImage::Format_Grayscale8).copy();
}

// ─────────────────────────────────────────────────────────────────────────────
// Blur
// ─────────────────────────────────────────────────────────────────────────────
QImage OpenCVProcessor::applyGaussBlur(const QImage &src, int amount)
{
    if (amount <= 0)
        return src;
    cv::Mat mat = qImageToMat(src);
    int ks = amount * 2 + 1;
    double sigma = amount * 0.5;

    if (m_haveOpenCL) {
        cv::UMat uSrc, uDst;
        mat.copyTo(uSrc);
        cv::GaussianBlur(uSrc, uDst, cv::Size(ks, ks), sigma);
        uDst.copyTo(m_work1);
    } else {
        cv::GaussianBlur(mat, m_work1, cv::Size(ks, ks), sigma);
    }

    return matToQImage(m_work1, QImage::Format_RGB888);
}

// ─────────────────────────────────────────────────────────────────────────────
// Brightness / Contrast  (LUT-accelerated)
// ─────────────────────────────────────────────────────────────────────────────
QImage OpenCVProcessor::applyBrightnessContrast(const QImage &src, int brightness, int contrast)
{
    if (brightness == 0 && contrast == 0)
        return src;
    cv::Mat mat = qImageToMat(src);

    double alpha = 1.0 + contrast / 100.0; // contrast scale
    double beta = brightness; // brightness offset

    cv::Mat lut(1, 256, CV_8U);
    uchar *p = lut.ptr();
    for (int i = 0; i < 256; ++i)
        p[i] = cv::saturate_cast<uchar>(alpha * i + beta);

    // Split channels, apply LUT to each, merge
    std::vector<cv::Mat> chs;
    cv::split(mat, chs);
    for (auto &ch : chs)
        cv::LUT(ch, lut, ch);
    cv::merge(chs, m_work1);

    return matToQImage(m_work1, QImage::Format_RGB888);
}

// ─────────────────────────────────────────────────────────────────────────────
// Colour temperature  (3-channel LUT)
// ─────────────────────────────────────────────────────────────────────────────
QImage OpenCVProcessor::applyColorTemperature(const QImage &src, int temperature)
{
    if (temperature == 0)
        return src;
    cv::Mat mat = qImageToMat(src);

    double t = temperature / 100.0; // -1.0 … +1.0
    double rScale = 1.0, bScale = 1.0;
    if (t < 0) { // warm → boost red, cut blue
        rScale = 1.0 - t * 0.30;
        bScale = 1.0 + t * 0.30;
    } else { // cool → cut red, boost blue
        rScale = 1.0 - t * 0.30;
        bScale = 1.0 + t * 0.30;
    }

    cv::Mat lutAll(1, 256, CV_8UC3);
    auto *q = lutAll.ptr<cv::Vec3b>();
    for (int i = 0; i < 256; ++i) {
        q[i][0] = cv::saturate_cast<uchar>(i * bScale); // B
        q[i][1] = cv::saturate_cast<uchar>(i); // G
        q[i][2] = cv::saturate_cast<uchar>(i * rScale); // R
    }

    cv::LUT(mat, lutAll, m_work1);
    return matToQImage(m_work1, QImage::Format_RGB888);
}

// ─────────────────────────────────────────────────────────────────────────────
// Codec-artifact spike detector
// ─────────────────────────────────────────────────────────────────────────────
bool OpenCVProcessor::isSpikeFrame(const cv::Mat &grayCur, const cv::Mat &grayPrev)
{
    if (grayCur.size() != grayPrev.size())
        return true; // mismatched sizes (e.g. stream resolution changed) – treat as spike

    cv::Mat diff;
    cv::absdiff(grayCur, grayPrev, diff);
    double globalDiff = cv::mean(diff)[0] / 255.0;

    if (m_globalDiffEma < 0.0) {
        // Initialise on the first frame; never treat first frame as spike
        m_globalDiffEma = globalDiff;
        return false;
    }

    bool spike = (globalDiff > kSpikeMultiplier * m_globalDiffEma) && (m_globalDiffEma > 0.002);

    if (!spike) {
        // Update EMA only from clean frames so it doesn't drift upward
        m_globalDiffEma = kGlobalDiffEmaAlpha * globalDiff + (1.0 - kGlobalDiffEmaAlpha) * m_globalDiffEma;
    }
    return spike;
}

// ─────────────────────────────────────────────────────────────────────────────
// Motion detection overlay (contour-based)
// ─────────────────────────────────────────────────────────────────────────────
QImage OpenCVProcessor::applyMotionDetectionOverlay(const QImage &drawTarget, const QImage &cleanCurrent, const QImage &cleanPrevious, int sensitivity)
{
    if (cleanPrevious.isNull())
        return drawTarget;

    cv::Mat cur = qImageToMat(cleanCurrent);
    cv::Mat prev;
    {
        QImage tmp = cleanPrevious.convertToFormat(QImage::Format_RGB888);
        cv::Mat raw(tmp.height(), tmp.width(), CV_8UC3, const_cast<uchar *>(tmp.constBits()), static_cast<size_t>(tmp.bytesPerLine()));
        cv::cvtColor(raw, prev, cv::COLOR_RGB2BGR);
    }

    cv::Mat grayCur, grayPrev;
    cv::cvtColor(cur, grayCur, cv::COLOR_BGR2GRAY);
    cv::cvtColor(prev, grayPrev, cv::COLOR_BGR2GRAY);

    // Suppress I-frame / codec-artifact spikes
    if (isSpikeFrame(grayCur, grayPrev))
        return drawTarget;

    cv::absdiff(grayCur, grayPrev, m_work2);
    int thresh = std::max(1, 100 - sensitivity);
    cv::threshold(m_work2, m_work3, thresh, 255, cv::THRESH_BINARY);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(m_work3, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    QImage result = drawTarget.copy();
    QPainter p(&result);
    p.setPen(QPen(Qt::red, 2));
    for (const auto &c : contours) {
        double area = cv::contourArea(c);
        if (area > 500) {
            cv::Rect r = cv::boundingRect(c);
            p.drawRect(r.x, r.y, r.width, r.height);
        }
    }
    p.end();
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// Motion vectors overlay (optical flow)
// ─────────────────────────────────────────────────────────────────────────────
QImage OpenCVProcessor::applyMotionVectorsOverlay(const QImage &drawTarget, const QImage &cleanCurrent, const QImage &cleanPrevious)
{
    if (cleanPrevious.isNull())
        return drawTarget;

    cv::Mat cur = qImageToMat(cleanCurrent);
    cv::Mat prev;
    {
        QImage tmp = cleanPrevious.convertToFormat(QImage::Format_RGB888);
        cv::Mat raw(tmp.height(), tmp.width(), CV_8UC3, const_cast<uchar *>(tmp.constBits()), static_cast<size_t>(tmp.bytesPerLine()));
        cv::cvtColor(raw, prev, cv::COLOR_RGB2BGR);
    }

    cv::Mat grayCur, grayPrev;
    cv::cvtColor(cur, grayCur, cv::COLOR_BGR2GRAY);
    cv::cvtColor(prev, grayPrev, cv::COLOR_BGR2GRAY);

    // Suppress I-frame / codec-artifact spikes
    if (isSpikeFrame(grayCur, grayPrev))
        return drawTarget;

    // Downscale to 10 % for performance
    cv::Mat smallCur, smallPrev;
    cv::resize(grayCur, smallCur, cv::Size(), 0.1, 0.1);
    cv::resize(grayPrev, smallPrev, cv::Size(), 0.1, 0.1);

    cv::Mat flow;
    cv::calcOpticalFlowFarneback(smallPrev, smallCur, flow, 0.5, 2, 9, 2, 5, 1.1, 0);

    QImage result = drawTarget.copy();
    QPainter p(&result);
    p.setRenderHint(QPainter::Antialiasing);
    const double scaleX = static_cast<double>(result.width()) / smallCur.cols;
    const double scaleY = static_cast<double>(result.height()) / smallCur.rows;

    for (int y = 0; y < flow.rows; y += 2) {
        for (int x = 0; x < flow.cols; x += 2) {
            const cv::Point2f &fxy = flow.at<cv::Point2f>(y, x);
            double mag = std::sqrt(fxy.x * fxy.x + fxy.y * fxy.y);
            if (mag < 0.3)
                continue;

            double normMag = std::min(mag / 5.0, 1.0);
            int r = static_cast<int>(normMag * 255);
            int b = static_cast<int>((1.0 - normMag) * 255);
            p.setPen(QPen(QColor(r, 0, b), 3));

            double px = x * scaleX;
            double py = y * scaleY;
            double ex = px + fxy.x * scaleX * 6;
            double ey = py + fxy.y * scaleY * 6;
            p.drawLine(QPointF(px, py), QPointF(ex, ey));

            // Draw arrowhead
            double angle = std::atan2(ey - py, ex - px);
            double arrowLen = 6.0;
            double ax1 = ex - arrowLen * std::cos(angle - 0.4);
            double ay1 = ey - arrowLen * std::sin(angle - 0.4);
            double ax2 = ex - arrowLen * std::cos(angle + 0.4);
            double ay2 = ey - arrowLen * std::sin(angle + 0.4);
            p.drawLine(QPointF(ex, ey), QPointF(ax1, ay1));
            p.drawLine(QPointF(ex, ey), QPointF(ax2, ay2));
        }
    }
    p.end();
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// Face detection
// ─────────────────────────────────────────────────────────────────────────────
QImage OpenCVProcessor::applyFaceDetection(const QImage &drawTarget, const QImage &cleanCurrent)
{
    if (!m_faceCascadeLoaded) {
        QString path = QCoreApplication::applicationDirPath() + QStringLiteral("/opencv/haarcascade_frontalface_default.xml");
        m_faceCascadeLoaded = m_faceCascade.load(path.toStdString());
        if (!m_faceCascadeLoaded)
            return drawTarget;
    }

    cv::Mat mat = qImageToMat(cleanCurrent);
    cv::Mat small, gray;
    cv::resize(mat, small, cv::Size(), 0.5, 0.5);
    cv::cvtColor(small, gray, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(gray, gray);

    std::vector<cv::Rect> faces;
    m_faceCascade.detectMultiScale(gray, faces, 1.1, 4, 0, cv::Size(30, 30));

    QImage result = drawTarget.copy();
    QPainter p(&result);
    p.setPen(QPen(Qt::green, 2));
    for (const auto &f : faces)
        p.drawRect(f.x * 2, f.y * 2, f.width * 2, f.height * 2);
    p.end();
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// Motion level (grid-based, EMA smoothed, spike-rejected)
// ─────────────────────────────────────────────────────────────────────────────
double OpenCVProcessor::computeMotionLevel(const QImage &cleanCurrent, const QImage &cleanPrevious, int sensitivity)
{
    if (cleanPrevious.isNull())
        return 0.0;

    cv::Mat cur = qImageToMat(cleanCurrent);
    cv::Mat prev;
    {
        QImage tmp = cleanPrevious.convertToFormat(QImage::Format_RGB888);
        cv::Mat raw(tmp.height(), tmp.width(), CV_8UC3, const_cast<uchar *>(tmp.constBits()), static_cast<size_t>(tmp.bytesPerLine()));
        cv::cvtColor(raw, prev, cv::COLOR_RGB2BGR);
    }

    cv::Mat grayCur, grayPrev;
    cv::cvtColor(cur, grayCur, cv::COLOR_BGR2GRAY);
    cv::cvtColor(prev, grayPrev, cv::COLOR_BGR2GRAY);

    // Suppress I-frame / codec-artifact spikes — keep previous cell levels
    if (isSpikeFrame(grayCur, grayPrev)) {
        // Decay existing levels toward zero so display fades naturally
        constexpr double decayEma = 0.85;
        double maxCell = 0.0, sumCell = 0.0;
        for (int i = 0; i < kGridCols * kGridRows; ++i) {
            m_cellLevels[i] *= decayEma;
            maxCell = std::max(maxCell, m_cellLevels[i]);
            sumCell += m_cellLevels[i];
        }
        return std::min(0.6 * maxCell + 0.4 * (sumCell / (kGridCols * kGridRows)), 1.0);
    }

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
            double raw = cv::mean(diff)[0] / 255.0; // 0 … 1

            int idx = row * kGridCols + col;
            m_cellLevels[idx] = ema * m_cellLevels[idx] + (1.0 - ema) * raw;

            double lv = m_cellLevels[idx] * (sensitivity / 25.0);
            lv = std::sqrt(std::min(lv, 1.0)); // non-linear boost
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
QImage OpenCVProcessor::applyGridMotionOverlay(const QImage &drawTarget, const QImage &cleanCurrent, const QImage &cleanPrevious, int sensitivity)
{
    if (cleanPrevious.isNull())
        return drawTarget;

    // Recompute cell levels (lightweight – reuses m_cellLevels updated in
    // computeMotionLevel which was just called)
    QImage result = drawTarget.copy();
    QPainter p(&result);

    int cellW = result.width() / kGridCols;
    int cellH = result.height() / kGridRows;

    for (int row = 0; row < kGridRows; ++row) {
        for (int col = 0; col < kGridCols; ++col) {
            int idx = row * kGridCols + col;
            double lv = m_cellLevels[idx] * (sensitivity / 25.0);
            lv = std::sqrt(std::min(lv, 1.0)); // non-linear boost

            QColor c;
            if (lv < 0.4)
                c = QColor(0, 255, 0); // green
            else if (lv < 0.7)
                c = QColor(255, 255, 0); // yellow
            else
                c = QColor(255, 0, 0); // red

            int alpha = static_cast<int>(lv * 180);
            if (alpha < 8)
                alpha = 0; // avoid barely-visible noise
            c.setAlpha(alpha);
            p.fillRect(col * cellW, row * cellH, cellW, cellH, c);

            // Draw grid lines
            p.setPen(QPen(QColor(255, 255, 255, 40), 1));
            p.drawRect(col * cellW, row * cellH, cellW, cellH);
        }
    }
    p.end();
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// Motion graph overlay (stacked bar chart, sliding window)
// ─────────────────────────────────────────────────────────────────────────────
QImage OpenCVProcessor::applyMotionGraphOverlay(const QImage &drawTarget, double motionLevel)
{
    m_graphHistory.push_back(motionLevel);
    if (static_cast<int>(m_graphHistory.size()) > kGraphHistoryLen)
        m_graphHistory.pop_front();

    // Also record per-row levels for stacked bars
    for (int row = 0; row < kGridRows; ++row) {
        double rowMax = 0.0;
        for (int col = 0; col < kGridCols; ++col)
            rowMax = std::max(rowMax, m_cellLevels[row * kGridCols + col]);
        m_cellHistory[row].push_back(rowMax);
        if (static_cast<int>(m_cellHistory[row].size()) > kGraphHistoryLen)
            m_cellHistory[row].pop_front();
    }

    QImage result = drawTarget.copy();
    QPainter p(&result);

    // Graph area: bottom-left corner
    const int gW = 320, gH = 120, margin = 10;
    int gX = margin;
    int gY = result.height() - gH - margin;
    if (gY < 0)
        gY = 0;

    // Semi-transparent background
    p.fillRect(gX, gY, gW, gH, QColor(0, 0, 0, 160));

    // Row colours
    static const QColor rowColors[4] = {
        QColor(66, 133, 244), // blue   (top row)
        QColor(0, 188, 212), // teal
        QColor(255, 193, 7), // amber
        QColor(244, 67, 54), // red    (bottom row)
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

    // Percentage readout
    p.setPen(Qt::white);
    p.setFont(QFont(QStringLiteral("Monospace"), 9));
    QString pct = QStringLiteral("Motion: %1 %").arg(static_cast<int>(motionLevel * 100));
    p.drawText(gX + 4, gY + 14, pct);

    // Legend
    int lx = gX + gW - 90;
    for (int row = 0; row < kGridRows; ++row) {
        p.fillRect(lx, gY + 4 + row * 14, 10, 10, rowColors[row]);
        p.drawText(lx + 14, gY + 13 + row * 14, QStringLiteral("Row %1").arg(row + 1));
    }

    p.end();
    return result;
}
