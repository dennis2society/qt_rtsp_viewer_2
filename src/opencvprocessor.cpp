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
// Codec-artifact spike detector
// ─────────────────────────────────────────────────────────────────────────────
bool OpenCVProcessor::isSpikeFrame(const cv::Mat &grayCur, const cv::Mat &grayPrev)
{
    if (grayCur.size() != grayPrev.size())
        return true;

    cv::Mat diff;
    cv::absdiff(grayCur, grayPrev, diff);
    double globalDiff = cv::mean(diff)[0] / 255.0;

    if (m_globalDiffEma < 0.0) {
        m_globalDiffEma = globalDiff;
        return false;
    }

    bool spike = (globalDiff > kSpikeMultiplier * m_globalDiffEma) && (m_globalDiffEma > 0.002);

    if (!spike)
        m_globalDiffEma = kGlobalDiffEmaAlpha * globalDiff + (1.0 - kGlobalDiffEmaAlpha) * m_globalDiffEma;

    return spike;
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
// Motion detection overlay (contour-based)
// ─────────────────────────────────────────────────────────────────────────────
void OpenCVProcessor::applyMotionDetectionOverlay(QImage &image, const cv::Mat &grayCur, const cv::Mat &grayPrev, int sensitivity)
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

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(m_work3, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    QPainter p(&image);
    p.setPen(QPen(Qt::red, 2));
    for (const auto &c : contours) {
        double area = cv::contourArea(c);
        if (area > 500) {
            cv::Rect r = cv::boundingRect(c);
            p.drawRect(r.x, r.y, r.width, r.height);
        }
    }
    p.end();
}

// ─────────────────────────────────────────────────────────────────────────────
// Motion vectors overlay (optical flow)
// ─────────────────────────────────────────────────────────────────────────────
void OpenCVProcessor::applyMotionVectorsOverlay(QImage &image, const cv::Mat &grayCur, const cv::Mat &grayPrev)
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

    QPainter p(&image);
    p.setRenderHint(QPainter::Antialiasing);
    const double scaleX = static_cast<double>(image.width()) / (grayCur.cols * 0.1);
    const double scaleY = static_cast<double>(image.height()) / (grayCur.rows * 0.1);

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

            int idx = row * kGridCols + col;
            m_cellLevels[idx] = ema * m_cellLevels[idx] + (1.0 - ema) * raw;

            double lv = m_cellLevels[idx] * (sensitivity / 25.0);
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
