#include "effectssidebar.h"
#include "onvifsettingsdialog.h"
#include "streamstatemanager.h"

#include <QCheckBox>
#include <QComboBox>
#include <QFileDialog>
#include <QFrame>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QSlider>
#include <QSpinBox>
#include <QToolButton>
#include <QVBoxLayout>

#include <opencv2/core/ocl.hpp>

// -----------------------------------------------------------------------------
EffectsSidebar::EffectsSidebar(QWidget *parent)
    : QWidget(parent)
{
    setupUI();
    connectSlots();
}

// -----------------------------------------------------------------------------
// UI construction
// -----------------------------------------------------------------------------
static QSlider *makeSlider(int min, int max, int val)
{
    auto *s = new QSlider(Qt::Horizontal);
    s->setRange(min, max);
    s->setValue(val);
    return s;
}

static QFrame *hLine()
{
    auto *f = new QFrame;
    f->setFrameShape(QFrame::HLine);
    f->setFrameShadow(QFrame::Sunken);
    return f;
}

void EffectsSidebar::setupUI()
{
    auto *mainLay = new QVBoxLayout(this);
    mainLay->setContentsMargins(6, 6, 6, 6);
    mainLay->setSpacing(4);

    bool gpuAvailable = cv::ocl::haveOpenCL();

    auto makeGpuBadge = [gpuAvailable]() -> QLabel * {
        auto *badge = new QLabel;
        if (gpuAvailable) {
            badge->setText(QStringLiteral("GPU"));
            badge->setStyleSheet(
                QStringLiteral("color:#fff;background:#2e7d32;border-radius:3px;padding:0 4px;"
                               "font-size:10px;font-weight:bold;"));
        } else {
            badge->setText(QStringLiteral("CPU"));
            badge->setStyleSheet(
                QStringLiteral("color:#555;background:#ddd;border-radius:3px;padding:0 4px;"
                               "font-size:10px;"));
        }
        return badge;
    };

    // Helper: create a collapsible section, returns the content QVBoxLayout
    auto makeSection = [&](const QString &title) -> QVBoxLayout * {
        auto *btn = new QToolButton;
        btn->setCheckable(true);
        btn->setChecked(true);
        btn->setArrowType(Qt::DownArrow);
        btn->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
        btn->setText(title);
        btn->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
        btn->setStyleSheet(QStringLiteral("font-weight:bold; border:none; background:transparent; text-align:left;"));
        mainLay->addWidget(btn);

        auto *content = new QWidget;
        auto *cLay = new QVBoxLayout(content);
        cLay->setContentsMargins(4, 0, 0, 4);
        cLay->setSpacing(4);
        mainLay->addWidget(content);

        QObject::connect(btn, &QToolButton::toggled, content, [btn, content](bool on) {
            btn->setArrowType(on ? Qt::DownArrow : Qt::RightArrow);
            content->setVisible(on);
        });

        return cLay;
    };

    // Helper: add a slider row with a value label
    auto addSliderRow = [](QVBoxLayout *lay, const QString &labelText, QLabel *&valueLabelOut) {
        auto *row = new QHBoxLayout;
        row->addWidget(new QLabel(labelText));
        row->addStretch();
        valueLabelOut = new QLabel(QStringLiteral("0"));
        valueLabelOut->setStyleSheet(QStringLiteral("color:gray;font-size:10px;"));
        row->addWidget(valueLabelOut);
        lay->addLayout(row);
    };

    // ----------------------------------------------------------------
    // Section: Image Adjustments
    // ----------------------------------------------------------------
    QVBoxLayout *imageLay = makeSection(QStringLiteral("Image Adjustments"));

    {
        auto *blurRow = new QHBoxLayout;
        blurRow->addWidget(new QLabel(QStringLiteral("Blur")));
        blurRow->addStretch();
        m_blurValueLabel = new QLabel(QStringLiteral("0"));
        m_blurValueLabel->setStyleSheet(QStringLiteral("color:gray;font-size:10px;"));
        blurRow->addWidget(m_blurValueLabel);
        m_blurGpuLabel = new QLabel;
        if (gpuAvailable) {
            m_blurGpuLabel->setText(QStringLiteral("GPU (OpenCL)"));
            m_blurGpuLabel->setStyleSheet(
                QStringLiteral("color:#fff;background:#2e7d32;border-radius:3px;padding:0 4px;"
                               "font-size:10px;font-weight:bold;"));
        } else {
            m_blurGpuLabel->setText(QStringLiteral("CPU"));
            m_blurGpuLabel->setStyleSheet(
                QStringLiteral("color:#555;background:#ddd;border-radius:3px;padding:0 4px;"
                               "font-size:10px;"));
        }
        blurRow->addWidget(m_blurGpuLabel);
        imageLay->addLayout(blurRow);
    }
    m_blurSlider = makeSlider(0, 30, 0);
    imageLay->addWidget(m_blurSlider);

    m_grayscaleCheck = new QCheckBox(QStringLiteral("Grayscale"));
    imageLay->addWidget(m_grayscaleCheck);

    {
        auto *row = new QHBoxLayout;
        row->addWidget(new QLabel(QStringLiteral("Brightness")));
        row->addStretch();
        m_brightnessValueLabel = new QLabel(QStringLiteral("0"));
        m_brightnessValueLabel->setStyleSheet(QStringLiteral("color:gray;font-size:10px;"));
        row->addWidget(m_brightnessValueLabel);
        row->addWidget(makeGpuBadge());
        imageLay->addLayout(row);
    }
    m_brightnessSlider = makeSlider(-100, 100, 0);
    imageLay->addWidget(m_brightnessSlider);

    {
        auto *row = new QHBoxLayout;
        row->addWidget(new QLabel(QStringLiteral("Contrast")));
        row->addStretch();
        m_contrastValueLabel = new QLabel(QStringLiteral("0"));
        m_contrastValueLabel->setStyleSheet(QStringLiteral("color:gray;font-size:10px;"));
        row->addWidget(m_contrastValueLabel);
        row->addWidget(makeGpuBadge());
        imageLay->addLayout(row);
    }
    m_contrastSlider = makeSlider(-100, 100, 0);
    imageLay->addWidget(m_contrastSlider);

    {
        auto *row = new QHBoxLayout;
        row->addWidget(new QLabel(QStringLiteral("Colour Temperature")));
        row->addStretch();
        m_colorTempValueLabel = new QLabel(QStringLiteral("0"));
        m_colorTempValueLabel->setStyleSheet(QStringLiteral("color:gray;font-size:10px;"));
        row->addWidget(m_colorTempValueLabel);
        row->addWidget(makeGpuBadge());
        imageLay->addLayout(row);
    }
    m_colorTempSlider = makeSlider(-100, 100, 0);
    imageLay->addWidget(m_colorTempSlider);

    // ----------------------------------------------------------------
    // Section: Detection / Overlays
    // ----------------------------------------------------------------
    QVBoxLayout *detLay = makeSection(QStringLiteral("Detection / Overlays"));

    m_motionDetCheck = new QCheckBox(QStringLiteral("Motion Detection"));
    detLay->addWidget(m_motionDetCheck);

    addSliderRow(detLay, QStringLiteral("  Sensitivity"), m_motionSensValueLabel);
    m_motionSensSlider = makeSlider(1, 100, 20);
    detLay->addWidget(m_motionSensSlider);

    m_motionVecCheck = new QCheckBox(QStringLiteral("Motion Vectors"));
    detLay->addWidget(m_motionVecCheck);

    addSliderRow(detLay, QStringLiteral("  Vectors Sensitivity"), m_motionVecSensValueLabel);
    m_motionVecSensSlider = makeSlider(1, 100, 50);
    detLay->addWidget(m_motionVecSensSlider);

    m_motionTraceCheck = new QCheckBox(QStringLiteral("  Motion Traces"));
    detLay->addWidget(m_motionTraceCheck);
    m_traceDecayLabel = new QLabel(QStringLiteral("  Trace Decay: 50"));
    detLay->addWidget(m_traceDecayLabel);
    m_traceDecaySlider = makeSlider(1, 100, 50);
    detLay->addWidget(m_traceDecaySlider);
    m_traceDecayLabel->setVisible(false);
    m_traceDecaySlider->setVisible(false);

    m_motionGraphCheck = new QCheckBox(QStringLiteral("Motion Graph"));
    detLay->addWidget(m_motionGraphCheck);

    addSliderRow(detLay, QStringLiteral("  Graph Sensitivity"), m_motionGraphSensValueLabel);
    m_motionGraphSensSlider = makeSlider(1, 100, 50);
    detLay->addWidget(m_motionGraphSensSlider);

    m_faceDetCheck = new QCheckBox(QStringLiteral("Face Detection"));
    detLay->addWidget(m_faceDetCheck);

    m_overlayCheck = new QCheckBox(QStringLiteral("FPS / Resolution Overlay"));
    m_overlayCheck->setChecked(true);
    detLay->addWidget(m_overlayCheck);

    // ----------------------------------------------------------------
    // Section: Recording
    // ----------------------------------------------------------------
    QVBoxLayout *recLay = makeSection(QStringLiteral("Recording"));

    recLay->addWidget(new QLabel(QStringLiteral("Codec")));
    m_codecCombo = new QComboBox;
    m_codecCombo->addItem(QStringLiteral("H.264 (libx264)"), QStringLiteral("libx264"));
    m_codecCombo->addItem(QStringLiteral("H.265 (libx265)"), QStringLiteral("libx265"));
    m_codecCombo->addItem(QStringLiteral("Raw Stream (copy, no re-encode)"), QStringLiteral("raw_copy"));
    m_codecCombo->setItemData(m_codecCombo->count() - 1,
                              QStringLiteral("Saves the camera's own H.264/H.265 stream directly.\n"
                                             "No quality loss, no GPU needed.\n"
                                             "Requires ffmpeg in PATH or next to the executable.\n"
                                             "Output is always MP4."),
                              Qt::ToolTipRole);
    recLay->addWidget(m_codecCombo);

    recLay->addWidget(new QLabel(QStringLiteral("Container")));
    m_formatCombo = new QComboBox;
    m_formatCombo->addItem(QStringLiteral("MP4"), QStringLiteral("mp4"));
    m_formatCombo->addItem(QStringLiteral("MKV"), QStringLiteral("mkv"));
    m_formatCombo->addItem(QStringLiteral("AVI"), QStringLiteral("avi"));
    recLay->addWidget(m_formatCombo);

    {
        auto *fpsRow = new QHBoxLayout;
        fpsRow->addWidget(new QLabel(QStringLiteral("Record FPS")));
        m_fpsSpin = new QSpinBox;
        m_fpsSpin->setRange(1, 60);
        m_fpsSpin->setValue(25);
        m_fpsSpin->setToolTip(QStringLiteral("Output framerate for recordings"));
        fpsRow->addWidget(m_fpsSpin);
        recLay->addLayout(fpsRow);
    }

    // CSV motion logging
    m_motionCsvCheck = new QCheckBox(QStringLiteral("Log Motion CSV"));
    m_motionCsvCheck->setToolTip(QStringLiteral("Logs motion events to a CSV file in the output folder (only during recording)"));
    recLay->addWidget(m_motionCsvCheck);
    m_recordCleanVideoCheck = new QCheckBox(QStringLiteral("  Record Clean Video"));
    m_recordCleanVideoCheck->setVisible(false);
    recLay->addWidget(m_recordCleanVideoCheck);

    // ----------------------------------------------------------------
    // Section: Auto-Record
    // ----------------------------------------------------------------
    QVBoxLayout *autoLay = makeSection(QStringLiteral("Auto-Record on Motion"));

    m_autoRecCheck = new QCheckBox(QStringLiteral("Enable auto-record"));
    autoLay->addWidget(m_autoRecCheck);

    m_thresholdLabel = new QLabel(QStringLiteral("Motion Threshold: 50 %"));
    autoLay->addWidget(m_thresholdLabel);
    m_thresholdSlider = makeSlider(1, 100, 50);
    autoLay->addWidget(m_thresholdSlider);

    m_timeoutLabel = new QLabel(QStringLiteral("Stop after (s):"));
    autoLay->addWidget(m_timeoutLabel);
    m_timeoutSpin = new QSpinBox;
    m_timeoutSpin->setRange(1, 120);
    m_timeoutSpin->setValue(5);
    autoLay->addWidget(m_timeoutSpin);

    // Initially hidden
    m_thresholdLabel->setVisible(false);
    m_thresholdSlider->setVisible(false);
    m_timeoutLabel->setVisible(false);
    m_timeoutSpin->setVisible(false);

    m_autoRecStatusLabel = new QLabel;
    m_autoRecStatusLabel->setWordWrap(true);
    m_autoRecStatusLabel->setVisible(false);
    autoLay->addWidget(m_autoRecStatusLabel);

    // ----------------------------------------------------------------
    // Section: Output Folder
    // ----------------------------------------------------------------
    QVBoxLayout *folderLay = makeSection(QStringLiteral("Output Folder"));

    folderLay->addWidget(new QLabel(QStringLiteral("This tab:")));
    m_tabFolderBtn = new QPushButton(QStringLiteral("Select Folder..."));
    m_tabFolderLabel = new QLabel(QStringLiteral("(using global default)"));
    m_tabFolderLabel->setWordWrap(true);
    m_tabFolderLabel->setStyleSheet(QStringLiteral("color:gray;"));
    folderLay->addWidget(m_tabFolderBtn);
    folderLay->addWidget(m_tabFolderLabel);

    folderLay->addWidget(new QLabel(QStringLiteral("Global default (all tabs):")));
    m_outputFolderBtn = new QPushButton(QStringLiteral("Select Global Folder..."));
    m_outputFolderLabel = new QLabel;
    m_outputFolderLabel->setWordWrap(true);
    m_outputFolderLabel->setStyleSheet(QStringLiteral("color:gray;"));
    folderLay->addWidget(m_outputFolderBtn);
    folderLay->addWidget(m_outputFolderLabel);

    // Show current global
    {
        QString cur = StreamStateManager::instance().outputFolder();
        m_outputFolderLabel->setText(cur.isEmpty() ? QStringLiteral("(not set)") : cur);
    }

    // ----------------------------------------------------------------
    // Bottom bar (always visible)
    // ----------------------------------------------------------------
    mainLay->addWidget(hLine());

    m_resetBtn = new QPushButton(QStringLiteral("Reset Effects"));
    mainLay->addWidget(m_resetBtn);

    mainLay->addWidget(hLine());

    m_onvifSettingsBtn = new QPushButton(QStringLiteral("ONVIF Settings..."));
    mainLay->addWidget(m_onvifSettingsBtn);

    mainLay->addStretch(1);

    setLayout(mainLay);
    setMinimumWidth(200);
    setMaximumWidth(270);
}

// -----------------------------------------------------------------------------
// Size hint
// -----------------------------------------------------------------------------
QSize EffectsSidebar::sizeHint() const
{
    return QSize(260, 600);
}

// -----------------------------------------------------------------------------
// Slot wiring
// -----------------------------------------------------------------------------
void EffectsSidebar::connectSlots()
{
    auto changed = [this]() {
        pushState();
    };

    // Value labels + state push
    connect(m_blurSlider, &QSlider::valueChanged, this, [this, changed](int v) {
        m_blurValueLabel->setText(QString::number(v));
        changed();
    });
    connect(m_grayscaleCheck, &QCheckBox::toggled, this, changed);
    connect(m_brightnessSlider, &QSlider::valueChanged, this, [this, changed](int v) {
        m_brightnessValueLabel->setText(QString::number(v));
        changed();
    });
    connect(m_contrastSlider, &QSlider::valueChanged, this, [this, changed](int v) {
        m_contrastValueLabel->setText(QString::number(v));
        changed();
    });
    connect(m_colorTempSlider, &QSlider::valueChanged, this, [this, changed](int v) {
        m_colorTempValueLabel->setText(QString::number(v));
        changed();
    });
    connect(m_motionDetCheck, &QCheckBox::toggled, this, changed);
    connect(m_motionSensSlider, &QSlider::valueChanged, this, [this, changed](int v) {
        m_motionSensValueLabel->setText(QString::number(v));
        changed();
    });
    connect(m_motionVecCheck, &QCheckBox::toggled, this, changed);
    connect(m_motionVecSensSlider, &QSlider::valueChanged, this, [this, changed](int v) {
        m_motionVecSensValueLabel->setText(QString::number(v));
        changed();
    });
    connect(m_motionTraceCheck, &QCheckBox::toggled, this, [this, changed](bool on) {
        m_traceDecayLabel->setVisible(on);
        m_traceDecaySlider->setVisible(on);
        changed();
    });
    connect(m_traceDecaySlider, &QSlider::valueChanged, this, [this, changed](int v) {
        m_traceDecayLabel->setText(QStringLiteral("  Trace Decay: %1").arg(v));
        changed();
    });
    connect(m_motionGraphCheck, &QCheckBox::toggled, this, changed);
    connect(m_motionGraphSensSlider, &QSlider::valueChanged, this, [this, changed](int v) {
        m_motionGraphSensValueLabel->setText(QString::number(v));
        changed();
    });
    connect(m_faceDetCheck, &QCheckBox::toggled, this, changed);
    connect(m_overlayCheck, &QCheckBox::toggled, this, changed);
    connect(m_motionCsvCheck, &QCheckBox::toggled, this, [this, changed](bool on) {
        m_recordCleanVideoCheck->setVisible(on);
        changed();
    });
    connect(m_recordCleanVideoCheck, &QCheckBox::toggled, this, changed);
    connect(m_codecCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this, changed]() {
        const bool isRaw = (m_codecCombo->currentData().toString() == QLatin1String("raw_copy"));
        m_formatCombo->setEnabled(!isRaw);
        m_formatCombo->setToolTip(isRaw ? QStringLiteral("Container is fixed to MP4 for raw stream copy") : QString());
        changed();
    });
    connect(m_formatCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, changed);
    connect(m_fpsSpin, QOverload<int>::of(&QSpinBox::valueChanged), this, changed);

    connect(m_autoRecCheck, &QCheckBox::toggled, this, [this](bool on) {
        m_thresholdLabel->setVisible(on);
        m_thresholdSlider->setVisible(on);
        m_timeoutLabel->setVisible(on);
        m_timeoutSpin->setVisible(on);
        pushState();
    });
    connect(m_thresholdSlider, &QSlider::valueChanged, this, [this](int v) {
        m_thresholdLabel->setText(QStringLiteral("Motion Threshold: %1 %").arg(v));
        pushState();
    });
    connect(m_timeoutSpin, QOverload<int>::of(&QSpinBox::valueChanged), this, changed);

    // Per-tab output folder
    connect(m_tabFolderBtn, &QPushButton::clicked, this, [this]() {
        if (m_boundStream < 0)
            return;
        QString startDir = m_outputFolderPath.isEmpty() ? StreamStateManager::instance().outputFolder() : m_outputFolderPath;
        QString dir = QFileDialog::getExistingDirectory(window(), QStringLiteral("Select Output Folder for This Tab"), startDir, QFileDialog::ShowDirsOnly);
        if (dir.isEmpty())
            return;
        m_outputFolderPath = dir;
        m_tabFolderLabel->setText(dir);
        int sid = m_boundStream;
        StreamStateManager::instance().modifyState(sid, [dir](StreamState &s) {
            s.outputFolder = dir;
        });
    });

    // Global output folder
    connect(m_outputFolderBtn, &QPushButton::clicked, this, [this]() {
        QString dir = QFileDialog::getExistingDirectory(window(),
                                                        QStringLiteral("Select Global Output Folder"),
                                                        StreamStateManager::instance().outputFolder(),
                                                        QFileDialog::ShowDirsOnly);
        if (dir.isEmpty())
            return;
        StreamStateManager::instance().setOutputFolder(dir);
        m_outputFolderLabel->setText(dir);
    });

    // Reset
    connect(m_resetBtn, &QPushButton::clicked, this, [this]() {
        blockAllSignals(true);
        m_blurSlider->setValue(0);
        m_grayscaleCheck->setChecked(false);
        m_brightnessSlider->setValue(0);
        m_contrastSlider->setValue(0);
        m_colorTempSlider->setValue(0);
        m_motionDetCheck->setChecked(false);
        m_motionSensSlider->setValue(20);
        m_motionVecCheck->setChecked(false);
        m_motionVecSensSlider->setValue(50);
        m_motionTraceCheck->setChecked(false);
        m_traceDecaySlider->setValue(50);
        m_traceDecayLabel->setText(QStringLiteral("  Trace Decay: 50"));
        m_traceDecayLabel->setVisible(false);
        m_traceDecaySlider->setVisible(false);
        m_motionGraphCheck->setChecked(false);
        m_motionGraphSensSlider->setValue(50);
        m_faceDetCheck->setChecked(false);
        m_overlayCheck->setChecked(true);
        m_motionCsvCheck->setChecked(false);
        m_recordCleanVideoCheck->setChecked(false);
        m_recordCleanVideoCheck->setVisible(false);
        m_autoRecCheck->setChecked(false);
        m_thresholdSlider->setValue(50);
        m_thresholdLabel->setText(QStringLiteral("Motion Threshold: 50 %"));
        m_timeoutSpin->setValue(5);
        m_thresholdLabel->setVisible(false);
        m_thresholdSlider->setVisible(false);
        m_timeoutLabel->setVisible(false);
        m_timeoutSpin->setVisible(false);
        blockAllSignals(false);
        pushState();
    });

    // React to global settings changes (e.g. output folder changed elsewhere)
    connect(&StreamStateManager::instance(), &StreamStateManager::globalSettingsChanged, this, [this]() {
        QString f = StreamStateManager::instance().outputFolder();
        m_outputFolderLabel->setText(f.isEmpty() ? QStringLiteral("(not set)") : f);
    });

    // React to stream state changes (e.g. auto-recording started/stopped)
    connect(&StreamStateManager::instance(), &StreamStateManager::streamStateChanged, this, [this](int streamId) {
        if (streamId != m_boundStream)
            return;
        StreamState st;
        StreamStateManager::instance().readState(streamId, [&](const StreamState &s) {
            st = s;
        });
        if (st.isAutoRecording) {
            m_autoRecStatusLabel->setText(QStringLiteral("⏺ Auto-recording in progress"));
            m_autoRecStatusLabel->setStyleSheet(QStringLiteral("color:white;background-color:#c62828;padding:4px;font-weight:bold;"));
            m_autoRecStatusLabel->setVisible(true);
        } else {
            m_autoRecStatusLabel->setText(QString());
            m_autoRecStatusLabel->setStyleSheet(QString());
            m_autoRecStatusLabel->setVisible(false);
        }
    });

    // -- ONVIF Settings --------------------------------------------
    connect(m_onvifSettingsBtn, &QPushButton::clicked, this, [this]() {
        if (m_boundStream < 0)
            return;
        auto *dlg = new OnvifSettingsDialog(m_boundStream, this);
        dlg->setAttribute(Qt::WA_DeleteOnClose);
        dlg->show();
    });
}

// -----------------------------------------------------------------------------
// Rebind to a different stream
// -----------------------------------------------------------------------------
void EffectsSidebar::bindToStream(int streamId)
{
    m_boundStream = streamId;

    StreamState st;
    StreamStateManager::instance().readState(streamId, [&](const StreamState &s) {
        st = s;
    });

    blockAllSignals(true);

    m_blurSlider->setValue(st.blurAmount);
    m_blurValueLabel->setText(QString::number(st.blurAmount));
    m_grayscaleCheck->setChecked(st.grayscaleEnabled);
    m_brightnessSlider->setValue(st.brightnessAmount);
    m_brightnessValueLabel->setText(QString::number(st.brightnessAmount));
    m_contrastSlider->setValue(st.contrastAmount);
    m_contrastValueLabel->setText(QString::number(st.contrastAmount));
    m_colorTempSlider->setValue(st.colorTemperature);
    m_colorTempValueLabel->setText(QString::number(st.colorTemperature));
    m_motionDetCheck->setChecked(st.motionDetectionEnabled);
    m_motionSensSlider->setValue(st.motionSensitivity);
    m_motionSensValueLabel->setText(QString::number(st.motionSensitivity));
    m_motionVecCheck->setChecked(st.motionVectorsEnabled);
    m_motionVecSensSlider->setValue(st.motionVectorsSensitivity);
    m_motionVecSensValueLabel->setText(QString::number(st.motionVectorsSensitivity));
    m_motionTraceCheck->setChecked(st.motionTracesEnabled);
    m_traceDecaySlider->setValue(st.motionTraceDecay);
    m_traceDecayLabel->setText(QStringLiteral("  Trace Decay: %1").arg(st.motionTraceDecay));
    m_traceDecayLabel->setVisible(st.motionTracesEnabled);
    m_traceDecaySlider->setVisible(st.motionTracesEnabled);
    m_motionGraphCheck->setChecked(st.motionGraphEnabled);
    m_motionGraphSensSlider->setValue(st.motionGraphSensitivity);
    m_motionGraphSensValueLabel->setText(QString::number(st.motionGraphSensitivity));
    m_faceDetCheck->setChecked(st.faceDetectionEnabled);
    m_overlayCheck->setChecked(st.overlayEnabled);
    m_motionCsvCheck->setChecked(st.motionCsvEnabled);
    m_recordCleanVideoCheck->setChecked(st.recordCleanVideo);
    m_recordCleanVideoCheck->setVisible(st.motionCsvEnabled);

    // Codec
    int ci = m_codecCombo->findData(st.recordCodec);
    if (ci >= 0)
        m_codecCombo->setCurrentIndex(ci);

    // Format
    int fi = m_formatCombo->findData(st.recordFormat);
    if (fi >= 0)
        m_formatCombo->setCurrentIndex(fi);

    // Record FPS
    m_fpsSpin->setValue(st.recordFps > 0 ? static_cast<int>(st.recordFps) : 25);

    // Disable container combo when raw copy is selected (extension is fixed to MP4)
    {
        const bool isRaw = (st.recordCodec == QLatin1String("raw_copy"));
        m_formatCombo->setEnabled(!isRaw);
        m_formatCombo->setToolTip(isRaw ? QStringLiteral("Container is fixed to MP4 for raw stream copy") : QString());
    }

    // Auto-record
    m_autoRecCheck->setChecked(st.autoRecordEnabled);
    int pct = static_cast<int>(st.autoRecordThreshold * 100);
    m_thresholdSlider->setValue(pct);
    m_thresholdLabel->setText(QStringLiteral("Motion Threshold: %1 %").arg(pct));
    m_timeoutSpin->setValue(st.autoRecordTimeout);

    bool ar = st.autoRecordEnabled;
    m_thresholdLabel->setVisible(ar);
    m_thresholdSlider->setVisible(ar);
    m_timeoutLabel->setVisible(ar);
    m_timeoutSpin->setVisible(ar);

    // Auto-record status
    if (st.isAutoRecording) {
        m_autoRecStatusLabel->setText(QStringLiteral("⏺ Auto-recording in progress"));
        m_autoRecStatusLabel->setStyleSheet(QStringLiteral("color:white;background-color:#c62828;padding:4px;font-weight:bold;"));
        m_autoRecStatusLabel->setVisible(true);
    } else {
        m_autoRecStatusLabel->setText(QString());
        m_autoRecStatusLabel->setStyleSheet(QString());
        m_autoRecStatusLabel->setVisible(false);
    }

    // Per-tab output folder
    m_outputFolderPath = st.outputFolder;
    m_tabFolderLabel->setText(m_outputFolderPath.isEmpty() ? QStringLiteral("(using global default)") : m_outputFolderPath);

    blockAllSignals(false);
}

// -----------------------------------------------------------------------------
// Push widget values -> StreamStateManager
// -----------------------------------------------------------------------------
void EffectsSidebar::pushState()
{
    if (m_boundStream < 0)
        return;

    int streamId = m_boundStream;

    StreamStateManager::instance().modifyState(streamId, [this](StreamState &s) {
        s.blurAmount = m_blurSlider->value();
        s.grayscaleEnabled = m_grayscaleCheck->isChecked();
        s.brightnessAmount = m_brightnessSlider->value();
        s.contrastAmount = m_contrastSlider->value();
        s.colorTemperature = m_colorTempSlider->value();
        s.motionDetectionEnabled = m_motionDetCheck->isChecked();
        s.motionSensitivity = m_motionSensSlider->value();
        s.motionVectorsEnabled = m_motionVecCheck->isChecked();
        s.motionVectorsSensitivity = m_motionVecSensSlider->value();
        s.motionTracesEnabled = m_motionTraceCheck->isChecked();
        s.motionTraceDecay = m_traceDecaySlider->value();
        s.motionGraphEnabled = m_motionGraphCheck->isChecked();
        s.motionGraphSensitivity = m_motionGraphSensSlider->value();
        s.faceDetectionEnabled = m_faceDetCheck->isChecked();
        s.overlayEnabled = m_overlayCheck->isChecked();
        s.motionCsvEnabled = m_motionCsvCheck->isChecked();
        s.recordCleanVideo = m_recordCleanVideoCheck->isChecked();
        s.recordCodec = m_codecCombo->currentData().toString();
        s.recordFormat = m_formatCombo->currentData().toString();
        s.recordFps = m_fpsSpin->value();
        s.autoRecordEnabled = m_autoRecCheck->isChecked();
        s.autoRecordThreshold = m_thresholdSlider->value() / 100.0;
        s.autoRecordTimeout = m_timeoutSpin->value();
    });

    emit effectsChanged(streamId);
}

// -----------------------------------------------------------------------------
void EffectsSidebar::blockAllSignals(bool block)
{
    for (auto *w : findChildren<QSlider *>())
        w->blockSignals(block);
    for (auto *w : findChildren<QCheckBox *>())
        w->blockSignals(block);
    for (auto *w : findChildren<QComboBox *>())
        w->blockSignals(block);
    for (auto *w : findChildren<QSpinBox *>())
        w->blockSignals(block);
}
