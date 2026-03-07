#include "effectssidebar.h"
#include "streamstatemanager.h"

#include <QCheckBox>
#include <QComboBox>
#include <QFileDialog>
#include <QFrame>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QSlider>
#include <QSpinBox>
#include <QVBoxLayout>

// ─────────────────────────────────────────────────────────────────────────────
EffectsSidebar::EffectsSidebar(QWidget *parent)
    : QWidget(parent)
{
    setupUI();
    connectSlots();
}

// ─────────────────────────────────────────────────────────────────────────────
// UI construction
// ─────────────────────────────────────────────────────────────────────────────
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
    auto *lay = new QVBoxLayout(this);
    lay->setContentsMargins(6, 6, 6, 6);
    lay->setSpacing(6);

    auto addLabel = [&](const QString &text) {
        auto *l = new QLabel(text);
        l->setStyleSheet(QStringLiteral("font-weight:bold;"));
        lay->addWidget(l);
    };

    // ── Image adjustments ───────────────────────────────────────────
    addLabel(QStringLiteral("Image Adjustments"));

    lay->addWidget(new QLabel(QStringLiteral("Blur")));
    m_blurSlider = makeSlider(0, 30, 0);
    lay->addWidget(m_blurSlider);

    m_grayscaleCheck = new QCheckBox(QStringLiteral("Grayscale"));
    lay->addWidget(m_grayscaleCheck);

    lay->addWidget(new QLabel(QStringLiteral("Brightness")));
    m_brightnessSlider = makeSlider(-100, 100, 0);
    lay->addWidget(m_brightnessSlider);

    lay->addWidget(new QLabel(QStringLiteral("Contrast")));
    m_contrastSlider = makeSlider(-100, 100, 0);
    lay->addWidget(m_contrastSlider);

    lay->addWidget(new QLabel(QStringLiteral("Colour Temperature")));
    m_colorTempSlider = makeSlider(-100, 100, 0);
    lay->addWidget(m_colorTempSlider);

    lay->addWidget(hLine());

    // ── Detection ───────────────────────────────────────────────────
    addLabel(QStringLiteral("Detection / Overlays"));

    m_motionDetCheck = new QCheckBox(QStringLiteral("Motion Detection"));
    lay->addWidget(m_motionDetCheck);
    lay->addWidget(new QLabel(QStringLiteral("  Sensitivity")));
    m_motionSensSlider = makeSlider(1, 100, 20);
    lay->addWidget(m_motionSensSlider);

    m_motionVecCheck = new QCheckBox(QStringLiteral("Motion Vectors"));
    lay->addWidget(m_motionVecCheck);

    m_motionGraphCheck = new QCheckBox(QStringLiteral("Motion Graph"));
    lay->addWidget(m_motionGraphCheck);
    lay->addWidget(new QLabel(QStringLiteral("  Graph Sensitivity")));
    m_motionGraphSensSlider = makeSlider(1, 100, 50);
    lay->addWidget(m_motionGraphSensSlider);

    m_faceDetCheck = new QCheckBox(QStringLiteral("Face Detection"));
    lay->addWidget(m_faceDetCheck);

    m_overlayCheck = new QCheckBox(QStringLiteral("FPS / Resolution Overlay"));
    m_overlayCheck->setChecked(true);
    lay->addWidget(m_overlayCheck);

    lay->addWidget(hLine());

    // ── Recording ───────────────────────────────────────────────────
    addLabel(QStringLiteral("Recording"));

    lay->addWidget(new QLabel(QStringLiteral("Codec")));
    m_codecCombo = new QComboBox;
    m_codecCombo->addItem(QStringLiteral("H.264 (libx264)"), QStringLiteral("libx264"));
    m_codecCombo->addItem(QStringLiteral("H.265 (libx265)"), QStringLiteral("libx265"));
    lay->addWidget(m_codecCombo);

    lay->addWidget(new QLabel(QStringLiteral("Container")));
    m_formatCombo = new QComboBox;
    m_formatCombo->addItem(QStringLiteral("MP4"), QStringLiteral("mp4"));
    m_formatCombo->addItem(QStringLiteral("MKV"), QStringLiteral("mkv"));
    m_formatCombo->addItem(QStringLiteral("AVI"), QStringLiteral("avi"));
    lay->addWidget(m_formatCombo);

    lay->addWidget(hLine());

    // ── Auto-record ─────────────────────────────────────────────────
    addLabel(QStringLiteral("Auto-Record on Motion"));

    m_autoRecCheck = new QCheckBox(QStringLiteral("Enable auto-record"));
    lay->addWidget(m_autoRecCheck);

    m_thresholdLabel = new QLabel(QStringLiteral("Motion Threshold: 50 %"));
    lay->addWidget(m_thresholdLabel);
    m_thresholdSlider = makeSlider(1, 100, 50);
    lay->addWidget(m_thresholdSlider);

    m_timeoutLabel = new QLabel(QStringLiteral("Stop after (s):"));
    lay->addWidget(m_timeoutLabel);
    m_timeoutSpin = new QSpinBox;
    m_timeoutSpin->setRange(1, 120);
    m_timeoutSpin->setValue(5);
    lay->addWidget(m_timeoutSpin);

    // Initially hidden
    m_thresholdLabel->setVisible(false);
    m_thresholdSlider->setVisible(false);
    m_timeoutLabel->setVisible(false);
    m_timeoutSpin->setVisible(false);

    m_autoRecStatusLabel = new QLabel;
    m_autoRecStatusLabel->setWordWrap(true);
    m_autoRecStatusLabel->setVisible(false);
    lay->addWidget(m_autoRecStatusLabel);

    lay->addWidget(hLine());

    // ── Global output folder ────────────────────────────────────────
    addLabel(QStringLiteral("Output Folder (global)"));
    m_outputFolderBtn = new QPushButton(QStringLiteral("Select Folder…"));
    m_outputFolderLabel = new QLabel;
    m_outputFolderLabel->setWordWrap(true);
    m_outputFolderLabel->setStyleSheet(QStringLiteral("color:gray;"));
    lay->addWidget(m_outputFolderBtn);
    lay->addWidget(m_outputFolderLabel);

    // Show current
    QString cur = StreamStateManager::instance().outputFolder();
    m_outputFolderLabel->setText(cur.isEmpty() ? QStringLiteral("(not set)") : cur);

    lay->addWidget(hLine());

    // ── Reset ───────────────────────────────────────────────────────
    m_resetBtn = new QPushButton(QStringLiteral("Reset Effects"));
    lay->addWidget(m_resetBtn);

    lay->addStretch(1);
    setLayout(lay);
    setMinimumWidth(200);
    setMaximumWidth(270);
}

// ─────────────────────────────────────────────────────────────────────────────
// Slot wiring
// ─────────────────────────────────────────────────────────────────────────────
void EffectsSidebar::connectSlots()
{
    auto changed = [this]() {
        pushState();
    };

    connect(m_blurSlider, &QSlider::valueChanged, this, changed);
    connect(m_grayscaleCheck, &QCheckBox::toggled, this, changed);
    connect(m_brightnessSlider, &QSlider::valueChanged, this, changed);
    connect(m_contrastSlider, &QSlider::valueChanged, this, changed);
    connect(m_colorTempSlider, &QSlider::valueChanged, this, changed);
    connect(m_motionDetCheck, &QCheckBox::toggled, this, changed);
    connect(m_motionSensSlider, &QSlider::valueChanged, this, changed);
    connect(m_motionVecCheck, &QCheckBox::toggled, this, changed);
    connect(m_motionGraphCheck, &QCheckBox::toggled, this, changed);
    connect(m_motionGraphSensSlider, &QSlider::valueChanged, this, changed);
    connect(m_faceDetCheck, &QCheckBox::toggled, this, changed);
    connect(m_overlayCheck, &QCheckBox::toggled, this, changed);
    connect(m_codecCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, changed);
    connect(m_formatCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, changed);

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

    // Global output folder
    connect(m_outputFolderBtn, &QPushButton::clicked, this, [this]() {
        QString dir = QFileDialog::getExistingDirectory(this, QStringLiteral("Select Output Folder"), StreamStateManager::instance().outputFolder());
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
        m_motionGraphCheck->setChecked(false);
        m_motionGraphSensSlider->setValue(50);
        m_faceDetCheck->setChecked(false);
        m_overlayCheck->setChecked(true);
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
}

// ─────────────────────────────────────────────────────────────────────────────
// Rebind to a different stream
// ─────────────────────────────────────────────────────────────────────────────
void EffectsSidebar::bindToStream(int streamId)
{
    m_boundStream = streamId;

    StreamState st;
    StreamStateManager::instance().readState(streamId, [&](const StreamState &s) {
        st = s;
    });

    blockAllSignals(true);

    m_blurSlider->setValue(st.blurAmount);
    m_grayscaleCheck->setChecked(st.grayscaleEnabled);
    m_brightnessSlider->setValue(st.brightnessAmount);
    m_contrastSlider->setValue(st.contrastAmount);
    m_colorTempSlider->setValue(st.colorTemperature);
    m_motionDetCheck->setChecked(st.motionDetectionEnabled);
    m_motionSensSlider->setValue(st.motionSensitivity);
    m_motionVecCheck->setChecked(st.motionVectorsEnabled);
    m_motionGraphCheck->setChecked(st.motionGraphEnabled);
    m_motionGraphSensSlider->setValue(st.motionGraphSensitivity);
    m_faceDetCheck->setChecked(st.faceDetectionEnabled);
    m_overlayCheck->setChecked(st.overlayEnabled);

    // Codec
    int ci = m_codecCombo->findData(st.recordCodec);
    if (ci >= 0)
        m_codecCombo->setCurrentIndex(ci);

    // Format
    int fi = m_formatCombo->findData(st.recordFormat);
    if (fi >= 0)
        m_formatCombo->setCurrentIndex(fi);

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

    blockAllSignals(false);
}

// ─────────────────────────────────────────────────────────────────────────────
// Push widget values → StreamStateManager
// ─────────────────────────────────────────────────────────────────────────────
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
        s.motionGraphEnabled = m_motionGraphCheck->isChecked();
        s.motionGraphSensitivity = m_motionGraphSensSlider->value();
        s.faceDetectionEnabled = m_faceDetCheck->isChecked();
        s.overlayEnabled = m_overlayCheck->isChecked();
        s.recordCodec = m_codecCombo->currentData().toString();
        s.recordFormat = m_formatCombo->currentData().toString();
        s.autoRecordEnabled = m_autoRecCheck->isChecked();
        s.autoRecordThreshold = m_thresholdSlider->value() / 100.0;
        s.autoRecordTimeout = m_timeoutSpin->value();
    });

    emit effectsChanged(streamId);
}

// ─────────────────────────────────────────────────────────────────────────────
void EffectsSidebar::blockAllSignals(bool block)
{
    m_blurSlider->blockSignals(block);
    m_grayscaleCheck->blockSignals(block);
    m_brightnessSlider->blockSignals(block);
    m_contrastSlider->blockSignals(block);
    m_colorTempSlider->blockSignals(block);
    m_motionDetCheck->blockSignals(block);
    m_motionSensSlider->blockSignals(block);
    m_motionVecCheck->blockSignals(block);
    m_motionGraphCheck->blockSignals(block);
    m_motionGraphSensSlider->blockSignals(block);
    m_faceDetCheck->blockSignals(block);
    m_overlayCheck->blockSignals(block);
    m_codecCombo->blockSignals(block);
    m_formatCombo->blockSignals(block);
    m_autoRecCheck->blockSignals(block);
    m_thresholdSlider->blockSignals(block);
    m_timeoutSpin->blockSignals(block);
}
