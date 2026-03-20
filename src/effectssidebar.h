#pragma once

#include <QWidget>

class QSlider;
class QCheckBox;
class QComboBox;
class QLabel;
class QLineEdit;
class QPushButton;
class QSpinBox;
class OnvifClient;

/// Shared sidebar — always visible, rebinds to the active stream tab.
class EffectsSidebar : public QWidget
{
    Q_OBJECT

public:
    explicit EffectsSidebar(QWidget *parent = nullptr);

    /// Rebind all controls to the stream with the given id.
    void bindToStream(int streamId);

signals:
    /// Emitted when any per-stream effect parameter changes.
    void effectsChanged(int streamId);

private:
    void setupUI();
    void connectSlots();

    // Push current widget values into StreamStateManager for m_boundStream.
    void pushState();

    // Block / unblock signals on all effect widgets.
    void blockAllSignals(bool block);

    int m_boundStream = -1;

    // ── controls ────────────────────────────────────────────────────
    // Image adjustments
    QSlider *m_blurSlider = nullptr;
    QLabel *m_blurGpuLabel = nullptr;
    QCheckBox *m_grayscaleCheck = nullptr;
    QSlider *m_brightnessSlider = nullptr;
    QSlider *m_contrastSlider = nullptr;
    QSlider *m_colorTempSlider = nullptr;

    // Detection
    QCheckBox *m_motionDetCheck = nullptr;
    QSlider *m_motionSensSlider = nullptr;
    QCheckBox *m_motionVecCheck = nullptr;
    QCheckBox *m_motionGraphCheck = nullptr;
    QSlider *m_motionGraphSensSlider = nullptr;
    QCheckBox *m_faceDetCheck = nullptr;

    // Overlay
    QCheckBox *m_overlayCheck = nullptr;

    // Recording
    QComboBox *m_codecCombo = nullptr;
    QComboBox *m_formatCombo = nullptr;

    // Auto-record
    QCheckBox *m_autoRecCheck = nullptr;
    QLabel *m_thresholdLabel = nullptr;
    QSlider *m_thresholdSlider = nullptr;
    QLabel *m_timeoutLabel = nullptr;
    QSpinBox *m_timeoutSpin = nullptr;
    QLabel *m_autoRecStatusLabel = nullptr;

    // Global output folder
    QPushButton *m_outputFolderBtn = nullptr;
    QLabel *m_outputFolderLabel = nullptr;

    // Reset
    QPushButton *m_resetBtn = nullptr;

    // ONVIF capabilities
    QLineEdit *m_onvifHostEdit = nullptr;
    QSpinBox *m_onvifPortSpin = nullptr;
    QLineEdit *m_onvifUserEdit = nullptr;
    QLineEdit *m_onvifPassEdit = nullptr;
    QPushButton *m_onvifQueryBtn = nullptr;
    QLabel *m_onvifStatusLabel = nullptr;
    OnvifClient *m_onvifClient = nullptr;
};
