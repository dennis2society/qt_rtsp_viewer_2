#pragma once

#include <QWidget>

class QSlider;
class QCheckBox;
class QComboBox;
class QLabel;
class QLineEdit;
class QPushButton;
class QSpinBox;
class QToolButton;

/// Shared sidebar - always visible, rebinds to the active stream tab.
class EffectsSidebar : public QWidget
{
    Q_OBJECT

public:
    explicit EffectsSidebar(QWidget *parent = nullptr);

    /// Rebind all controls to the stream with the given id.
    void bindToStream(int streamId);

    QSize sizeHint() const override;

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
    QString m_outputFolderPath; // per-stream folder currently shown in sidebar

    // -- controls ----------------------------------------------------
    // Image adjustments
    QSlider *m_blurSlider = nullptr;
    QLabel *m_blurValueLabel = nullptr;
    QLabel *m_blurGpuLabel = nullptr;
    QCheckBox *m_grayscaleCheck = nullptr;
    QSlider *m_brightnessSlider = nullptr;
    QLabel *m_brightnessValueLabel = nullptr;
    QSlider *m_contrastSlider = nullptr;
    QLabel *m_contrastValueLabel = nullptr;
    QSlider *m_colorTempSlider = nullptr;
    QLabel *m_colorTempValueLabel = nullptr;

    // Detection
    QCheckBox *m_motionDetCheck = nullptr;
    QSlider *m_motionSensSlider = nullptr;
    QLabel *m_motionSensValueLabel = nullptr;
    QCheckBox *m_motionVecCheck = nullptr;
    QSlider *m_motionVecSensSlider = nullptr;
    QLabel *m_motionVecSensValueLabel = nullptr;
    QCheckBox *m_motionTraceCheck = nullptr;
    QLabel *m_traceDecayLabel = nullptr;
    QSlider *m_traceDecaySlider = nullptr;
    QCheckBox *m_motionGraphCheck = nullptr;
    QSlider *m_motionGraphSensSlider = nullptr;
    QLabel *m_motionGraphSensValueLabel = nullptr;
    QCheckBox *m_faceDetCheck = nullptr;

    // Overlay
    QCheckBox *m_overlayCheck = nullptr;

    // CSV motion logging
    QCheckBox *m_motionCsvCheck = nullptr;
    QCheckBox *m_recordCleanVideoCheck = nullptr;

    // Recording
    QComboBox *m_codecCombo = nullptr;
    QComboBox *m_formatCombo = nullptr;
    QSpinBox *m_fpsSpin = nullptr;

    // Auto-record
    QCheckBox *m_autoRecCheck = nullptr;
    QLabel *m_thresholdLabel = nullptr;
    QSlider *m_thresholdSlider = nullptr;
    QLabel *m_timeoutLabel = nullptr;
    QSpinBox *m_timeoutSpin = nullptr;
    QLabel *m_autoRecStatusLabel = nullptr;

    // Per-tab output folder
    QPushButton *m_tabFolderBtn = nullptr;
    QLabel *m_tabFolderLabel = nullptr;

    // Global output folder (default for all tabs)
    QPushButton *m_outputFolderBtn = nullptr;
    QLabel *m_outputFolderLabel = nullptr;

    // Reset
    QPushButton *m_resetBtn = nullptr;

    // ONVIF
    QPushButton *m_onvifSettingsBtn = nullptr;
};
