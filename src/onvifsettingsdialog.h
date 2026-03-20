#pragma once

#include "onvifclient.h"

#include <QDialog>

class QCheckBox;
class QComboBox;
class QGroupBox;
class QLabel;
class QLineEdit;
class QPushButton;
class QSlider;
class QSpinBox;

/// Dialog for connecting to an ONVIF camera and adjusting its imaging settings.
class OnvifSettingsDialog : public QDialog
{
    Q_OBJECT

public:
    explicit OnvifSettingsDialog(int streamId, QWidget *parent = nullptr);

private slots:
    void onConnect();
    void onApply();
    void onRefresh();

private:
    void setupUI();
    void connectSignals();
    void loadCredentials();
    void saveCredentials();
    void setImagingUIEnabled(bool enabled);
    void populateImagingUI();
    OnvifImagingSettings gatherSettings() const;

    int m_streamId;
    OnvifClient *m_client;

    // Connection state
    OnvifCapabilities m_caps;
    OnvifImagingSettings m_currentSettings;
    OnvifImagingOptions m_options;
    QString m_videoSourceToken;
    bool m_gotSettings = false;
    bool m_gotOptions = false;

    // Connection widgets
    QLineEdit *m_hostEdit = nullptr;
    QSpinBox *m_portSpin = nullptr;
    QLineEdit *m_userEdit = nullptr;
    QLineEdit *m_passEdit = nullptr;
    QPushButton *m_connectBtn = nullptr;
    QLabel *m_statusLabel = nullptr;
    QLabel *m_servicesLabel = nullptr;

    // Imaging group
    QGroupBox *m_imagingGroup = nullptr;

    QLabel *m_brightnessLabel = nullptr;
    QSlider *m_brightnessSlider = nullptr;
    QLabel *m_brightnessVal = nullptr;

    QLabel *m_contrastLabel = nullptr;
    QSlider *m_contrastSlider = nullptr;
    QLabel *m_contrastVal = nullptr;

    QLabel *m_saturationLabel = nullptr;
    QSlider *m_saturationSlider = nullptr;
    QLabel *m_saturationVal = nullptr;

    QLabel *m_sharpnessLabel = nullptr;
    QSlider *m_sharpnessSlider = nullptr;
    QLabel *m_sharpnessVal = nullptr;

    QLabel *m_irCutLabel = nullptr;
    QComboBox *m_irCutCombo = nullptr;

    QCheckBox *m_backlightCheck = nullptr;
    QSlider *m_backlightSlider = nullptr;
    QLabel *m_backlightVal = nullptr;

    QCheckBox *m_wdrCheck = nullptr;
    QSlider *m_wdrSlider = nullptr;
    QLabel *m_wdrVal = nullptr;

    QPushButton *m_applyBtn = nullptr;
    QPushButton *m_refreshBtn = nullptr;
    QPushButton *m_closeBtn = nullptr;
};
