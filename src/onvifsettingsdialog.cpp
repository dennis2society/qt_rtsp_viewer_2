#include "onvifsettingsdialog.h"
#include "streamstatemanager.h"

#include <QCheckBox>
#include <QComboBox>
#include <QFormLayout>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QPushButton>
#include <QSettings>
#include <QSlider>
#include <QSpinBox>
#include <QTextEdit>
#include <QTimer>
#include <QUrl>
#include <QVBoxLayout>

// -----------------------------------------------------------------------------
OnvifSettingsDialog::OnvifSettingsDialog(int streamId, QWidget *parent)
    : QDialog(parent)
    , m_streamId(streamId)
    , m_client(new OnvifClient(this))
{
    setWindowTitle(QStringLiteral("ONVIF Camera Settings"));
    setMinimumWidth(420);
    setupUI();
    connectSignals();
    loadCredentials();
    setImagingUIEnabled(false);
}

// -----------------------------------------------------------------------------
void OnvifSettingsDialog::setupUI()
{
    auto *mainLay = new QVBoxLayout(this);

    // -- Connection group --------------------------------------------
    auto *connGroup = new QGroupBox(QStringLiteral("Connection"));
    auto *connLay = new QGridLayout(connGroup);

    connLay->addWidget(new QLabel(QStringLiteral("Host:")), 0, 0);
    m_hostEdit = new QLineEdit;
    m_hostEdit->setPlaceholderText(QStringLiteral("e.g. 192.168.1.100"));
    connLay->addWidget(m_hostEdit, 0, 1);

    connLay->addWidget(new QLabel(QStringLiteral("Port:")), 0, 2);
    m_portSpin = new QSpinBox;
    m_portSpin->setRange(1, 65535);
    m_portSpin->setValue(80);
    m_portSpin->setMaximumWidth(80);
    connLay->addWidget(m_portSpin, 0, 3);

    connLay->addWidget(new QLabel(QStringLiteral("Username:")), 1, 0);
    m_userEdit = new QLineEdit;
    connLay->addWidget(m_userEdit, 1, 1);

    connLay->addWidget(new QLabel(QStringLiteral("Password:")), 1, 2);
    m_passEdit = new QLineEdit;
    m_passEdit->setEchoMode(QLineEdit::Password);
    connLay->addWidget(m_passEdit, 1, 3);

    auto *connBtnLay = new QHBoxLayout;
    m_connectBtn = new QPushButton(QStringLiteral("Connect"));
    connBtnLay->addWidget(m_connectBtn);
    connBtnLay->addStretch();
    connLay->addLayout(connBtnLay, 2, 0, 1, 4);

    m_statusLabel = new QLabel(QStringLiteral("Not connected"));
    m_statusLabel->setStyleSheet(QStringLiteral("color:gray;"));
    connLay->addWidget(m_statusLabel, 3, 0, 1, 4);

    m_servicesLabel = new QLabel;
    m_servicesLabel->setWordWrap(true);
    m_servicesLabel->setVisible(false);
    connLay->addWidget(m_servicesLabel, 4, 0, 1, 4);

    mainLay->addWidget(connGroup);

    // -- Imaging settings group --------------------------------------
    m_imagingGroup = new QGroupBox(QStringLiteral("Imaging Settings"));
    auto *imgLay = new QGridLayout(m_imagingGroup);
    int row = 0;

    auto addSliderRow = [&](const QString &text, QLabel *&label, QSlider *&slider, QLabel *&valLabel) {
        label = new QLabel(text);
        slider = new QSlider(Qt::Horizontal);
        slider->setRange(0, 100);
        valLabel = new QLabel(QStringLiteral("-"));
        valLabel->setMinimumWidth(30);
        valLabel->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
        imgLay->addWidget(label, row, 0);
        imgLay->addWidget(slider, row, 1);
        imgLay->addWidget(valLabel, row, 2);
        connect(slider, &QSlider::valueChanged, valLabel, [valLabel](int v) {
            valLabel->setText(QString::number(v));
        });
        ++row;
    };

    addSliderRow(QStringLiteral("Brightness:"), m_brightnessLabel, m_brightnessSlider, m_brightnessVal);
    addSliderRow(QStringLiteral("Contrast:"), m_contrastLabel, m_contrastSlider, m_contrastVal);
    addSliderRow(QStringLiteral("Color Saturation:"), m_saturationLabel, m_saturationSlider, m_saturationVal);
    addSliderRow(QStringLiteral("Sharpness:"), m_sharpnessLabel, m_sharpnessSlider, m_sharpnessVal);

    // IR Cut Filter
    m_irCutLabel = new QLabel(QStringLiteral("IR Cut Filter:"));
    m_irCutCombo = new QComboBox;
    m_irCutCombo->addItems({QStringLiteral("ON"), QStringLiteral("OFF"), QStringLiteral("AUTO")});
    imgLay->addWidget(m_irCutLabel, row, 0);
    imgLay->addWidget(m_irCutCombo, row, 1, 1, 2);
    ++row;

    // Backlight Compensation
    m_backlightCheck = new QCheckBox(QStringLiteral("Backlight Compensation"));
    m_backlightSlider = new QSlider(Qt::Horizontal);
    m_backlightSlider->setRange(0, 100);
    m_backlightVal = new QLabel(QStringLiteral("-"));
    m_backlightVal->setMinimumWidth(30);
    m_backlightVal->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    imgLay->addWidget(m_backlightCheck, row, 0);
    imgLay->addWidget(m_backlightSlider, row, 1);
    imgLay->addWidget(m_backlightVal, row, 2);
    connect(m_backlightSlider, &QSlider::valueChanged, m_backlightVal, [this](int v) {
        m_backlightVal->setText(QString::number(v));
    });
    ++row;

    // Wide Dynamic Range
    m_wdrCheck = new QCheckBox(QStringLiteral("Wide Dynamic Range"));
    m_wdrSlider = new QSlider(Qt::Horizontal);
    m_wdrSlider->setRange(0, 100);
    m_wdrVal = new QLabel(QStringLiteral("-"));
    m_wdrVal->setMinimumWidth(30);
    m_wdrVal->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    imgLay->addWidget(m_wdrCheck, row, 0);
    imgLay->addWidget(m_wdrSlider, row, 1);
    imgLay->addWidget(m_wdrVal, row, 2);
    connect(m_wdrSlider, &QSlider::valueChanged, m_wdrVal, [this](int v) {
        m_wdrVal->setText(QString::number(v));
    });
    ++row;

    // -- Exposure group ----------------------------------------------
    m_exposureModeLabel = new QLabel(QStringLiteral("Exposure Mode:"));
    m_exposureModeCombo = new QComboBox;
    m_exposureModeCombo->addItems({QStringLiteral("AUTO"), QStringLiteral("MANUAL")});
    imgLay->addWidget(m_exposureModeLabel, row, 0);
    imgLay->addWidget(m_exposureModeCombo, row, 1, 1, 2);
    ++row;

    auto addExpSlider = [&](const QString &text, QLabel *&label, QSlider *&slider, QLabel *&valLabel) {
        label = new QLabel(text);
        slider = new QSlider(Qt::Horizontal);
        slider->setRange(0, 100);
        valLabel = new QLabel(QStringLiteral("-"));
        valLabel->setMinimumWidth(50);
        valLabel->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
        imgLay->addWidget(label, row, 0);
        imgLay->addWidget(slider, row, 1);
        imgLay->addWidget(valLabel, row, 2);
        connect(slider, &QSlider::valueChanged, valLabel, [valLabel, slider](int v) {
            Q_UNUSED(v);
            // Show the real double value mapped from slider position
            valLabel->setText(QString::number(slider->value()));
        });
        ++row;
    };

    addExpSlider(QStringLiteral("Exposure Time:"), m_exposureTimeLabel, m_exposureTimeSlider, m_exposureTimeVal);
    addExpSlider(QStringLiteral("Gain:"), m_gainLabel, m_gainSlider, m_gainVal);
    addExpSlider(QStringLiteral("Iris:"), m_irisLabel, m_irisSlider, m_irisVal);

    // -- White Balance -----------------------------------------------
    m_wbModeLabel = new QLabel(QStringLiteral("White Balance:"));
    m_wbModeCombo = new QComboBox;
    m_wbModeCombo->addItems({QStringLiteral("AUTO"), QStringLiteral("MANUAL")});
    imgLay->addWidget(m_wbModeLabel, row, 0);
    imgLay->addWidget(m_wbModeCombo, row, 1, 1, 2);
    ++row;

    mainLay->addWidget(m_imagingGroup);

    // -- Buttons -----------------------------------------------------
    auto *btnLay = new QHBoxLayout;
    m_refreshBtn = new QPushButton(QStringLiteral("Refresh"));
    m_refreshBtn->setEnabled(false);
    m_applyBtn = new QPushButton(QStringLiteral("Apply"));
    m_applyBtn->setEnabled(false);
    m_closeBtn = new QPushButton(QStringLiteral("Close"));
    btnLay->addWidget(m_refreshBtn);
    btnLay->addStretch();
    btnLay->addWidget(m_applyBtn);
    btnLay->addWidget(m_closeBtn);
    mainLay->addLayout(btnLay);

    // -- Log area ---------------------------------------------------
    auto *logGroup = new QGroupBox(QStringLiteral("SOAP Log"));
    logGroup->setCheckable(true);
    logGroup->setChecked(false);
    auto *logLay = new QVBoxLayout(logGroup);
    m_logEdit = new QTextEdit;
    m_logEdit->setReadOnly(true);
    m_logEdit->setFont(QFont(QStringLiteral("Monospace"), 8));
    m_logEdit->setVisible(false);
    logLay->addWidget(m_logEdit);
    connect(logGroup, &QGroupBox::toggled, m_logEdit, &QTextEdit::setVisible);
    mainLay->addWidget(logGroup, 1);
}

// -----------------------------------------------------------------------------
void OnvifSettingsDialog::connectSignals()
{
    connect(m_connectBtn, &QPushButton::clicked, this, &OnvifSettingsDialog::onConnect);
    connect(m_applyBtn, &QPushButton::clicked, this, &OnvifSettingsDialog::onApply);
    connect(m_refreshBtn, &QPushButton::clicked, this, &OnvifSettingsDialog::onRefresh);
    connect(m_closeBtn, &QPushButton::clicked, this, &QDialog::accept);

    // ONVIF SOAP log
    connect(m_client, &OnvifClient::soapLog, this, [this](const QString &entry) {
        m_logEdit->append(entry);
    });

    // ONVIF client signals
    connect(m_client, &OnvifClient::capabilitiesReady, this, [this](const OnvifCapabilities &caps) {
        m_caps = caps;

        auto icon = [](bool ok) {
            return ok ? QStringLiteral("✓") : QStringLiteral("✗");
        };
        m_servicesLabel->setText(QStringLiteral("%1 Media  %2 Imaging  %3 PTZ  %4 Events  %5 Analytics")
                                     .arg(icon(caps.hasMedia), icon(caps.hasImaging), icon(caps.hasPTZ), icon(caps.hasEvents), icon(caps.hasAnalytics)));
        m_servicesLabel->setVisible(true);

        if (!caps.hasMedia) {
            m_statusLabel->setText(QStringLiteral("Connected - no Media service available"));
            m_statusLabel->setStyleSheet(QStringLiteral("color:orange;"));
            m_connectBtn->setEnabled(true);
            return;
        }

        m_statusLabel->setText(QStringLiteral("Fetching video sources..."));
        m_client->fetchVideoSources(caps.mediaXAddr, m_userEdit->text(), m_passEdit->text());
    });

    connect(m_client, &OnvifClient::videoSourcesReady, this, [this](const QStringList &tokens) {
        if (tokens.isEmpty()) {
            m_statusLabel->setText(QStringLiteral("No video sources found"));
            m_statusLabel->setStyleSheet(QStringLiteral("color:orange;"));
            m_connectBtn->setEnabled(true);
            return;
        }

        m_videoSourceToken = tokens.first();

        if (!m_caps.hasImaging) {
            m_statusLabel->setText(QStringLiteral("Connected - no Imaging service available"));
            m_statusLabel->setStyleSheet(QStringLiteral("color:orange;"));
            m_connectBtn->setEnabled(true);
            saveCredentials();
            return;
        }

        m_statusLabel->setText(QStringLiteral("Fetching imaging settings..."));
        m_gotSettings = false;
        m_gotOptions = false;
        m_client->fetchImagingSettings(m_caps.imagingXAddr, m_videoSourceToken, m_userEdit->text(), m_passEdit->text());
        m_client->fetchImagingOptions(m_caps.imagingXAddr, m_videoSourceToken, m_userEdit->text(), m_passEdit->text());
    });

    connect(m_client, &OnvifClient::imagingSettingsReady, this, [this](const OnvifImagingSettings &s) {
        m_currentSettings = s;
        m_gotSettings = true;

        if (m_applyRefreshPending) {
            // Post-apply verification: compare sent values with camera values
            m_applyRefreshPending = false;

            QStringList diffs;
            bool anyMismatch = false;
            auto cmp = [&](const QString &name, const std::optional<double> &sent, const std::optional<double> &got) {
                if (!sent)
                    return;
                double sv = *sent;
                QString gotStr = got ? QString::number(*got, 'f', 1) : QStringLiteral("n/a");
                double delta = got ? std::abs(sv - *got) : 9999;
                bool ok = got.has_value() && delta <= 1.5;
                diffs << QStringLiteral("%1: %2\u2192%3%4").arg(name).arg(sv, 0, 'f', 1).arg(gotStr, ok ? QString() : QStringLiteral(" \u2717"));
                if (!ok)
                    anyMismatch = true;
            };
            cmp(QStringLiteral("Brightness"), m_pendingApply.brightness, s.brightness);
            cmp(QStringLiteral("Contrast"), m_pendingApply.contrast, s.contrast);
            cmp(QStringLiteral("Saturation"), m_pendingApply.colorSaturation, s.colorSaturation);
            cmp(QStringLiteral("Sharpness"), m_pendingApply.sharpness, s.sharpness);

            QString verifyLine = diffs.join(QStringLiteral("  |  "));
            m_logEdit->append(QStringLiteral("Verify: %1").arg(verifyLine));

            // Always update sliders to what the camera actually reports
            populateImagingUI();

            // Override the status from populateImagingUI
            if (anyMismatch) {
                m_statusLabel->setText(verifyLine);
                m_statusLabel->setStyleSheet(QStringLiteral("color:orange;"));
            } else {
                m_statusLabel->setText(QStringLiteral("Settings applied \u2713  ") + verifyLine);
                m_statusLabel->setStyleSheet(QStringLiteral("color:green;"));
            }
        } else if (m_gotOptions) {
            populateImagingUI();
        }
    });

    connect(m_client, &OnvifClient::imagingOptionsReady, this, [this](const OnvifImagingOptions &o) {
        m_options = o;
        m_gotOptions = true;
        if (m_gotSettings)
            populateImagingUI();
    });

    connect(m_client, &OnvifClient::imagingSettingsApplied, this, [this]() {
        m_statusLabel->setText(QStringLiteral("Set request accepted - verifying\u2026"));
        m_statusLabel->setStyleSheet(QStringLiteral("color:gray;"));
        m_logEdit->append(QStringLiteral("\u2713 SetImagingSettings response received OK"));
        // Re-read settings from camera to verify - wait 500 ms for the
        // camera to actually commit the values before reading back.
        m_applyRefreshPending = true;
        m_gotSettings = false;
        QTimer::singleShot(500, this, [this]() {
            m_client->fetchImagingSettings(m_caps.imagingXAddr, m_videoSourceToken, m_userEdit->text(), m_passEdit->text());
        });
    });

    connect(m_client, &OnvifClient::queryFailed, this, [this](const QString &err) {
        m_statusLabel->setText(QStringLiteral("Error: %1").arg(err));
        m_statusLabel->setStyleSheet(QStringLiteral("color:red;"));
        m_logEdit->append(QStringLiteral("\u2717 Error: %1").arg(err));
        m_connectBtn->setEnabled(true);
        m_applyBtn->setEnabled(m_gotSettings && m_gotOptions);
        m_refreshBtn->setEnabled(m_gotSettings && m_gotOptions);
        m_applyRefreshPending = false;
        QMessageBox::warning(this, QStringLiteral("ONVIF Error"), err);
    });
}

// -----------------------------------------------------------------------------
void OnvifSettingsDialog::onConnect()
{
    QString host = m_hostEdit->text().trimmed();
    if (host.isEmpty()) {
        m_statusLabel->setText(QStringLiteral("Please enter a host address."));
        m_statusLabel->setStyleSheet(QStringLiteral("color:red;"));
        return;
    }

    m_connectBtn->setEnabled(false);
    m_statusLabel->setText(QStringLiteral("Connecting..."));
    m_statusLabel->setStyleSheet(QStringLiteral("color:gray;"));
    m_servicesLabel->setVisible(false);
    setImagingUIEnabled(false);

    m_gotSettings = false;
    m_gotOptions = false;

    m_client->fetchCapabilities(host, static_cast<quint16>(m_portSpin->value()), m_userEdit->text(), m_passEdit->text());
}

void OnvifSettingsDialog::onApply()
{
    if (m_videoSourceToken.isEmpty() || m_caps.imagingXAddr.isEmpty())
        return;

    m_applyBtn->setEnabled(false);
    m_statusLabel->setText(QStringLiteral("Applying settings..."));
    m_statusLabel->setStyleSheet(QStringLiteral("color:gray;"));

    OnvifImagingSettings s = gatherSettings();
    m_pendingApply = s;

    // Log what we're sending
    QStringList vals;
    if (s.brightness)
        vals << QStringLiteral("Brightness=%1").arg(*s.brightness, 0, 'f', 1);
    if (s.contrast)
        vals << QStringLiteral("Contrast=%1").arg(*s.contrast, 0, 'f', 1);
    if (s.colorSaturation)
        vals << QStringLiteral("Saturation=%1").arg(*s.colorSaturation, 0, 'f', 1);
    if (s.sharpness)
        vals << QStringLiteral("Sharpness=%1").arg(*s.sharpness, 0, 'f', 1);
    if (!s.irCutFilter.isEmpty())
        vals << QStringLiteral("IRCut=%1").arg(s.irCutFilter);
    m_logEdit->append(QStringLiteral("\u25b6 SetImagingSettings: %1").arg(vals.join(QStringLiteral(", "))));
    m_client->applyImagingSettings(m_caps.imagingXAddr, m_videoSourceToken, s, m_userEdit->text(), m_passEdit->text());
}

void OnvifSettingsDialog::onRefresh()
{
    if (m_videoSourceToken.isEmpty() || m_caps.imagingXAddr.isEmpty())
        return;

    m_refreshBtn->setEnabled(false);
    m_applyBtn->setEnabled(false);
    m_statusLabel->setText(QStringLiteral("Refreshing settings..."));
    m_statusLabel->setStyleSheet(QStringLiteral("color:gray;"));

    m_gotSettings = false;
    m_gotOptions = false;
    m_client->fetchImagingSettings(m_caps.imagingXAddr, m_videoSourceToken, m_userEdit->text(), m_passEdit->text());
    m_client->fetchImagingOptions(m_caps.imagingXAddr, m_videoSourceToken, m_userEdit->text(), m_passEdit->text());
}

// -----------------------------------------------------------------------------
void OnvifSettingsDialog::loadCredentials()
{
    // Pre-fill from the stream's RTSP URL
    StreamState st;
    StreamStateManager::instance().readState(m_streamId, [&](const StreamState &s) {
        st = s;
    });

    QUrl rtspUrl(st.rtspUrl);
    QString host = rtspUrl.host();
    QString user = rtspUrl.userName();
    QString pass = rtspUrl.password();

    // Load saved credentials for this host (overrides RTSP URL values)
    if (!host.isEmpty()) {
        QSettings settings;
        settings.beginGroup(QStringLiteral("OnvifCredentials"));
        settings.beginGroup(host);
        if (settings.contains(QStringLiteral("port"))) {
            m_portSpin->setValue(settings.value(QStringLiteral("port"), 80).toInt());
            user = settings.value(QStringLiteral("username"), user).toString();
            pass = settings.value(QStringLiteral("password"), pass).toString();
        }
        settings.endGroup();
        settings.endGroup();
    }

    m_hostEdit->setText(host);
    if (!user.isEmpty())
        m_userEdit->setText(user);
    if (!pass.isEmpty())
        m_passEdit->setText(pass);

    // Set dialog title to include camera name
    if (!st.cameraName.isEmpty())
        setWindowTitle(QStringLiteral("ONVIF Settings - %1").arg(st.cameraName));
}

void OnvifSettingsDialog::saveCredentials()
{
    QString host = m_hostEdit->text().trimmed();
    if (host.isEmpty())
        return;

    QSettings settings;
    settings.beginGroup(QStringLiteral("OnvifCredentials"));
    settings.beginGroup(host);
    settings.setValue(QStringLiteral("port"), m_portSpin->value());
    settings.setValue(QStringLiteral("username"), m_userEdit->text());
    settings.setValue(QStringLiteral("password"), m_passEdit->text());
    settings.endGroup();
    settings.endGroup();
    settings.sync();
}

// -----------------------------------------------------------------------------
void OnvifSettingsDialog::setImagingUIEnabled(bool enabled)
{
    m_imagingGroup->setEnabled(enabled);
    m_applyBtn->setEnabled(enabled);
    m_refreshBtn->setEnabled(enabled);
}

void OnvifSettingsDialog::populateImagingUI()
{
    m_connectBtn->setEnabled(true);
    saveCredentials();

    m_statusLabel->setText(QStringLiteral("Connected ✓"));
    m_statusLabel->setStyleSheet(QStringLiteral("color:green;"));

    // Helper: configure a slider row from options + current value
    auto configSlider =
        [](QLabel *label, QSlider *slider, QLabel *valLabel, const std::optional<OnvifImagingOptions::Range> &range, const std::optional<double> &value) {
            bool avail = range.has_value();
            label->setVisible(avail);
            slider->setVisible(avail);
            valLabel->setVisible(avail);
            if (avail) {
                int mn = static_cast<int>(range->min);
                int mx = static_cast<int>(range->max);
                slider->setRange(mn, mx);
                int v = value.has_value() ? static_cast<int>(*value) : mn;
                slider->setValue(v);
                valLabel->setText(QString::number(v));
            }
        };

    configSlider(m_brightnessLabel, m_brightnessSlider, m_brightnessVal, m_options.brightness, m_currentSettings.brightness);
    configSlider(m_contrastLabel, m_contrastSlider, m_contrastVal, m_options.contrast, m_currentSettings.contrast);
    configSlider(m_saturationLabel, m_saturationSlider, m_saturationVal, m_options.colorSaturation, m_currentSettings.colorSaturation);
    configSlider(m_sharpnessLabel, m_sharpnessSlider, m_sharpnessVal, m_options.sharpness, m_currentSettings.sharpness);

    // IR Cut Filter
    bool hasIR = m_options.hasIrCutFilter;
    m_irCutLabel->setVisible(hasIR);
    m_irCutCombo->setVisible(hasIR);
    if (hasIR && !m_currentSettings.irCutFilter.isEmpty()) {
        int idx = m_irCutCombo->findText(m_currentSettings.irCutFilter);
        if (idx >= 0)
            m_irCutCombo->setCurrentIndex(idx);
    }

    // Backlight
    bool hasBLC = m_options.hasBacklightComp;
    m_backlightCheck->setVisible(hasBLC);
    m_backlightSlider->setVisible(hasBLC);
    m_backlightVal->setVisible(hasBLC);
    if (hasBLC) {
        m_backlightCheck->setChecked(m_currentSettings.backlightCompEnabled);
        if (m_options.backlightLevel) {
            m_backlightSlider->setRange(static_cast<int>(m_options.backlightLevel->min), static_cast<int>(m_options.backlightLevel->max));
        }
        int v = m_currentSettings.backlightCompLevel ? static_cast<int>(*m_currentSettings.backlightCompLevel) : 0;
        m_backlightSlider->setValue(v);
        m_backlightVal->setText(QString::number(v));
    }

    // WDR
    bool hasWDR = m_options.hasWDR;
    m_wdrCheck->setVisible(hasWDR);
    m_wdrSlider->setVisible(hasWDR);
    m_wdrVal->setVisible(hasWDR);
    if (hasWDR) {
        m_wdrCheck->setChecked(m_currentSettings.wdrEnabled);
        if (m_options.wdrLevel) {
            m_wdrSlider->setRange(static_cast<int>(m_options.wdrLevel->min), static_cast<int>(m_options.wdrLevel->max));
        }
        int v = m_currentSettings.wdrLevel ? static_cast<int>(*m_currentSettings.wdrLevel) : 0;
        m_wdrSlider->setValue(v);
        m_wdrVal->setText(QString::number(v));
    }

    // Exposure
    bool hasExp = m_options.hasExposure;
    m_exposureModeLabel->setVisible(hasExp);
    m_exposureModeCombo->setVisible(hasExp);
    if (hasExp) {
        m_exposureModeCombo->clear();
        if (!m_options.exposureModes.isEmpty())
            m_exposureModeCombo->addItems(m_options.exposureModes);
        else
            m_exposureModeCombo->addItems({QStringLiteral("AUTO"), QStringLiteral("MANUAL")});
        if (!m_currentSettings.exposureMode.isEmpty()) {
            int idx = m_exposureModeCombo->findText(m_currentSettings.exposureMode);
            if (idx >= 0)
                m_exposureModeCombo->setCurrentIndex(idx);
        }
    }
    configSlider(m_exposureTimeLabel, m_exposureTimeSlider, m_exposureTimeVal, m_options.exposureTimeRange, m_currentSettings.exposureTime);
    configSlider(m_gainLabel, m_gainSlider, m_gainVal, m_options.gainRange, m_currentSettings.gain);
    configSlider(m_irisLabel, m_irisSlider, m_irisVal, m_options.irisRange, m_currentSettings.iris);

    // WhiteBalance
    bool hasWB = m_options.hasWhiteBalance;
    m_wbModeLabel->setVisible(hasWB);
    m_wbModeCombo->setVisible(hasWB);
    if (hasWB) {
        m_wbModeCombo->clear();
        if (!m_options.whiteBalanceModes.isEmpty())
            m_wbModeCombo->addItems(m_options.whiteBalanceModes);
        else
            m_wbModeCombo->addItems({QStringLiteral("AUTO"), QStringLiteral("MANUAL")});
        if (!m_currentSettings.whiteBalanceMode.isEmpty()) {
            int idx = m_wbModeCombo->findText(m_currentSettings.whiteBalanceMode);
            if (idx >= 0)
                m_wbModeCombo->setCurrentIndex(idx);
        }
    }

    setImagingUIEnabled(true);
}

OnvifImagingSettings OnvifSettingsDialog::gatherSettings() const
{
    OnvifImagingSettings s;
    s.valid = true;

    if (m_brightnessSlider->isVisible())
        s.brightness = m_brightnessSlider->value();
    if (m_contrastSlider->isVisible())
        s.contrast = m_contrastSlider->value();
    if (m_saturationSlider->isVisible())
        s.colorSaturation = m_saturationSlider->value();
    if (m_sharpnessSlider->isVisible())
        s.sharpness = m_sharpnessSlider->value();

    if (m_irCutCombo->isVisible())
        s.irCutFilter = m_irCutCombo->currentText();

    if (m_backlightCheck->isVisible()) {
        s.backlightCompEnabled = m_backlightCheck->isChecked();
        s.backlightCompLevel = m_backlightSlider->value();
    }

    if (m_wdrCheck->isVisible()) {
        s.wdrEnabled = m_wdrCheck->isChecked();
        s.wdrLevel = m_wdrSlider->value();
    }

    // Exposure - always send current mode so camera doesn't reset to defaults
    if (m_exposureModeCombo->isVisible()) {
        s.exposureMode = m_exposureModeCombo->currentText();
        // Preserve AUTO-mode limits from what the camera reported
        if (m_currentSettings.exposurePriority.size())
            s.exposurePriority = m_currentSettings.exposurePriority;
        if (m_currentSettings.minExposureTime)
            s.minExposureTime = m_currentSettings.minExposureTime;
        if (m_currentSettings.maxExposureTime)
            s.maxExposureTime = m_currentSettings.maxExposureTime;
        if (m_currentSettings.minGain)
            s.minGain = m_currentSettings.minGain;
        if (m_currentSettings.maxGain)
            s.maxGain = m_currentSettings.maxGain;
        if (m_currentSettings.minIris)
            s.minIris = m_currentSettings.minIris;
        if (m_currentSettings.maxIris)
            s.maxIris = m_currentSettings.maxIris;
    }
    if (m_exposureTimeSlider->isVisible())
        s.exposureTime = m_exposureTimeSlider->value();
    if (m_gainSlider->isVisible())
        s.gain = m_gainSlider->value();
    if (m_irisSlider->isVisible())
        s.iris = m_irisSlider->value();

    // WhiteBalance
    if (m_wbModeCombo->isVisible()) {
        s.whiteBalanceMode = m_wbModeCombo->currentText();
        if (m_currentSettings.crGain)
            s.crGain = m_currentSettings.crGain;
        if (m_currentSettings.cbGain)
            s.cbGain = m_currentSettings.cbGain;
    }

    return s;
}
