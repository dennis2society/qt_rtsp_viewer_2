#pragma once

#include <QObject>
#include <QString>
#include <QStringList>
#include <functional>
#include <optional>

class QNetworkAccessManager;

/// Capabilities reported by an ONVIF-compliant camera device.
struct OnvifCapabilities {
    bool valid = false;
    QString error;

    // Which ONVIF services the device exposes
    bool hasAnalytics = false;
    bool hasEvents = false;
    bool hasImaging = false;
    bool hasMedia = false;
    bool hasPTZ = false;

    // Service endpoint URLs
    QString deviceXAddr;
    QString analyticsXAddr;
    QString eventsXAddr;
    QString imagingXAddr;
    QString mediaXAddr;
    QString ptzXAddr;
};

/// Current imaging settings reported by the camera.
struct OnvifImagingSettings {
    bool valid = false;
    std::optional<double> brightness;
    std::optional<double> colorSaturation;
    std::optional<double> contrast;
    std::optional<double> sharpness;
    QString irCutFilter; // "ON", "OFF", "AUTO"
    bool backlightCompEnabled = false;
    std::optional<double> backlightCompLevel;
    bool wdrEnabled = false;
    std::optional<double> wdrLevel;

    // Exposure
    QString exposureMode; // "AUTO", "MANUAL"
    QString exposurePriority; // "LowNoise", "FrameRate"
    std::optional<double> exposureTime;
    std::optional<double> gain;
    std::optional<double> iris;
    std::optional<double> minExposureTime;
    std::optional<double> maxExposureTime;
    std::optional<double> minGain;
    std::optional<double> maxGain;
    std::optional<double> minIris;
    std::optional<double> maxIris;

    // WhiteBalance
    QString whiteBalanceMode; // "AUTO", "MANUAL"
    std::optional<double> crGain;
    std::optional<double> cbGain;
};

/// Allowable ranges for imaging settings.
struct OnvifImagingOptions {
    bool valid = false;
    struct Range {
        double min = 0;
        double max = 100;
    };
    std::optional<Range> brightness;
    std::optional<Range> colorSaturation;
    std::optional<Range> contrast;
    std::optional<Range> sharpness;
    bool hasIrCutFilter = false;
    bool hasBacklightComp = false;
    std::optional<Range> backlightLevel;
    bool hasWDR = false;
    std::optional<Range> wdrLevel;

    // Exposure options
    bool hasExposure = false;
    QStringList exposureModes; // e.g. {"AUTO", "MANUAL"}
    std::optional<Range> exposureTimeRange;
    std::optional<Range> gainRange;
    std::optional<Range> irisRange;
    std::optional<Range> minExposureTimeRange;
    std::optional<Range> maxExposureTimeRange;
    std::optional<Range> minGainRange;
    std::optional<Range> maxGainRange;

    // WhiteBalance options
    bool hasWhiteBalance = false;
    QStringList whiteBalanceModes; // e.g. {"AUTO", "MANUAL"}
    std::optional<Range> crGainRange;
    std::optional<Range> cbGainRange;
};

/// Queries an ONVIF device for its GetCapabilities response via SOAP 1.2
/// over HTTP.  All I/O is asynchronous; results arrive via signals.
class OnvifClient : public QObject
{
    Q_OBJECT

public:
    explicit OnvifClient(QObject *parent = nullptr);

    void fetchCapabilities(const QString &host, quint16 port, const QString &username, const QString &password);
    void fetchVideoSources(const QString &mediaXAddr, const QString &user, const QString &pass);
    void fetchImagingSettings(const QString &imagingXAddr, const QString &token, const QString &user, const QString &pass);
    void fetchImagingOptions(const QString &imagingXAddr, const QString &token, const QString &user, const QString &pass);
    void
    applyImagingSettings(const QString &imagingXAddr, const QString &token, const OnvifImagingSettings &settings, const QString &user, const QString &pass);

signals:
    void capabilitiesReady(const OnvifCapabilities &caps);
    void videoSourcesReady(const QStringList &tokens);
    void imagingSettingsReady(const OnvifImagingSettings &settings);
    void imagingOptionsReady(const OnvifImagingOptions &options);
    void imagingSettingsApplied();
    void queryFailed(const QString &errorMessage);
    void soapLog(const QString &entry);

private:
    QByteArray buildSecurityHeader(const QString &user, const QString &pass) const;
    QByteArray wrapSoap(const QString &body, const QString &user, const QString &pass) const;
    void postSoap(const QUrl &url, const QByteArray &soap, const QByteArray &soapAction, std::function<void(const QByteArray &)> onSuccess);

    static OnvifCapabilities parseCapabilities(const QByteArray &data);
    static QStringList parseVideoSources(const QByteArray &data);
    static OnvifImagingSettings parseImagingSettings(const QByteArray &data);
    static OnvifImagingOptions parseImagingOptions(const QByteArray &data);
    static QString parseSoapFault(const QByteArray &data);

    QNetworkAccessManager *m_nam;
};
