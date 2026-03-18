#pragma once

#include <QObject>
#include <QString>

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

/// Queries an ONVIF device for its GetCapabilities response via SOAP 1.2
/// over HTTP.  All I/O is asynchronous; results arrive via signals.
class OnvifClient : public QObject
{
    Q_OBJECT

public:
    explicit OnvifClient(QObject *parent = nullptr);

    /// Start an async GetCapabilities query.
    /// Emits capabilitiesReady() on success or queryFailed() on error.
    void fetchCapabilities(const QString &host, quint16 port, const QString &username, const QString &password);

signals:
    void capabilitiesReady(const OnvifCapabilities &caps);
    void queryFailed(const QString &errorMessage);

private:
    /// Build the SOAP 1.2 GetCapabilities request body.
    /// Includes a WS-Security PasswordDigest header when username is non-empty.
    QByteArray buildSoap(const QString &username, const QString &password) const;

    /// Parse a raw SOAP response body into an OnvifCapabilities struct.
    static OnvifCapabilities parseResponse(const QByteArray &data);

    QNetworkAccessManager *m_nam;
};
