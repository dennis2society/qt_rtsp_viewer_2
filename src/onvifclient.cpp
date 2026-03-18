#include "onvifclient.h"

#include <QCryptographicHash>
#include <QDateTime>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QRandomGenerator>
#include <QUrl>
#include <QXmlStreamReader>

// ─────────────────────────────────────────────────────────────────────────────
OnvifClient::OnvifClient(QObject *parent)
    : QObject(parent)
    , m_nam(new QNetworkAccessManager(this))
{
    connect(m_nam, &QNetworkAccessManager::finished, this, [this](QNetworkReply *reply) {
        reply->deleteLater();

        if (reply->error() != QNetworkReply::NoError) {
            emit queryFailed(reply->errorString());
            return;
        }

        const QByteArray body = reply->readAll();
        OnvifCapabilities caps = parseResponse(body);
        if (caps.valid)
            emit capabilitiesReady(caps);
        else
            emit queryFailed(caps.error.isEmpty() ? QStringLiteral("Unknown parse error") : caps.error);
    });
}

// ─────────────────────────────────────────────────────────────────────────────
void OnvifClient::fetchCapabilities(const QString &host, quint16 port, const QString &username, const QString &password)
{
    QUrl url;
    url.setScheme(QStringLiteral("http"));
    url.setHost(host);
    url.setPort(static_cast<int>(port));
    url.setPath(QStringLiteral("/onvif/device_service"));

    QNetworkRequest req(url);
    req.setHeader(QNetworkRequest::ContentTypeHeader, QByteArrayLiteral("application/soap+xml; charset=utf-8"));

    m_nam->post(req, buildSoap(username, password));
}

// ─────────────────────────────────────────────────────────────────────────────
QByteArray OnvifClient::buildSoap(const QString &username, const QString &password) const
{
    QString secHeader;
    if (!username.isEmpty()) {
        // Generate 20 random bytes for the WS-Security nonce
        QByteArray nonceBytes;
        nonceBytes.reserve(20);
        for (int i = 0; i < 20; ++i)
            nonceBytes.append(static_cast<char>(QRandomGenerator::global()->bounded(256)));

        const QString created = QDateTime::currentDateTimeUtc().toString(QStringLiteral("yyyy-MM-ddTHH:mm:ssZ"));

        // PasswordDigest = Base64( SHA1( nonceBytes + created_utf8 + password_utf8 ) )
        const QByteArray digestRaw = QCryptographicHash::hash(nonceBytes + created.toUtf8() + password.toUtf8(), QCryptographicHash::Sha1);

        secHeader = QStringLiteral(
                        "<s:Header>"
                        "<wsse:Security s:mustUnderstand=\"1\""
                        " xmlns:wsse=\"http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd\""
                        " xmlns:wsu=\"http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-utility-1.0.xsd\">"
                        "<wsse:UsernameToken>"
                        "<wsse:Username>%1</wsse:Username>"
                        "<wsse:Password"
                        " Type=\"http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-username-token-profile-1.0#PasswordDigest\""
                        ">%2</wsse:Password>"
                        "<wsse:Nonce"
                        " EncodingType=\"http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-soap-message-security-1.0#Base64Binary\""
                        ">%3</wsse:Nonce>"
                        "<wsu:Created>%4</wsu:Created>"
                        "</wsse:UsernameToken>"
                        "</wsse:Security>"
                        "</s:Header>")
                        .arg(username, QString::fromLatin1(digestRaw.toBase64()), QString::fromLatin1(nonceBytes.toBase64()), created);
    }

    return QStringLiteral(
               "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
               "<s:Envelope"
               " xmlns:s=\"http://www.w3.org/2003/05/soap-envelope\""
               " xmlns:tds=\"http://www.onvif.org/ver10/device/wsdl\">"
               "%1"
               "<s:Body>"
               "<tds:GetCapabilities>"
               "<tds:Category>All</tds:Category>"
               "</tds:GetCapabilities>"
               "</s:Body>"
               "</s:Envelope>")
        .arg(secHeader)
        .toUtf8();
}

// ─────────────────────────────────────────────────────────────────────────────
OnvifCapabilities OnvifClient::parseResponse(const QByteArray &data)
{
    OnvifCapabilities caps;
    QXmlStreamReader xml(data);

    // Track which top-level capability block we're currently inside
    QString currentSection;

    while (!xml.atEnd()) {
        xml.readNext();

        if (xml.isStartElement()) {
            const QString local = xml.name().toString();

            // Check for SOAP Fault
            if (local == QLatin1String("Fault")) {
                while (!xml.atEnd()) {
                    xml.readNext();
                    if (xml.isStartElement() && xml.name() == QLatin1String("Text")) {
                        caps.error = xml.readElementText();
                        return caps;
                    }
                }
                caps.error = QStringLiteral("SOAP Fault received");
                return caps;
            }

            // Enter a capability block
            if (local == QLatin1String("Analytics") || local == QLatin1String("Device") || local == QLatin1String("Events") || local == QLatin1String("Imaging")
                || local == QLatin1String("Media") || local == QLatin1String("PTZ")) {
                currentSection = local;
            }
            // Capture the service endpoint URL for the current block
            else if (local == QLatin1String("XAddr") && !currentSection.isEmpty()) {
                const QString xaddr = xml.readElementText();
                if (currentSection == QLatin1String("Analytics")) {
                    caps.hasAnalytics = true;
                    caps.analyticsXAddr = xaddr;
                } else if (currentSection == QLatin1String("Device")) {
                    caps.deviceXAddr = xaddr;
                } else if (currentSection == QLatin1String("Events")) {
                    caps.hasEvents = true;
                    caps.eventsXAddr = xaddr;
                } else if (currentSection == QLatin1String("Imaging")) {
                    caps.hasImaging = true;
                    caps.imagingXAddr = xaddr;
                } else if (currentSection == QLatin1String("Media")) {
                    caps.hasMedia = true;
                    caps.mediaXAddr = xaddr;
                } else if (currentSection == QLatin1String("PTZ")) {
                    caps.hasPTZ = true;
                    caps.ptzXAddr = xaddr;
                }
            }
        } else if (xml.isEndElement()) {
            if (xml.name().toString() == currentSection)
                currentSection.clear();
        }
    }

    if (xml.hasError()) {
        caps.error = QStringLiteral("XML parse error: ") + xml.errorString();
        return caps;
    }

    caps.valid = true;
    return caps;
}
