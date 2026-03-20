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
}

// ─────────────────────────────────────────────────────────────────────────────
// SOAP helpers
// ─────────────────────────────────────────────────────────────────────────────
QByteArray OnvifClient::buildSecurityHeader(const QString &user, const QString &pass) const
{
    if (user.isEmpty())
        return {};

    QByteArray nonceBytes;
    nonceBytes.reserve(20);
    for (int i = 0; i < 20; ++i)
        nonceBytes.append(static_cast<char>(QRandomGenerator::global()->bounded(256)));

    const QString created = QDateTime::currentDateTimeUtc().toString(QStringLiteral("yyyy-MM-ddTHH:mm:ssZ"));
    const QByteArray digestRaw = QCryptographicHash::hash(nonceBytes + created.toUtf8() + pass.toUtf8(), QCryptographicHash::Sha1);

    return QStringLiteral(
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
        .arg(user, QString::fromLatin1(digestRaw.toBase64()), QString::fromLatin1(nonceBytes.toBase64()), created)
        .toUtf8();
}

QByteArray OnvifClient::wrapSoap(const QString &body, const QString &user, const QString &pass) const
{
    QString secHeader = QString::fromUtf8(buildSecurityHeader(user, pass));
    return QStringLiteral(
               "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
               "<s:Envelope"
               " xmlns:s=\"http://www.w3.org/2003/05/soap-envelope\""
               " xmlns:tds=\"http://www.onvif.org/ver10/device/wsdl\""
               " xmlns:trt=\"http://www.onvif.org/ver10/media/wsdl\""
               " xmlns:timg=\"http://www.onvif.org/ver20/imaging/wsdl\""
               " xmlns:tt=\"http://www.onvif.org/ver10/schema\">"
               "%1"
               "<s:Body>%2</s:Body>"
               "</s:Envelope>")
        .arg(secHeader, body)
        .toUtf8();
}

void OnvifClient::postSoap(const QUrl &url, const QByteArray &soap, std::function<void(const QByteArray &)> onSuccess)
{
    QNetworkRequest req(url);
    req.setHeader(QNetworkRequest::ContentTypeHeader, QByteArrayLiteral("application/soap+xml; charset=utf-8"));
    auto *reply = m_nam->post(req, soap);
    connect(reply, &QNetworkReply::finished, this, [this, reply, onSuccess = std::move(onSuccess)]() {
        reply->deleteLater();
        if (reply->error() != QNetworkReply::NoError) {
            emit queryFailed(reply->errorString());
            return;
        }
        const QByteArray body = reply->readAll();
        QString fault = parseSoapFault(body);
        if (!fault.isEmpty()) {
            emit queryFailed(fault);
            return;
        }
        onSuccess(body);
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────
void OnvifClient::fetchCapabilities(const QString &host, quint16 port, const QString &user, const QString &pass)
{
    QUrl url;
    url.setScheme(QStringLiteral("http"));
    url.setHost(host);
    url.setPort(static_cast<int>(port));
    url.setPath(QStringLiteral("/onvif/device_service"));

    QByteArray soap = wrapSoap(QStringLiteral("<tds:GetCapabilities><tds:Category>All</tds:Category></tds:GetCapabilities>"), user, pass);

    postSoap(url, soap, [this](const QByteArray &data) {
        OnvifCapabilities caps = parseCapabilities(data);
        if (caps.valid)
            emit capabilitiesReady(caps);
        else
            emit queryFailed(caps.error.isEmpty() ? QStringLiteral("Unknown parse error") : caps.error);
    });
}

void OnvifClient::fetchVideoSources(const QString &mediaXAddr, const QString &user, const QString &pass)
{
    QByteArray soap = wrapSoap(QStringLiteral("<trt:GetVideoSources/>"), user, pass);
    postSoap(QUrl(mediaXAddr), soap, [this](const QByteArray &data) {
        QStringList tokens = parseVideoSources(data);
        emit videoSourcesReady(tokens);
    });
}

void OnvifClient::fetchImagingSettings(const QString &imagingXAddr, const QString &token, const QString &user, const QString &pass)
{
    QByteArray soap = wrapSoap(QStringLiteral("<timg:GetImagingSettings>"
                                              "<timg:VideoSourceToken>%1</timg:VideoSourceToken>"
                                              "</timg:GetImagingSettings>")
                                   .arg(token.toHtmlEscaped()),
                               user,
                               pass);
    postSoap(QUrl(imagingXAddr), soap, [this](const QByteArray &data) {
        OnvifImagingSettings s = parseImagingSettings(data);
        if (s.valid)
            emit imagingSettingsReady(s);
        else
            emit queryFailed(QStringLiteral("Failed to parse imaging settings"));
    });
}

void OnvifClient::fetchImagingOptions(const QString &imagingXAddr, const QString &token, const QString &user, const QString &pass)
{
    QByteArray soap = wrapSoap(QStringLiteral("<timg:GetOptions>"
                                              "<timg:VideoSourceToken>%1</timg:VideoSourceToken>"
                                              "</timg:GetOptions>")
                                   .arg(token.toHtmlEscaped()),
                               user,
                               pass);
    postSoap(QUrl(imagingXAddr), soap, [this](const QByteArray &data) {
        OnvifImagingOptions o = parseImagingOptions(data);
        if (o.valid)
            emit imagingOptionsReady(o);
        else
            emit queryFailed(QStringLiteral("Failed to parse imaging options"));
    });
}

void OnvifClient::applyImagingSettings(const QString &imagingXAddr,
                                       const QString &token,
                                       const OnvifImagingSettings &settings,
                                       const QString &user,
                                       const QString &pass)
{
    QString inner;
    if (settings.brightness)
        inner += QStringLiteral("<tt:Brightness>%1</tt:Brightness>").arg(*settings.brightness);
    if (settings.colorSaturation)
        inner += QStringLiteral("<tt:ColorSaturation>%1</tt:ColorSaturation>").arg(*settings.colorSaturation);
    if (settings.contrast)
        inner += QStringLiteral("<tt:Contrast>%1</tt:Contrast>").arg(*settings.contrast);
    if (settings.sharpness)
        inner += QStringLiteral("<tt:Sharpness>%1</tt:Sharpness>").arg(*settings.sharpness);
    if (!settings.irCutFilter.isEmpty())
        inner += QStringLiteral("<tt:IrCutFilter>%1</tt:IrCutFilter>").arg(settings.irCutFilter);
    if (settings.backlightCompLevel) {
        inner += QStringLiteral(
                     "<tt:BacklightCompensation>"
                     "<tt:Mode>%1</tt:Mode>"
                     "<tt:Level>%2</tt:Level>"
                     "</tt:BacklightCompensation>")
                     .arg(settings.backlightCompEnabled ? QStringLiteral("ON") : QStringLiteral("OFF"))
                     .arg(*settings.backlightCompLevel);
    }
    if (settings.wdrLevel) {
        inner += QStringLiteral(
                     "<tt:WideDynamicRange>"
                     "<tt:Mode>%1</tt:Mode>"
                     "<tt:Level>%2</tt:Level>"
                     "</tt:WideDynamicRange>")
                     .arg(settings.wdrEnabled ? QStringLiteral("ON") : QStringLiteral("OFF"))
                     .arg(*settings.wdrLevel);
    }

    QString body = QStringLiteral(
                       "<timg:SetImagingSettings>"
                       "<timg:VideoSourceToken>%1</timg:VideoSourceToken>"
                       "<timg:ImagingSettings>%2</timg:ImagingSettings>"
                       "</timg:SetImagingSettings>")
                       .arg(token.toHtmlEscaped(), inner);

    QByteArray soap = wrapSoap(body, user, pass);
    postSoap(QUrl(imagingXAddr), soap, [this](const QByteArray &) {
        emit imagingSettingsApplied();
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// Parsers
// ─────────────────────────────────────────────────────────────────────────────
QString OnvifClient::parseSoapFault(const QByteArray &data)
{
    QXmlStreamReader xml(data);
    while (!xml.atEnd()) {
        xml.readNext();
        if (xml.isStartElement() && xml.name() == QLatin1String("Fault")) {
            while (!xml.atEnd()) {
                xml.readNext();
                if (xml.isStartElement() && xml.name() == QLatin1String("Text"))
                    return xml.readElementText();
            }
            return QStringLiteral("SOAP Fault");
        }
    }
    return {};
}

OnvifCapabilities OnvifClient::parseCapabilities(const QByteArray &data)
{
    OnvifCapabilities caps;
    QXmlStreamReader xml(data);
    QString currentSection;

    while (!xml.atEnd()) {
        xml.readNext();

        if (xml.isStartElement()) {
            const QString local = xml.name().toString();

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

            if (local == QLatin1String("Analytics") || local == QLatin1String("Device") || local == QLatin1String("Events") || local == QLatin1String("Imaging")
                || local == QLatin1String("Media") || local == QLatin1String("PTZ")) {
                currentSection = local;
            } else if (local == QLatin1String("XAddr") && !currentSection.isEmpty()) {
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

QStringList OnvifClient::parseVideoSources(const QByteArray &data)
{
    QStringList tokens;
    QXmlStreamReader xml(data);
    while (!xml.atEnd()) {
        xml.readNext();
        if (xml.isStartElement() && xml.name() == QLatin1String("VideoSources")) {
            QString token = xml.attributes().value(QLatin1String("token")).toString();
            if (!token.isEmpty())
                tokens.append(token);
        }
    }
    return tokens;
}

OnvifImagingSettings OnvifClient::parseImagingSettings(const QByteArray &data)
{
    OnvifImagingSettings s;
    QXmlStreamReader xml(data);
    bool inImaging = false;
    bool inBacklight = false;
    bool inWDR = false;

    while (!xml.atEnd()) {
        xml.readNext();
        if (xml.isStartElement()) {
            QString name = xml.name().toString();
            if (name == QLatin1String("ImagingSettings")) {
                inImaging = true;
            } else if (inImaging) {
                if (name == QLatin1String("BacklightCompensation"))
                    inBacklight = true;
                else if (name == QLatin1String("WideDynamicRange"))
                    inWDR = true;
                else if (inBacklight) {
                    if (name == QLatin1String("Mode"))
                        s.backlightCompEnabled = (xml.readElementText() == QLatin1String("ON"));
                    else if (name == QLatin1String("Level"))
                        s.backlightCompLevel = xml.readElementText().toDouble();
                } else if (inWDR) {
                    if (name == QLatin1String("Mode"))
                        s.wdrEnabled = (xml.readElementText() == QLatin1String("ON"));
                    else if (name == QLatin1String("Level"))
                        s.wdrLevel = xml.readElementText().toDouble();
                } else if (name == QLatin1String("Brightness"))
                    s.brightness = xml.readElementText().toDouble();
                else if (name == QLatin1String("ColorSaturation"))
                    s.colorSaturation = xml.readElementText().toDouble();
                else if (name == QLatin1String("Contrast"))
                    s.contrast = xml.readElementText().toDouble();
                else if (name == QLatin1String("Sharpness"))
                    s.sharpness = xml.readElementText().toDouble();
                else if (name == QLatin1String("IrCutFilter"))
                    s.irCutFilter = xml.readElementText();
            }
        } else if (xml.isEndElement()) {
            QString name = xml.name().toString();
            if (name == QLatin1String("BacklightCompensation"))
                inBacklight = false;
            else if (name == QLatin1String("WideDynamicRange"))
                inWDR = false;
            else if (name == QLatin1String("ImagingSettings"))
                inImaging = false;
        }
    }

    s.valid = !xml.hasError();
    return s;
}

OnvifImagingOptions OnvifClient::parseImagingOptions(const QByteArray &data)
{
    OnvifImagingOptions o;
    QXmlStreamReader xml(data);
    bool inOptions = false;
    QString currentRange;
    double rangeMin = 0, rangeMax = 100;
    // Nested elements inside BacklightCompensation/WideDynamicRange level
    bool inBLC = false, inWDR = false, inLevel = false;
    double levelMin = 0, levelMax = 100;

    while (!xml.atEnd()) {
        xml.readNext();
        if (xml.isStartElement()) {
            QString name = xml.name().toString();
            if (name == QLatin1String("ImagingOptions") || name == QLatin1String("ImagingOptions20")) {
                inOptions = true;
            } else if (inOptions) {
                if (!inBLC && !inWDR && !inLevel && currentRange.isEmpty()) {
                    if (name == QLatin1String("Brightness") || name == QLatin1String("ColorSaturation") || name == QLatin1String("Contrast")
                        || name == QLatin1String("Sharpness")) {
                        currentRange = name;
                        rangeMin = 0;
                        rangeMax = 100;
                    } else if (name == QLatin1String("BacklightCompensation")) {
                        inBLC = true;
                        o.hasBacklightComp = true;
                    } else if (name == QLatin1String("WideDynamicRange")) {
                        inWDR = true;
                        o.hasWDR = true;
                    } else if (name == QLatin1String("IrCutFilterModes")) {
                        o.hasIrCutFilter = true;
                    }
                } else if ((inBLC || inWDR) && name == QLatin1String("Level")) {
                    inLevel = true;
                    levelMin = 0;
                    levelMax = 100;
                } else if (inLevel) {
                    if (name == QLatin1String("Min"))
                        levelMin = xml.readElementText().toDouble();
                    else if (name == QLatin1String("Max"))
                        levelMax = xml.readElementText().toDouble();
                } else if (!currentRange.isEmpty()) {
                    if (name == QLatin1String("Min"))
                        rangeMin = xml.readElementText().toDouble();
                    else if (name == QLatin1String("Max"))
                        rangeMax = xml.readElementText().toDouble();
                }
            }
        } else if (xml.isEndElement()) {
            QString name = xml.name().toString();
            if (inLevel && name == QLatin1String("Level")) {
                OnvifImagingOptions::Range r{levelMin, levelMax};
                if (inBLC)
                    o.backlightLevel = r;
                else if (inWDR)
                    o.wdrLevel = r;
                inLevel = false;
            } else if (name == QLatin1String("BacklightCompensation")) {
                inBLC = false;
            } else if (name == QLatin1String("WideDynamicRange")) {
                inWDR = false;
            } else if (!currentRange.isEmpty() && name == currentRange) {
                OnvifImagingOptions::Range r{rangeMin, rangeMax};
                if (currentRange == QLatin1String("Brightness"))
                    o.brightness = r;
                else if (currentRange == QLatin1String("ColorSaturation"))
                    o.colorSaturation = r;
                else if (currentRange == QLatin1String("Contrast"))
                    o.contrast = r;
                else if (currentRange == QLatin1String("Sharpness"))
                    o.sharpness = r;
                currentRange.clear();
            } else if (name == QLatin1String("ImagingOptions") || name == QLatin1String("ImagingOptions20")) {
                inOptions = false;
            }
        }
    }

    o.valid = !xml.hasError();
    return o;
}
