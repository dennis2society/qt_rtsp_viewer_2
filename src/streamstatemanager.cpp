#include "streamstatemanager.h"

#include <QReadLocker>
#include <QWriteLocker>

// -----------------------------------------------------------------------------
// Singleton
// -----------------------------------------------------------------------------
StreamStateManager &StreamStateManager::instance()
{
    static StreamStateManager inst;
    return inst;
}

StreamStateManager::StreamStateManager()
    : m_settings(QStringLiteral("QtRtspViewer"), QStringLiteral("QtRtspViewer"))
{
    loadSettings();
}

// -----------------------------------------------------------------------------
// Stream lifecycle
// -----------------------------------------------------------------------------
int StreamStateManager::createStream()
{
    QWriteLocker lk(&m_lock);
    int id = m_nextId++;
    StreamState s;
    s.streamId = id;
    s.cameraName = generateCameraName(id);
    m_streams.insert(id, s);
    return id;
}

void StreamStateManager::removeStream(int id)
{
    {
        QWriteLocker lk(&m_lock);
        m_streams.remove(id);
        if (m_activeStream == id)
            m_activeStream = m_streams.isEmpty() ? -1 : m_streams.firstKey();
    }
    emit streamStateChanged(id);
}

// -----------------------------------------------------------------------------
// Thread-safe access
// -----------------------------------------------------------------------------
void StreamStateManager::readState(int id, const std::function<void(const StreamState &)> &fn) const
{
    QReadLocker lk(&m_lock);
    auto it = m_streams.constFind(id);
    if (it != m_streams.cend())
        fn(*it);
}

void StreamStateManager::modifyState(int id, const std::function<void(StreamState &)> &fn)
{
    {
        QWriteLocker lk(&m_lock);
        auto it = m_streams.find(id);
        if (it == m_streams.end())
            return;
        fn(*it);
    }
    emit streamStateChanged(id);
}

StreamState StreamStateManager::stateCopy(int id) const
{
    QReadLocker lk(&m_lock);
    return m_streams.value(id);
}

bool StreamStateManager::hasStream(int id) const
{
    QReadLocker lk(&m_lock);
    return m_streams.contains(id);
}

int StreamStateManager::streamCount() const
{
    QReadLocker lk(&m_lock);
    return m_streams.size();
}

QList<int> StreamStateManager::streamIds() const
{
    QReadLocker lk(&m_lock);
    return m_streams.keys();
}

// -----------------------------------------------------------------------------
// Active stream
// -----------------------------------------------------------------------------
int StreamStateManager::activeStreamId() const
{
    return m_activeStream;
}

void StreamStateManager::setActiveStream(int id)
{
    if (m_activeStream == id)
        return;
    m_activeStream = id;
    emit activeStreamChanged(id);
}

// -----------------------------------------------------------------------------
// Global settings accessors
// -----------------------------------------------------------------------------
QString StreamStateManager::outputFolder() const
{
    return m_outputFolder;
}

void StreamStateManager::setOutputFolder(const QString &path)
{
    m_outputFolder = path;
    m_settings.setValue(QStringLiteral("OutputFolder"), path);
    m_settings.sync();
    emit globalSettingsChanged();
}

QList<StreamStateManager::UrlEntry> StreamStateManager::urlHistory() const
{
    return m_urlHistory;
}

void StreamStateManager::setUrlHistory(const QList<UrlEntry> &history)
{
    m_urlHistory = history;
    saveSettings();
    emit globalSettingsChanged();
}

void StreamStateManager::addUrlToHistory(const QString &url, const QString &cameraName)
{
    // Remove duplicate if present
    for (int i = 0; i < m_urlHistory.size(); ++i) {
        if (m_urlHistory[i].url == url) {
            m_urlHistory[i].cameraName = cameraName;
            saveSettings();
            emit globalSettingsChanged();
            return;
        }
    }
    if (m_urlHistory.size() >= MaxUrlHistory)
        m_urlHistory.removeLast();

    m_urlHistory.prepend({url, cameraName});
    saveSettings();
    emit globalSettingsChanged();
}

void StreamStateManager::removeUrlFromHistory(const QString &url)
{
    for (int i = 0; i < m_urlHistory.size(); ++i) {
        if (m_urlHistory[i].url == url) {
            m_urlHistory.removeAt(i);
            break;
        }
    }
    saveSettings();
    emit globalSettingsChanged();
}

QString StreamStateManager::lastPlayedUrl() const
{
    return m_lastPlayedUrl;
}

void StreamStateManager::setLastPlayedUrl(const QString &url)
{
    m_lastPlayedUrl = url;
    m_settings.setValue(QStringLiteral("LastPlayedUrl"), url);
    m_settings.sync();
}

QList<StreamStateManager::TabEntry> StreamStateManager::openTabs() const
{
    return m_openTabs;
}

void StreamStateManager::setOpenTabs(const QList<TabEntry> &tabs)
{
    m_openTabs = tabs;
    saveSettings();
}

int StreamStateManager::lastActiveTabIndex() const
{
    return m_lastActiveTabIndex;
}

void StreamStateManager::setLastActiveTabIndex(int index)
{
    m_lastActiveTabIndex = index;
}

// -----------------------------------------------------------------------------
// Persistence
// -----------------------------------------------------------------------------
void StreamStateManager::loadSettings()
{
    m_outputFolder = m_settings.value(QStringLiteral("OutputFolder")).toString();
    m_lastPlayedUrl = m_settings.value(QStringLiteral("LastPlayedUrl")).toString();
    m_lastActiveTabIndex = m_settings.value(QStringLiteral("LastActiveTabIndex"), 0).toInt();

    // URL history
    m_urlHistory.clear();
    m_settings.beginGroup(QStringLiteral("UrlHistory"));
    int size = m_settings.beginReadArray(QStringLiteral("urls"));
    for (int i = 0; i < size; ++i) {
        m_settings.setArrayIndex(i);
        UrlEntry e;
        e.url = m_settings.value(QStringLiteral("url")).toString();
        e.cameraName = m_settings.value(QStringLiteral("cameraName")).toString();
        if (!e.url.isEmpty())
            m_urlHistory.append(e);
    }
    m_settings.endArray();
    m_settings.endGroup();

    // Open tabs
    m_openTabs.clear();
    m_settings.beginGroup(QStringLiteral("OpenTabs"));
    size = m_settings.beginReadArray(QStringLiteral("tabs"));
    for (int i = 0; i < size; ++i) {
        m_settings.setArrayIndex(i);
        TabEntry e;
        e.url = m_settings.value(QStringLiteral("url")).toString();
        e.cameraName = m_settings.value(QStringLiteral("cameraName")).toString();
        e.blurAmount = m_settings.value(QStringLiteral("blurAmount"), 0).toInt();
        e.grayscaleEnabled = m_settings.value(QStringLiteral("grayscaleEnabled"), false).toBool();
        e.brightnessAmount = m_settings.value(QStringLiteral("brightnessAmount"), 0).toInt();
        e.contrastAmount = m_settings.value(QStringLiteral("contrastAmount"), 0).toInt();
        e.colorTemperature = m_settings.value(QStringLiteral("colorTemperature"), 0).toInt();
        e.motionDetectionEnabled = m_settings.value(QStringLiteral("motionDetectionEnabled"), false).toBool();
        e.motionSensitivity = m_settings.value(QStringLiteral("motionSensitivity"), 20).toInt();
        e.motionVectorsEnabled = m_settings.value(QStringLiteral("motionVectorsEnabled"), false).toBool();
        e.motionVectorsSensitivity = m_settings.value(QStringLiteral("motionVectorsSensitivity"), 50).toInt();
        e.motionTracesEnabled = m_settings.value(QStringLiteral("motionTracesEnabled"), false).toBool();
        e.motionTraceDecay = m_settings.value(QStringLiteral("motionTraceDecay"), 50).toInt();
        e.motionGraphEnabled = m_settings.value(QStringLiteral("motionGraphEnabled"), false).toBool();
        e.motionGraphSensitivity = m_settings.value(QStringLiteral("motionGraphSensitivity"), 50).toInt();
        e.faceDetectionEnabled = m_settings.value(QStringLiteral("faceDetectionEnabled"), false).toBool();
        e.overlayEnabled = m_settings.value(QStringLiteral("overlayEnabled"), true).toBool();
        e.motionCsvEnabled = m_settings.value(QStringLiteral("motionCsvEnabled"), false).toBool();
        e.recordCleanVideo = m_settings.value(QStringLiteral("recordCleanVideo"), false).toBool();
        e.recordCodec = m_settings.value(QStringLiteral("recordCodec"), QStringLiteral("libx264")).toString();
        e.recordFormat = m_settings.value(QStringLiteral("recordFormat"), QStringLiteral("mp4")).toString();
        e.recordFps = m_settings.value(QStringLiteral("recordFps"), 25.0).toDouble();
        e.autoRecordEnabled = m_settings.value(QStringLiteral("autoRecordEnabled"), false).toBool();
        e.autoRecordThreshold = m_settings.value(QStringLiteral("autoRecordThreshold"), 0.50).toDouble();
        e.autoRecordTimeout = m_settings.value(QStringLiteral("autoRecordTimeout"), 5).toInt();
        m_openTabs.append(e);
    }
    m_settings.endArray();
    m_settings.endGroup();
}

void StreamStateManager::saveSettings()
{
    m_settings.setValue(QStringLiteral("OutputFolder"), m_outputFolder);
    m_settings.setValue(QStringLiteral("LastPlayedUrl"), m_lastPlayedUrl);
    m_settings.setValue(QStringLiteral("LastActiveTabIndex"), m_lastActiveTabIndex);

    m_settings.beginGroup(QStringLiteral("UrlHistory"));
    m_settings.beginWriteArray(QStringLiteral("urls"), m_urlHistory.size());
    for (int i = 0; i < m_urlHistory.size(); ++i) {
        m_settings.setArrayIndex(i);
        m_settings.setValue(QStringLiteral("url"), m_urlHistory[i].url);
        m_settings.setValue(QStringLiteral("cameraName"), m_urlHistory[i].cameraName);
    }
    m_settings.endArray();
    m_settings.endGroup();

    // Open tabs
    m_settings.beginGroup(QStringLiteral("OpenTabs"));
    m_settings.beginWriteArray(QStringLiteral("tabs"), m_openTabs.size());
    for (int i = 0; i < m_openTabs.size(); ++i) {
        m_settings.setArrayIndex(i);
        m_settings.setValue(QStringLiteral("url"), m_openTabs[i].url);
        m_settings.setValue(QStringLiteral("cameraName"), m_openTabs[i].cameraName);
        m_settings.setValue(QStringLiteral("blurAmount"), m_openTabs[i].blurAmount);
        m_settings.setValue(QStringLiteral("grayscaleEnabled"), m_openTabs[i].grayscaleEnabled);
        m_settings.setValue(QStringLiteral("brightnessAmount"), m_openTabs[i].brightnessAmount);
        m_settings.setValue(QStringLiteral("contrastAmount"), m_openTabs[i].contrastAmount);
        m_settings.setValue(QStringLiteral("colorTemperature"), m_openTabs[i].colorTemperature);
        m_settings.setValue(QStringLiteral("motionDetectionEnabled"), m_openTabs[i].motionDetectionEnabled);
        m_settings.setValue(QStringLiteral("motionSensitivity"), m_openTabs[i].motionSensitivity);
        m_settings.setValue(QStringLiteral("motionVectorsEnabled"), m_openTabs[i].motionVectorsEnabled);
        m_settings.setValue(QStringLiteral("motionVectorsSensitivity"), m_openTabs[i].motionVectorsSensitivity);
        m_settings.setValue(QStringLiteral("motionTracesEnabled"), m_openTabs[i].motionTracesEnabled);
        m_settings.setValue(QStringLiteral("motionTraceDecay"), m_openTabs[i].motionTraceDecay);
        m_settings.setValue(QStringLiteral("motionGraphEnabled"), m_openTabs[i].motionGraphEnabled);
        m_settings.setValue(QStringLiteral("motionGraphSensitivity"), m_openTabs[i].motionGraphSensitivity);
        m_settings.setValue(QStringLiteral("faceDetectionEnabled"), m_openTabs[i].faceDetectionEnabled);
        m_settings.setValue(QStringLiteral("overlayEnabled"), m_openTabs[i].overlayEnabled);
        m_settings.setValue(QStringLiteral("motionCsvEnabled"), m_openTabs[i].motionCsvEnabled);
        m_settings.setValue(QStringLiteral("recordCleanVideo"), m_openTabs[i].recordCleanVideo);
        m_settings.setValue(QStringLiteral("recordCodec"), m_openTabs[i].recordCodec);
        m_settings.setValue(QStringLiteral("recordFormat"), m_openTabs[i].recordFormat);
        m_settings.setValue(QStringLiteral("recordFps"), m_openTabs[i].recordFps);
        m_settings.setValue(QStringLiteral("autoRecordEnabled"), m_openTabs[i].autoRecordEnabled);
        m_settings.setValue(QStringLiteral("autoRecordThreshold"), m_openTabs[i].autoRecordThreshold);
        m_settings.setValue(QStringLiteral("autoRecordTimeout"), m_openTabs[i].autoRecordTimeout);
    }
    m_settings.endArray();
    m_settings.endGroup();
    m_settings.sync();
}

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------
QString StreamStateManager::generateCameraName(int index) const
{
    return QStringLiteral("cam_%1").arg(index, 2, 10, QLatin1Char('0'));
}
