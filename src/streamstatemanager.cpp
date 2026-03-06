#include "streamstatemanager.h"

#include <QReadLocker>
#include <QWriteLocker>

// ─────────────────────────────────────────────────────────────────────────────
// Singleton
// ─────────────────────────────────────────────────────────────────────────────
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

// ─────────────────────────────────────────────────────────────────────────────
// Stream lifecycle
// ─────────────────────────────────────────────────────────────────────────────
int StreamStateManager::createStream()
{
    QWriteLocker lk(&m_lock);
    int id = m_nextId++;
    StreamState s;
    s.streamId   = id;
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

// ─────────────────────────────────────────────────────────────────────────────
// Thread-safe access
// ─────────────────────────────────────────────────────────────────────────────
void StreamStateManager::readState(int id,
                                   const std::function<void(const StreamState &)> &fn) const
{
    QReadLocker lk(&m_lock);
    auto it = m_streams.constFind(id);
    if (it != m_streams.cend())
        fn(*it);
}

void StreamStateManager::modifyState(int id,
                                     const std::function<void(StreamState &)> &fn)
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

// ─────────────────────────────────────────────────────────────────────────────
// Active stream
// ─────────────────────────────────────────────────────────────────────────────
int StreamStateManager::activeStreamId() const
{
    return m_activeStream;
}

void StreamStateManager::setActiveStream(int id)
{
    if (m_activeStream == id) return;
    m_activeStream = id;
    emit activeStreamChanged(id);
}

// ─────────────────────────────────────────────────────────────────────────────
// Global settings accessors
// ─────────────────────────────────────────────────────────────────────────────
QString StreamStateManager::outputFolder() const { return m_outputFolder; }

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

QString StreamStateManager::lastPlayedUrl() const { return m_lastPlayedUrl; }

void StreamStateManager::setLastPlayedUrl(const QString &url)
{
    m_lastPlayedUrl = url;
    m_settings.setValue(QStringLiteral("LastPlayedUrl"), url);
    m_settings.sync();
}

// ─────────────────────────────────────────────────────────────────────────────
// Persistence
// ─────────────────────────────────────────────────────────────────────────────
void StreamStateManager::loadSettings()
{
    m_outputFolder  = m_settings.value(QStringLiteral("OutputFolder")).toString();
    m_lastPlayedUrl = m_settings.value(QStringLiteral("LastPlayedUrl")).toString();

    // URL history
    m_urlHistory.clear();
    m_settings.beginGroup(QStringLiteral("UrlHistory"));
    int size = m_settings.beginReadArray(QStringLiteral("urls"));
    for (int i = 0; i < size; ++i) {
        m_settings.setArrayIndex(i);
        UrlEntry e;
        e.url        = m_settings.value(QStringLiteral("url")).toString();
        e.cameraName = m_settings.value(QStringLiteral("cameraName")).toString();
        if (!e.url.isEmpty())
            m_urlHistory.append(e);
    }
    m_settings.endArray();
    m_settings.endGroup();
}

void StreamStateManager::saveSettings()
{
    m_settings.setValue(QStringLiteral("OutputFolder"), m_outputFolder);
    m_settings.setValue(QStringLiteral("LastPlayedUrl"), m_lastPlayedUrl);

    m_settings.beginGroup(QStringLiteral("UrlHistory"));
    m_settings.beginWriteArray(QStringLiteral("urls"), m_urlHistory.size());
    for (int i = 0; i < m_urlHistory.size(); ++i) {
        m_settings.setArrayIndex(i);
        m_settings.setValue(QStringLiteral("url"),        m_urlHistory[i].url);
        m_settings.setValue(QStringLiteral("cameraName"), m_urlHistory[i].cameraName);
    }
    m_settings.endArray();
    m_settings.endGroup();
    m_settings.sync();
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────
QString StreamStateManager::generateCameraName(int index) const
{
    return QStringLiteral("cam_%1").arg(index, 2, 10, QLatin1Char('0'));
}
