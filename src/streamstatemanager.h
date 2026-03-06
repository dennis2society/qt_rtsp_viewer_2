#pragma once

#include "streamstate.h"

#include <QObject>
#include <QReadWriteLock>
#include <QSettings>
#include <QMap>
#include <QList>
#include <QPair>
#include <functional>

/// Thread-safe singleton that owns every StreamState and the global settings.
/// Worker threads call readState(); the UI thread calls modifyState().
class StreamStateManager : public QObject {
    Q_OBJECT

public:
    static StreamStateManager &instance();

    // ── stream lifecycle ────────────────────────────────────────────
    int  createStream();                        ///< returns new streamId
    void removeStream(int id);

    // ── thread-safe state access ────────────────────────────────────
    /// Read-only snapshot – acquires a shared (read) lock.
    void readState(int id, const std::function<void(const StreamState &)> &fn) const;

    /// Read-write access – acquires an exclusive (write) lock.
    /// Emits streamStateChanged(id) after the functor returns.
    void modifyState(int id, const std::function<void(StreamState &)> &fn);

    /// Convenience: get a copy (read-locked).
    StreamState stateCopy(int id) const;

    bool hasStream(int id) const;
    int  streamCount() const;
    QList<int> streamIds() const;

    // ── active stream (UI concept) ──────────────────────────────────
    int  activeStreamId() const;
    void setActiveStream(int id);

    // ── global settings ─────────────────────────────────────────────
    QString outputFolder() const;
    void    setOutputFolder(const QString &path);

    struct UrlEntry { QString url; QString cameraName; };
    QList<UrlEntry> urlHistory() const;
    void setUrlHistory(const QList<UrlEntry> &history);
    void addUrlToHistory(const QString &url, const QString &cameraName);
    void removeUrlFromHistory(const QString &url);
    QString lastPlayedUrl() const;
    void    setLastPlayedUrl(const QString &url);

    /// Open-tab persistence
    struct TabEntry { QString url; QString cameraName; };
    QList<TabEntry> openTabs() const;
    void setOpenTabs(const QList<TabEntry> &tabs);

    static constexpr int MaxTabs = 4;
    static constexpr int MaxUrlHistory = 20;

    // ── persistence ─────────────────────────────────────────────────
    void loadSettings();
    void saveSettings();

signals:
    void streamStateChanged(int streamId);
    void activeStreamChanged(int streamId);
    void globalSettingsChanged();

private:
    StreamStateManager();
    ~StreamStateManager() override = default;
    StreamStateManager(const StreamStateManager &) = delete;
    StreamStateManager &operator=(const StreamStateManager &) = delete;

    QString generateCameraName(int index) const;

    mutable QReadWriteLock  m_lock;
    QMap<int, StreamState>  m_streams;
    int                     m_nextId       = 1;
    int                     m_activeStream = -1;

    // global settings
    QString                 m_outputFolder;
    QList<UrlEntry>         m_urlHistory;
    QList<TabEntry>         m_openTabs;
    QString                 m_lastPlayedUrl;

    QSettings               m_settings;
};
