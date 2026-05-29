#pragma once

#include <QMainWindow>
#include <QMap>

class QScrollArea;
class QTabWidget;
class QPushButton;
class EffectsSidebar;
class StreamTab;
class FilePlayerTab;

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow() override;

private slots:
    void addNewTab();
    void closeTab(int index);
    void onCurrentTabChanged(int index);
    void onTabTitleChanged(int streamId, const QString &title);
    void openFilePlayerTab();
    void updateTabIcon(int streamId);

private:
    int streamIdForTab(int index) const;
    static QIcon makeStatusIcon(const QColor &color);

    /// Mute all StreamTab audio outputs except the one with playingStreamId.
    void autoMuteOtherTabs(int playingStreamId);
    /// Remove auto-mute from all StreamTabs (lets user-mute state take over).
    void clearAllAutoMutes();

    QTabWidget *m_tabs = nullptr;
    EffectsSidebar *m_sidebar = nullptr;
    QScrollArea *m_sidebarScroll = nullptr;
    QPushButton *m_addTabBtn = nullptr;
    QPushButton *m_openFileBtn = nullptr;

    // Saved scroll positions per stream id (so they survive tab switches)
    QMap<int, int> m_sidebarScrollPos;
};
