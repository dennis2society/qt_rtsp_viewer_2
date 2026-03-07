#pragma once

#include <QMainWindow>

class QTabWidget;
class QPushButton;
class EffectsSidebar;
class StreamTab;

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

private:
    int streamIdForTab(int index) const;

    QTabWidget *m_tabs = nullptr;
    EffectsSidebar *m_sidebar = nullptr;
    QPushButton *m_addTabBtn = nullptr; // corner widget
};
