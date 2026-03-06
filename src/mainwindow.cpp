#include "mainwindow.h"
#include "streamtab.h"
#include "effectssidebar.h"
#include "streamstatemanager.h"

#include <QComboBox>
#include <QLineEdit>
#include <QTabWidget>
#include <QTabBar>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QScrollArea>
#include <QPushButton>
#include <QStatusBar>
#include <QLabel>
#include <QCloseEvent>
#include <QTimer>

// ─────────────────────────────────────────────────────────────────────────────
MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent)
{
    setWindowTitle(QStringLiteral("RTSP Stream Viewer"));
    resize(1200, 720);

    // ── Central widget ──────────────────────────────────────────────
    auto *central = new QWidget;
    auto *hlay    = new QHBoxLayout(central);
    hlay->setContentsMargins(4, 4, 4, 4);

    // Tab widget
    m_tabs = new QTabWidget;
    m_tabs->setTabsClosable(true);
    m_tabs->setMovable(true);

    // "+" corner button
    m_addTabBtn = new QPushButton(QStringLiteral("+"));
    m_addTabBtn->setToolTip(QStringLiteral("Add new stream tab"));
    m_addTabBtn->setFixedSize(28, 28);
    m_tabs->setCornerWidget(m_addTabBtn, Qt::TopRightCorner);

    hlay->addWidget(m_tabs, 1);

    // Sidebar in scroll area
    auto *scroll = new QScrollArea;
    scroll->setWidgetResizable(true);
    scroll->setMinimumWidth(210);
    scroll->setMaximumWidth(280);
    m_sidebar = new EffectsSidebar;
    scroll->setWidget(m_sidebar);
    hlay->addWidget(scroll);

    setCentralWidget(central);

    // Status bar
    if (!statusBar())
        setStatusBar(new QStatusBar);

    // ── Connections ─────────────────────────────────────────────────
    connect(m_addTabBtn, &QPushButton::clicked, this, &MainWindow::addNewTab);
    connect(m_tabs, &QTabWidget::tabCloseRequested, this, &MainWindow::closeTab);
    connect(m_tabs, &QTabWidget::currentChanged,    this, &MainWindow::onCurrentTabChanged);

    // When active stream changes, rebind sidebar
    connect(&StreamStateManager::instance(), &StreamStateManager::activeStreamChanged,
            this, [this](int streamId) {
                m_sidebar->bindToStream(streamId);
            });

    // ── Create first tab ────────────────────────────────────────────
    addNewTab();

    // Auto-play last URL in the first tab
    QString lastUrl = StreamStateManager::instance().lastPlayedUrl();
    if (!lastUrl.isEmpty()) {
        auto *tab = qobject_cast<StreamTab *>(m_tabs->widget(0));
        if (tab) {
            // Set the URL in the combo and trigger play after a short delay
            // (allows the event loop to settle)
            QTimer::singleShot(500, this, [tab, lastUrl]() {
                auto *combo = tab->findChild<QComboBox *>();
                if (combo && combo->lineEdit())
                    combo->lineEdit()->setText(lastUrl);
            });
        }
    }
}

MainWindow::~MainWindow()
{
    // Shut down all tabs
    for (int i = 0; i < m_tabs->count(); ++i) {
        auto *tab = qobject_cast<StreamTab *>(m_tabs->widget(i));
        if (tab) tab->shutDown();
    }
    StreamStateManager::instance().saveSettings();
}

// ─────────────────────────────────────────────────────────────────────────────
// Tab management
// ─────────────────────────────────────────────────────────────────────────────
void MainWindow::addNewTab()
{
    if (m_tabs->count() >= StreamStateManager::MaxTabs) return;

    int id = StreamStateManager::instance().createStream();
    auto *tab = new StreamTab(id, this);

    StreamState st = StreamStateManager::instance().stateCopy(id);
    int idx = m_tabs->addTab(tab, st.cameraName);
    m_tabs->setCurrentIndex(idx);

    // Wire tab signals
    connect(tab, &StreamTab::tabTitleChanged, this, &MainWindow::onTabTitleChanged);
    connect(tab, &StreamTab::statusMessage, this, [this](const QString &msg) {
        statusBar()->showMessage(msg, 5000);
    });

    // Enable / disable "+"
    m_addTabBtn->setEnabled(m_tabs->count() < StreamStateManager::MaxTabs);

    // Disable close when only 1 tab
    m_tabs->tabBar()->setTabsClosable(m_tabs->count() > 1);
}

void MainWindow::closeTab(int index)
{
    if (m_tabs->count() <= 1) return;   // keep at least one

    auto *tab = qobject_cast<StreamTab *>(m_tabs->widget(index));
    if (tab) {
        tab->shutDown();
        int id = tab->streamId();
        m_tabs->removeTab(index);
        StreamStateManager::instance().removeStream(id);
        tab->deleteLater();
    }

    m_addTabBtn->setEnabled(m_tabs->count() < StreamStateManager::MaxTabs);
    m_tabs->tabBar()->setTabsClosable(m_tabs->count() > 1);
}

void MainWindow::onCurrentTabChanged(int index)
{
    int id = streamIdForTab(index);
    if (id >= 0)
        StreamStateManager::instance().setActiveStream(id);
}

void MainWindow::onTabTitleChanged(int streamId, const QString &title)
{
    for (int i = 0; i < m_tabs->count(); ++i) {
        auto *tab = qobject_cast<StreamTab *>(m_tabs->widget(i));
        if (tab && tab->streamId() == streamId) {
            m_tabs->setTabText(i, title);
            break;
        }
    }
}

int MainWindow::streamIdForTab(int index) const
{
    auto *tab = qobject_cast<StreamTab *>(m_tabs->widget(index));
    return tab ? tab->streamId() : -1;
}
