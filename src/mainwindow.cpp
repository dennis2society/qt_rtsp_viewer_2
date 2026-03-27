#include "mainwindow.h"
#include "effectssidebar.h"
#include "streamstatemanager.h"
#include "streamtab.h"

#include <QCloseEvent>
#include <QComboBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QScrollArea>
#include <QStatusBar>
#include <QTabBar>
#include <QTabWidget>
#include <QTimer>
#include <QVBoxLayout>

// ─────────────────────────────────────────────────────────────────────────────
MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
{
    setWindowTitle(QStringLiteral("RTSP Stream Viewer"));
    setMinimumSize(800, 480);
    resize(1200, 720);

    // ── Central widget ──────────────────────────────────────────────
    auto *central = new QWidget;
    auto *hlay = new QHBoxLayout(central);
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
    scroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
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
    connect(m_tabs, &QTabWidget::currentChanged, this, &MainWindow::onCurrentTabChanged);

    // When active stream changes, rebind sidebar
    connect(&StreamStateManager::instance(), &StreamStateManager::activeStreamChanged, this, [this](int streamId) {
        m_sidebar->bindToStream(streamId);
    });

    // ── Restore saved tabs, or create first tab ──────────────────
    auto savedTabs = StreamStateManager::instance().openTabs();
    if (savedTabs.isEmpty()) {
        addNewTab();
        // Auto-play last URL in the first tab
        QString lastUrl = StreamStateManager::instance().lastPlayedUrl();
        if (!lastUrl.isEmpty()) {
            auto *tab = qobject_cast<StreamTab *>(m_tabs->widget(0));
            if (tab) {
                QTimer::singleShot(500, this, [tab, lastUrl]() {
                    auto *combo = tab->findChild<QComboBox *>();
                    if (combo && combo->lineEdit())
                        combo->lineEdit()->setText(lastUrl);
                });
            }
        }
    } else {
        for (const auto &entry : savedTabs) {
            addNewTab();
            int tabIdx = m_tabs->count() - 1;
            auto *tab = qobject_cast<StreamTab *>(m_tabs->widget(tabIdx));
            if (tab) {
                int sid = tab->streamId();

                // Restore effect settings into the stream state
                StreamStateManager::instance().modifyState(sid, [&](StreamState &s) {
                    s.cameraName = entry.cameraName;
                    s.rtspUrl = entry.url;
                    s.blurAmount = entry.blurAmount;
                    s.grayscaleEnabled = entry.grayscaleEnabled;
                    s.brightnessAmount = entry.brightnessAmount;
                    s.contrastAmount = entry.contrastAmount;
                    s.colorTemperature = entry.colorTemperature;
                    s.motionDetectionEnabled = entry.motionDetectionEnabled;
                    s.motionSensitivity = entry.motionSensitivity;
                    s.motionVectorsEnabled = entry.motionVectorsEnabled;
                    s.motionTracesEnabled = entry.motionTracesEnabled;
                    s.motionTraceDecay = entry.motionTraceDecay;
                    s.motionGraphEnabled = entry.motionGraphEnabled;
                    s.motionGraphSensitivity = entry.motionGraphSensitivity;
                    s.faceDetectionEnabled = entry.faceDetectionEnabled;
                    s.overlayEnabled = entry.overlayEnabled;
                    s.recordCodec = entry.recordCodec;
                    s.recordFormat = entry.recordFormat;
                    s.recordFps = entry.recordFps;
                    s.autoRecordEnabled = entry.autoRecordEnabled;
                    s.autoRecordThreshold = entry.autoRecordThreshold;
                    s.autoRecordTimeout = entry.autoRecordTimeout;
                });

                // Set camera name in the line edit
                auto *nameEdit = tab->findChild<QLineEdit *>();
                if (nameEdit && !entry.cameraName.isEmpty())
                    nameEdit->setText(entry.cameraName);

                // Update tab title
                if (!entry.cameraName.isEmpty())
                    m_tabs->setTabText(tabIdx, entry.cameraName);

                // Set URL (deferred so event loop settles)
                if (!entry.url.isEmpty()) {
                    QString url = entry.url;
                    QTimer::singleShot(500, this, [tab, url]() {
                        auto *combo = tab->findChild<QComboBox *>();
                        if (combo && combo->lineEdit())
                            combo->lineEdit()->setText(url);
                    });
                }
            }
        }

        // Restore the last active tab
        int lastIdx = StreamStateManager::instance().lastActiveTabIndex();
        if (lastIdx >= 0 && lastIdx < m_tabs->count())
            m_tabs->setCurrentIndex(lastIdx);

        // Ensure sidebar is bound with all settings loaded
        int activeId = StreamStateManager::instance().activeStreamId();
        if (activeId >= 0)
            m_sidebar->bindToStream(activeId);
    }
}

MainWindow::~MainWindow()
{
    // Save open tabs before shutting down
    QList<StreamStateManager::TabEntry> tabs;
    for (int i = 0; i < m_tabs->count(); ++i) {
        auto *tab = qobject_cast<StreamTab *>(m_tabs->widget(i));
        if (tab) {
            StreamState st = StreamStateManager::instance().stateCopy(tab->streamId());
            StreamStateManager::TabEntry entry;
            entry.url = st.rtspUrl;
            entry.cameraName = st.cameraName;
            entry.blurAmount = st.blurAmount;
            entry.grayscaleEnabled = st.grayscaleEnabled;
            entry.brightnessAmount = st.brightnessAmount;
            entry.contrastAmount = st.contrastAmount;
            entry.colorTemperature = st.colorTemperature;
            entry.motionDetectionEnabled = st.motionDetectionEnabled;
            entry.motionSensitivity = st.motionSensitivity;
            entry.motionVectorsEnabled = st.motionVectorsEnabled;
            entry.motionTracesEnabled = st.motionTracesEnabled;
            entry.motionTraceDecay = st.motionTraceDecay;
            entry.motionGraphEnabled = st.motionGraphEnabled;
            entry.motionGraphSensitivity = st.motionGraphSensitivity;
            entry.faceDetectionEnabled = st.faceDetectionEnabled;
            entry.overlayEnabled = st.overlayEnabled;
            entry.recordCodec = st.recordCodec;
            entry.recordFormat = st.recordFormat;
            entry.recordFps = st.recordFps;
            entry.autoRecordEnabled = st.autoRecordEnabled;
            entry.autoRecordThreshold = st.autoRecordThreshold;
            entry.autoRecordTimeout = st.autoRecordTimeout;
            tabs.append(entry);
        }
    }
    StreamStateManager::instance().setOpenTabs(tabs);

    // Save the active tab index
    StreamStateManager::instance().setLastActiveTabIndex(m_tabs->currentIndex());

    // Shut down all tabs (disconnects signals first, so no callbacks into dead objects)
    for (int i = 0; i < m_tabs->count(); ++i) {
        auto *tab = qobject_cast<StreamTab *>(m_tabs->widget(i));
        if (tab)
            tab->shutDown();
    }
    StreamStateManager::instance().saveSettings();
}

// ─────────────────────────────────────────────────────────────────────────────
// Tab management
// ─────────────────────────────────────────────────────────────────────────────
void MainWindow::addNewTab()
{
    if (m_tabs->count() >= StreamStateManager::MaxTabs)
        return;

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
    connect(tab, &StreamTab::closeTabRequested, this, [this, tab]() {
        // Find the tab index and close it
        int idx = m_tabs->indexOf(tab);
        if (idx >= 0)
            closeTab(idx);
    });

    // Enable / disable "+"
    m_addTabBtn->setEnabled(m_tabs->count() < StreamStateManager::MaxTabs);

    // Disable close when only 1 tab
    m_tabs->tabBar()->setTabsClosable(m_tabs->count() > 1);
}

void MainWindow::closeTab(int index)
{
    if (m_tabs->count() <= 1)
        return; // keep at least one

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
