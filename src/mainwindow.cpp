#include "mainwindow.h"
#include "effectssidebar.h"
#include "fileplayertab.h"
#include "streamstatemanager.h"
#include "streamtab.h"
#include "videoplayer.h"

#include <QCloseEvent>
#include <QColor>
#include <QComboBox>
#include <QHBoxLayout>
#include <QIcon>
#include <QImage>
#include <QLabel>
#include <QLineEdit>
#include <QMediaPlayer>
#include <QPainter>
#include <QPixmap>
#include <QPushButton>
#include <QScrollArea>
#include <QScrollBar>
#include <QShortcut>
#include <QStatusBar>
#include <QTabBar>
#include <QTabWidget>
#include <QTimer>
#include <QVBoxLayout>

// -----------------------------------------------------------------------------
MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
{
    setWindowTitle(QStringLiteral("RTSP Stream Viewer"));
    setMinimumSize(800, 480);
    resize(1200, 720);

    // -- Central widget ----------------------------------------------
    auto *central = new QWidget;
    auto *hlay = new QHBoxLayout(central);
    hlay->setContentsMargins(4, 4, 4, 4);

    // Tab widget
    m_tabs = new QTabWidget;
    m_tabs->setTabsClosable(true);
    m_tabs->setMovable(true);

    // "📁" open-file button + "+" add-tab button in a shared corner widget
    auto *cornerWidget = new QWidget;
    auto *cornerLay = new QHBoxLayout(cornerWidget);
    cornerLay->setContentsMargins(0, 0, 2, 0);
    cornerLay->setSpacing(2);

    m_openFileBtn = new QPushButton(QStringLiteral("📁"));
    m_openFileBtn->setToolTip(QStringLiteral("Open video file in new tab"));
    m_openFileBtn->setFixedSize(28, 28);
    cornerLay->addWidget(m_openFileBtn);

    m_addTabBtn = new QPushButton(QStringLiteral("+"));
    m_addTabBtn->setToolTip(QStringLiteral("Add new stream tab"));
    m_addTabBtn->setFixedSize(28, 28);
    cornerLay->addWidget(m_addTabBtn);

    m_tabs->setCornerWidget(cornerWidget, Qt::TopRightCorner);

    hlay->addWidget(m_tabs, 1);

    // Sidebar in scroll area
    m_sidebarScroll = new QScrollArea;
    m_sidebarScroll->setWidgetResizable(true);
    m_sidebarScroll->setMinimumWidth(210);
    m_sidebarScroll->setMaximumWidth(280);
    m_sidebarScroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    m_sidebar = new EffectsSidebar;
    m_sidebarScroll->setWidget(m_sidebar);
    hlay->addWidget(m_sidebarScroll);

    setCentralWidget(central);

    // Status bar
    if (!statusBar())
        setStatusBar(new QStatusBar);

    // -- Connections -------------------------------------------------
    connect(m_addTabBtn, &QPushButton::clicked, this, &MainWindow::addNewTab);
    connect(m_openFileBtn, &QPushButton::clicked, this, &MainWindow::openFilePlayerTab);
    connect(m_tabs, &QTabWidget::tabCloseRequested, this, &MainWindow::closeTab);
    connect(m_tabs, &QTabWidget::currentChanged, this, &MainWindow::onCurrentTabChanged);

    // When active stream changes, rebind sidebar
    connect(&StreamStateManager::instance(), &StreamStateManager::activeStreamChanged, this, [this](int streamId) {
        m_sidebar->bindToStream(streamId);
    });

    // When stream state changes (playing/recording), update the tab icon
    connect(&StreamStateManager::instance(), &StreamStateManager::streamStateChanged, this, &MainWindow::updateTabIcon);

    // -- Keyboard shortcuts ------------------------------------------
    // Space = play / stop the current stream tab
    auto *shortcutPlay = new QShortcut(QKeySequence(Qt::Key_Space), this);
    connect(shortcutPlay, &QShortcut::activated, this, [this]() {
        if (auto *tab = qobject_cast<StreamTab *>(m_tabs->currentWidget()))
            tab->onPlayStopClicked();
    });

    // Ctrl+S = save snapshot
    auto *shortcutSnap = new QShortcut(QKeySequence::Save, this);
    connect(shortcutSnap, &QShortcut::activated, this, [this]() {
        if (auto *tab = qobject_cast<StreamTab *>(m_tabs->currentWidget()))
            tab->videoPlayer()->saveSnapshot();
    });

    // Ctrl+R = toggle record
    auto *shortcutRec = new QShortcut(QKeySequence(Qt::CTRL | Qt::Key_R), this);
    connect(shortcutRec, &QShortcut::activated, this, [this]() {
        if (auto *tab = qobject_cast<StreamTab *>(m_tabs->currentWidget()))
            tab->toggleRecord();
    });

    // F11 = toggle full-screen overlay
    auto *shortcutFs = new QShortcut(QKeySequence(Qt::Key_F11), this);
    connect(shortcutFs, &QShortcut::activated, this, [this]() {
        if (auto *tab = qobject_cast<StreamTab *>(m_tabs->currentWidget()))
            tab->videoPlayer()->toggleFullScreen();
    });
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
                    s.motionVectorsSensitivity = entry.motionVectorsSensitivity;
                    s.motionTracesEnabled = entry.motionTracesEnabled;
                    s.motionTraceDecay = entry.motionTraceDecay;
                    s.motionGraphEnabled = entry.motionGraphEnabled;
                    s.motionGraphSensitivity = entry.motionGraphSensitivity;
                    s.faceDetectionEnabled = entry.faceDetectionEnabled;
                    s.overlayEnabled = entry.overlayEnabled;
                    s.motionCsvEnabled = entry.motionCsvEnabled;
                    s.recordCleanVideo = entry.recordCleanVideo;
                    s.recordCodec = entry.recordCodec;
                    s.recordFormat = entry.recordFormat;
                    s.recordFps = entry.recordFps;
                    s.autoRecordEnabled = entry.autoRecordEnabled;
                    s.autoRecordThreshold = entry.autoRecordThreshold;
                    s.autoRecordTimeout = entry.autoRecordTimeout;
                    s.outputFolder = entry.outputFolder;
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
            entry.motionVectorsSensitivity = st.motionVectorsSensitivity;
            entry.motionTracesEnabled = st.motionTracesEnabled;
            entry.motionTraceDecay = st.motionTraceDecay;
            entry.motionGraphEnabled = st.motionGraphEnabled;
            entry.motionGraphSensitivity = st.motionGraphSensitivity;
            entry.faceDetectionEnabled = st.faceDetectionEnabled;
            entry.overlayEnabled = st.overlayEnabled;
            entry.motionCsvEnabled = st.motionCsvEnabled;
            entry.recordCleanVideo = st.recordCleanVideo;
            entry.recordCodec = st.recordCodec;
            entry.recordFormat = st.recordFormat;
            entry.recordFps = st.recordFps;
            entry.autoRecordEnabled = st.autoRecordEnabled;
            entry.autoRecordThreshold = st.autoRecordThreshold;
            entry.autoRecordTimeout = st.autoRecordTimeout;
            entry.outputFolder = st.outputFolder;
            tabs.append(entry);
        }
    }
    StreamStateManager::instance().setOpenTabs(tabs);

    // Save the active tab index
    StreamStateManager::instance().setLastActiveTabIndex(m_tabs->currentIndex());

    // Shut down all tabs (disconnects signals first, so no callbacks into dead objects)
    for (int i = 0; i < m_tabs->count(); ++i) {
        if (auto *tab = qobject_cast<StreamTab *>(m_tabs->widget(i)))
            tab->shutDown();
        else if (auto *ftab = qobject_cast<FilePlayerTab *>(m_tabs->widget(i)))
            ftab->shutDown();
    }
    StreamStateManager::instance().saveSettings();
}

// -----------------------------------------------------------------------------
// Tab management
void MainWindow::autoMuteOtherTabs(int playingStreamId)
{
    for (int i = 0; i < m_tabs->count(); ++i) {
        if (auto *tab = qobject_cast<StreamTab *>(m_tabs->widget(i)))
            tab->videoPlayer()->setAutoMuted(tab->streamId() != playingStreamId);
    }
}

void MainWindow::clearAllAutoMutes()
{
    for (int i = 0; i < m_tabs->count(); ++i) {
        if (auto *tab = qobject_cast<StreamTab *>(m_tabs->widget(i)))
            tab->videoPlayer()->setAutoMuted(false);
    }
}

// -----------------------------------------------------------------------------
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

    // Auto-mute: when this tab starts playing, mute all others.
    // When it stops, clear all auto-mutes (let user-mute prefs take over).
    connect(tab->videoPlayer(), &VideoPlayer::playbackStarted, this, [this, tab]() {
        autoMuteOtherTabs(tab->streamId());
    });
    connect(tab->videoPlayer(), &VideoPlayer::playbackStopped, this, [this]() {
        clearAllAutoMutes();
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

    QWidget *w = m_tabs->widget(index);
    int id = -1;

    if (auto *tab = qobject_cast<StreamTab *>(w)) {
        tab->shutDown();
        id = tab->streamId();
    } else if (auto *ftab = qobject_cast<FilePlayerTab *>(w)) {
        ftab->shutDown();
        id = ftab->streamId();
    }

    m_tabs->removeTab(index);
    if (id >= 0)
        StreamStateManager::instance().removeStream(id);
    w->deleteLater();

    m_addTabBtn->setEnabled(m_tabs->count() < StreamStateManager::MaxTabs);
    m_tabs->tabBar()->setTabsClosable(m_tabs->count() > 1);
}

void MainWindow::onCurrentTabChanged(int index)
{
    const int id = streamIdForTab(index);
    if (id < 0)
        return;

    // Save previous stream's scroll position
    const int prevId = StreamStateManager::instance().activeStreamId();
    if (prevId >= 0 && m_sidebarScroll)
        m_sidebarScrollPos[prevId] = m_sidebarScroll->verticalScrollBar()->value();

    StreamStateManager::instance().setActiveStream(id);

    // Restore scroll position for the new stream (deferred so layout settles)
    if (m_sidebarScroll) {
        const int savedPos = m_sidebarScrollPos.value(id, 0);
        QTimer::singleShot(0, this, [this, savedPos]() {
            m_sidebarScroll->verticalScrollBar()->setValue(savedPos);
        });
    }
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
    QWidget *w = m_tabs->widget(index);
    if (auto *tab = qobject_cast<StreamTab *>(w))
        return tab->streamId();
    if (auto *ftab = qobject_cast<FilePlayerTab *>(w))
        return ftab->streamId();
    return -1;
}

void MainWindow::openFilePlayerTab()
{
    int id = StreamStateManager::instance().createStream();
    auto *ftab = new FilePlayerTab(id, this);

    int idx = m_tabs->addTab(ftab, QStringLiteral("📁 File Player"));
    m_tabs->setCurrentIndex(idx);

    connect(ftab, &FilePlayerTab::statusMessage, this, [this](const QString &msg) {
        statusBar()->showMessage(msg, 5000);
    });

    m_tabs->tabBar()->setTabsClosable(m_tabs->count() > 1);
}

// -----------------------------------------------------------------------------
// Tab status icons
// -----------------------------------------------------------------------------
QIcon MainWindow::makeStatusIcon(const QColor &color)
{
    QPixmap pm(12, 12);
    pm.fill(Qt::transparent);
    QPainter p(&pm);
    p.setRenderHint(QPainter::Antialiasing);
    p.setBrush(color);
    p.setPen(Qt::NoPen);
    p.drawEllipse(1, 1, 10, 10);
    return QIcon(pm);
}

void MainWindow::updateTabIcon(int streamId)
{
    for (int i = 0; i < m_tabs->count(); ++i) {
        auto *tab = qobject_cast<StreamTab *>(m_tabs->widget(i));
        if (!tab || tab->streamId() != streamId)
            continue;
        StreamState st = StreamStateManager::instance().stateCopy(streamId);
        QIcon icon;
        if (st.isRecording || st.isAutoRecording)
            icon = makeStatusIcon(QColor(0xc6, 0x28, 0x28)); // red
        else if (st.playbackState == PlaybackState::Playing)
            icon = makeStatusIcon(QColor(0x2e, 0x7d, 0x32)); // green
        else
            icon = makeStatusIcon(QColor(0xaa, 0xaa, 0xaa)); // gray
        m_tabs->setTabIcon(i, icon);
        break;
    }
}
