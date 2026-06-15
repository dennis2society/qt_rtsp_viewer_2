#include "streamtab.h"
#include "recorddialog.h"
#include "streamstatemanager.h"
#include "videoplayer.h"

#include <QComboBox>
#include <QDateTime>
#include <QDir>
#include <QEvent>
#include <QFileDialog>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QPushButton>
#include <QRegularExpression>
#include <QResizeEvent>
#include <QSlider>
#include <QStyle>
#include <QTimer>
#include <QVBoxLayout>

static constexpr int kOverlayHeight = 52;
static constexpr int kAutoHideMs = 3000;

// -----------------------------------------------------------------------------
StreamTab::StreamTab(int streamId, QWidget *parent)
    : QWidget(parent)
    , m_streamId(streamId)
{
    auto *mainLay = new QVBoxLayout(this);
    mainLay->setContentsMargins(4, 4, 4, 0);
    mainLay->setSpacing(0);

    // -- Top bar: URL + camera name + remove -------------------------
    auto *ctrlLay = new QHBoxLayout;
    ctrlLay->setContentsMargins(0, 0, 0, 4);

    m_urlCombo = new QComboBox;
    m_urlCombo->setEditable(true);
    m_urlCombo->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    populateUrlCombo();

    m_cameraNameEdit = new QLineEdit;
    m_cameraNameEdit->setMinimumWidth(80);
    m_cameraNameEdit->setPlaceholderText(QStringLiteral("Camera name"));

    m_removeBtn = new QPushButton(QStringLiteral("✕ Remove"));
    m_removeBtn->setToolTip(QStringLiteral("Remove selected URL from history (does not close this tab)"));

    ctrlLay->addWidget(m_urlCombo, 3);
    ctrlLay->addWidget(m_cameraNameEdit, 2);
    ctrlLay->addWidget(m_removeBtn);

    mainLay->addLayout(ctrlLay);

    // Set initial camera name from state
    StreamState st = StreamStateManager::instance().stateCopy(m_streamId);
    m_cameraNameEdit->setText(st.cameraName);

    // -- Video player ------------------------------------------------
    m_player = new VideoPlayer(m_streamId, this);
    mainLay->addWidget(m_player, 1);

    setLayout(mainLay);

    // -- Overlay (absolute child, not in layout) ---------------------
    m_overlay = new QWidget(this);
    m_overlay->setObjectName(QStringLiteral("streamOverlay"));
    m_overlay->setStyleSheet(QStringLiteral("#streamOverlay { background-color: rgba(20,20,20,210); }"));

    auto *overlayLay = new QHBoxLayout(m_overlay);
    overlayLay->setContentsMargins(8, 6, 8, 6);
    overlayLay->setSpacing(6);

    m_playStopBtn = new QPushButton;
    m_playStopBtn->setIcon(style()->standardIcon(QStyle::SP_MediaPlay));
    m_playStopBtn->setToolTip(QStringLiteral("Play / Stop"));
    m_playStopBtn->setFixedSize(44, 44);
    m_playStopBtn->setIconSize(QSize(26, 26));
    m_playStopBtn->setStyleSheet(QStringLiteral("color:white;border:none;background:transparent;"));

    m_recordBtn = new QPushButton(QStringLiteral("⏺"));
    m_recordBtn->setCheckable(true);
    m_recordBtn->setToolTip(QStringLiteral("Start / stop recording"));
    m_recordBtn->setFixedSize(36, 36);
    m_recordBtn->setStyleSheet(
        QStringLiteral("QPushButton{color:#ff5555;border:1px solid rgba(255,80,80,120);"
                       "border-radius:5px;background:rgba(180,30,30,80);font-size:16px;}"
                       "QPushButton:checked{color:#ff2222;border:1px solid rgba(255,60,60,200);"
                       "background:rgba(200,20,20,140);}"));

    m_snapshotBtn = new QPushButton(QStringLiteral("📷"));
    m_snapshotBtn->setToolTip(QStringLiteral("Save snapshot (Ctrl+S)"));
    m_snapshotBtn->setFixedSize(36, 36);
    m_snapshotBtn->setStyleSheet(QStringLiteral("color:white;border:none;background:transparent;font-size:16px;"));

    m_muteBtn = new QPushButton(st.isMuted ? QStringLiteral("🔇") : QStringLiteral("🔊"));
    m_muteBtn->setCheckable(true);
    m_muteBtn->setChecked(st.isMuted);
    m_muteBtn->setToolTip(QStringLiteral("Mute / Unmute audio"));
    m_muteBtn->setFixedSize(36, 36);
    m_muteBtn->setStyleSheet(QStringLiteral("color:white;border:none;background:transparent;font-size:16px;"));

    m_volumeSlider = new QSlider(Qt::Horizontal);
    m_volumeSlider->setRange(0, 100);
    m_volumeSlider->setValue(80);
    m_volumeSlider->setFixedWidth(90);
    m_volumeSlider->setToolTip(QStringLiteral("Volume"));
    m_volumeSlider->setEnabled(!st.isMuted);
    m_volumeSlider->setStyleSheet(
        QStringLiteral("QSlider::groove:horizontal{height:4px;background:rgba(255,255,255,60);border-radius:2px;}"
                       "QSlider::sub-page:horizontal{background:rgba(255,255,255,180);border-radius:2px;}"
                       "QSlider::handle:horizontal{width:12px;height:12px;margin:-4px 0;"
                       "background:white;border-radius:6px;}"));

    // Layout: [⏺ rec] ─── stretch ─── [▶/⏹ play] ─── stretch ─── [📷] [🔊] [vol]
    overlayLay->addWidget(m_recordBtn);
    overlayLay->addStretch(1);
    overlayLay->addWidget(m_playStopBtn);
    overlayLay->addStretch(1);
    overlayLay->addWidget(m_snapshotBtn);
    overlayLay->addWidget(m_muteBtn);
    overlayLay->addWidget(m_volumeSlider);

    m_overlay->setLayout(overlayLay);

    // Apply saved mute state to audio output immediately
    m_player->setMuted(st.isMuted);
    m_player->setVolume(m_volumeSlider->value() / 100.0f);

    // Auto-hide timer
    m_hideTimer = new QTimer(this);
    m_hideTimer->setSingleShot(true);
    connect(m_hideTimer, &QTimer::timeout, m_overlay, &QWidget::hide);

    updateOverlayButtons();
    // Position overlay after the layout has settled
    QTimer::singleShot(0, this, &StreamTab::repositionOverlay);
    m_overlay->show();
    m_overlay->raise();

    // Watch player resize to reposition overlay
    m_player->installEventFilter(this);

    // -- Connections -------------------------------------------------
    connect(m_playStopBtn, &QPushButton::clicked, this, &StreamTab::onPlayStopClicked);
    connect(m_recordBtn, &QPushButton::toggled, this, &StreamTab::onRecordToggled);
    connect(m_removeBtn, &QPushButton::clicked, this, &StreamTab::onRemoveUrlClicked);
    connect(m_snapshotBtn, &QPushButton::clicked, m_player, &VideoPlayer::saveSnapshot);

    connect(m_muteBtn, &QPushButton::toggled, this, [this](bool muted) {
        m_player->setMuted(muted);
        m_muteBtn->setText(muted ? QStringLiteral("🔇") : QStringLiteral("🔊"));
        m_volumeSlider->setEnabled(!muted);
        StreamStateManager::instance().modifyState(m_streamId, [muted](StreamState &s) {
            s.isMuted = muted;
        });
    });

    connect(m_volumeSlider, &QSlider::valueChanged, this, [this](int v) {
        m_player->setVolume(v / 100.0f);
    });

    connect(m_cameraNameEdit, &QLineEdit::textChanged, this, &StreamTab::onCameraNameEdited);
    connect(m_urlCombo->lineEdit(), &QLineEdit::textChanged, this, &StreamTab::onUrlChanged);

    // When user picks an item from the dropdown, fill URL and camera name.
    connect(m_urlCombo, QOverload<int>::of(&QComboBox::activated), this, [this](int index) {
        if (index < 0)
            return;
        const QString url = m_urlCombo->itemData(index).toString();
        if (url.isEmpty())
            return;
        m_urlCombo->lineEdit()->setText(url);
        for (const auto &e : StreamStateManager::instance().urlHistory()) {
            if (e.url == url && !e.cameraName.isEmpty()) {
                m_cameraNameEdit->setText(e.cameraName);
                break;
            }
        }
    });

    // Player signals
    connect(m_player, &VideoPlayer::playbackStarted, this, [this]() {
        m_isPlaying = true;
        updateOverlayButtons();
        scheduleHideOverlay();
    });

    connect(m_player, &VideoPlayer::playbackStopped, this, [this]() {
        m_isPlaying = false;
        m_reconnectTimer->stop();
        m_player->stopRecording();
        resetRecordButton();
        updateOverlayButtons();
        showOverlay();
    });

    connect(m_player, &VideoPlayer::errorOccurred, this, [this](const QString &msg) {
        emit statusMessage(QStringLiteral("Stream %1 error: %2").arg(m_streamId).arg(msg));
        StreamState st = StreamStateManager::instance().stateCopy(m_streamId);
        if (st.rtspUrl.startsWith(QStringLiteral("rtsp://"), Qt::CaseInsensitive) || st.rtspUrl.startsWith(QStringLiteral("rtsps://"), Qt::CaseInsensitive)) {
            emit statusMessage(QStringLiteral("Stream %1: reconnecting in 5 s...").arg(m_streamId));
            m_reconnectTimer->start();
        }
    });

    connect(m_player, &VideoPlayer::recordingFinished, this, [this](const QString &path) {
        resetRecordButton();
        emit statusMessage(QStringLiteral("Recording saved: %1").arg(path));
    });

    connect(m_player, &VideoPlayer::recordingError, this, [this](const QString &msg) {
        resetRecordButton();
        emit statusMessage(QStringLiteral("Recording error: %1").arg(msg));
    });

    connect(m_player, &VideoPlayer::autoRecordingStarted, this, [this](const QString &path) {
        m_recordBtn->blockSignals(true);
        m_recordBtn->setChecked(true);
        m_recordBtn->setStyleSheet(QStringLiteral("background-color:#c62828;color:white;border-radius:4px;font-size:16px;"));
        m_recordBtn->blockSignals(false);
        emit statusMessage(QStringLiteral("Auto-recording started: %1").arg(path));
    });

    connect(m_player, &VideoPlayer::autoRecordingStopped, this, [this](const QString &path) {
        resetRecordButton();
        emit statusMessage(QStringLiteral("Auto-recording saved: %1").arg(path));
    });

    connect(m_player, &VideoPlayer::snapshotSaved, this, [this](const QString &path) {
        emit statusMessage(QStringLiteral("Snapshot saved: %1").arg(path));
    });

    connect(m_player, &VideoPlayer::faceDetectionUnavailable, this, [this]() {
        emit statusMessage(QStringLiteral("Stream %1: face detection unavailable — "
                                          "opencv/haarcascade_frontalface_default.xml not found")
                               .arg(m_streamId));
    });

    connect(m_player, &VideoPlayer::mouseMoved, this, &StreamTab::showOverlay);

    // React to global URL history changes (refresh combo)
    connect(&StreamStateManager::instance(), &StreamStateManager::globalSettingsChanged, this, [this]() {
        populateUrlCombo();
    });

    // Auto-reconnect timer
    m_reconnectTimer = new QTimer(this);
    m_reconnectTimer->setSingleShot(true);
    m_reconnectTimer->setInterval(5000);
    connect(m_reconnectTimer, &QTimer::timeout, this, [this]() {
        StreamState st = StreamStateManager::instance().stateCopy(m_streamId);
        if (!st.rtspUrl.isEmpty()) {
            emit statusMessage(QStringLiteral("Auto-reconnecting: %1").arg(st.cameraName));
            m_player->play(st.rtspUrl);
        }
    });
}

StreamTab::~StreamTab()
{
    shutDown();
}

// -----------------------------------------------------------------------------
void StreamTab::shutDown()
{
    if (m_isShutDown)
        return;
    m_isShutDown = true;

    if (m_reconnectTimer)
        m_reconnectTimer->stop();

    // Disconnect all signals to prevent callbacks into partially-destroyed objects
    disconnect(m_player, nullptr, this, nullptr);

    m_player->stopRecording();
    m_player->stop();
}

// -----------------------------------------------------------------------------
// URL combo population
// -----------------------------------------------------------------------------
void StreamTab::populateUrlCombo()
{
    const QString current = m_urlCombo->currentText();
    m_urlCombo->blockSignals(true);
    m_urlCombo->clear();
    for (const auto &e : StreamStateManager::instance().urlHistory()) {
        // Show "[CamName]  url" in the list, store the raw URL as item data
        const QString display = e.cameraName.isEmpty() ? e.url : QStringLiteral("[%1]  %2").arg(e.cameraName, e.url);
        m_urlCombo->addItem(display, e.url);
    }
    // Restore text (keep whatever was typed / selected)
    if (!current.isEmpty())
        m_urlCombo->lineEdit()->setText(current);
    m_urlCombo->blockSignals(false);
}

// -----------------------------------------------------------------------------
// Overlay positioning + visibility
// -----------------------------------------------------------------------------
void StreamTab::repositionOverlay()
{
    if (!m_overlay || !m_player)
        return;
    const QRect pr = m_player->geometry();
    m_overlay->setGeometry(pr.x(), pr.y() + pr.height() - kOverlayHeight, pr.width(), kOverlayHeight);
    m_overlay->raise();
}

void StreamTab::showOverlay()
{
    m_overlay->show();
    m_overlay->raise();
    scheduleHideOverlay();
}

void StreamTab::scheduleHideOverlay()
{
    if (m_isPlaying)
        m_hideTimer->start(kAutoHideMs);
    else
        m_hideTimer->stop();
}

void StreamTab::resizeEvent(QResizeEvent *ev)
{
    QWidget::resizeEvent(ev);
    repositionOverlay();
}

bool StreamTab::eventFilter(QObject *obj, QEvent *ev)
{
    if (obj == m_player && ev->type() == QEvent::Resize)
        repositionOverlay();
    return QWidget::eventFilter(obj, ev);
}

// -----------------------------------------------------------------------------
// Transport controls
// -----------------------------------------------------------------------------
void StreamTab::onPlayStopClicked()
{
    if (m_isPlaying) {
        m_reconnectTimer->stop();
        m_player->stopRecording();
        resetRecordButton();
        m_player->stop();
        StreamState st = StreamStateManager::instance().stateCopy(m_streamId);
        emit statusMessage(QStringLiteral("Playback stopped: %1").arg(st.cameraName));
    } else {
        QString url = m_urlCombo->currentText().trimmed();
        if (url.isEmpty())
            return;

        QString cam = m_cameraNameEdit->text().trimmed();
        if (cam.isEmpty()) {
            StreamStateManager::instance().readState(m_streamId, [&](const StreamState &s) {
                cam = s.cameraName;
            });
        }
        StreamStateManager::instance().addUrlToHistory(url, cam);
        StreamStateManager::instance().setLastPlayedUrl(url);
        StreamStateManager::instance().modifyState(m_streamId, [&](StreamState &s) {
            s.rtspUrl = url;
            s.cameraName = cam;
        });

        m_player->play(url);
        emit tabTitleChanged(m_streamId, cam);
        emit statusMessage(QStringLiteral("Playing: %1").arg(cam));
    }
    showOverlay();
}

void StreamTab::onRecordToggled(bool checked)
{
    StreamState st = StreamStateManager::instance().stateCopy(m_streamId);

    if (checked) {
        // Resolve output folder: per-tab first, then global, then prompt
        QString folder = st.outputFolder.isEmpty() ? StreamStateManager::instance().outputFolder() : st.outputFolder;

        if (folder.isEmpty()) {
            // First recording with no custom folder set — ask user
            auto btn = QMessageBox::question(this,
                                             QStringLiteral("Output Folder"),
                                             QStringLiteral("No output folder is set for this tab.\n"
                                                            "Set a custom folder for this tab now?"),
                                             QMessageBox::Yes | QMessageBox::No);
            if (btn == QMessageBox::Yes) {
                folder =
                    QFileDialog::getExistingDirectory(this, QStringLiteral("Select Output Folder for This Tab"), QDir::homePath(), QFileDialog::ShowDirsOnly);
                if (!folder.isEmpty()) {
                    // Save as per-tab folder
                    StreamStateManager::instance().modifyState(m_streamId, [&folder](StreamState &s) {
                        s.outputFolder = folder;
                    });
                }
            }
            // If still empty, abort
            if (folder.isEmpty()) {
                m_recordBtn->blockSignals(true);
                m_recordBtn->setChecked(false);
                m_recordBtn->blockSignals(false);
                return;
            }
        }

        // Re-read state (may have been modified above)
        st = StreamStateManager::instance().stateCopy(m_streamId);

        // Auto-generate path
        QDir().mkpath(folder);
        QString ts = QDateTime::currentDateTime().toString(QStringLiteral("yyyy-MM-dd_HH-mm-ss"));
        QString cam = st.cameraName;
        cam.replace(QRegularExpression(QStringLiteral("[^a-zA-Z0-9_-]")), QStringLiteral("_"));
        const bool isRaw = (st.recordCodec == QLatin1String("raw_copy"));
        QString ext = isRaw ? QStringLiteral("mp4") : st.recordFormat;
        QString path =
            isRaw ? QStringLiteral("%1/%2_%3_recording_raw.%4").arg(folder, ts, cam, ext) : QStringLiteral("%1/%2_%3_recording.%4").arg(folder, ts, cam, ext);
        m_player->startRecording(path, st.recordCodec, st.recordFps);
        m_recordBtn->setStyleSheet(QStringLiteral("background-color:#c62828;color:white;border-radius:4px;font-size:16px;"));
        emit statusMessage(QStringLiteral("Recording started: %1").arg(st.cameraName));
    } else {
        m_player->stopRecording();
        m_recordBtn->setStyleSheet(QStringLiteral("color:white;border:none;background:transparent;font-size:16px;"));
        emit statusMessage(QStringLiteral("Recording stopped: %1").arg(st.cameraName));
    }
}

void StreamTab::onRemoveUrlClicked()
{
    const QString url = m_urlCombo->currentText().trimmed();
    if (url.isEmpty())
        return;
    // Remove only from history - do NOT stop playback or close the tab.
    StreamStateManager::instance().removeUrlFromHistory(url);
    emit statusMessage(QStringLiteral("Removed from history: %1").arg(url));
}

void StreamTab::onCameraNameEdited(const QString &name)
{
    StreamStateManager::instance().modifyState(m_streamId, [&](StreamState &s) {
        s.cameraName = name;
    });
    emit tabTitleChanged(m_streamId, name);
}

void StreamTab::onUrlChanged(const QString &url)
{
    StreamStateManager::instance().modifyState(m_streamId, [&](StreamState &s) {
        s.rtspUrl = url;
    });
}

// -----------------------------------------------------------------------------
void StreamTab::updateOverlayButtons()
{
    m_playStopBtn->setIcon(style()->standardIcon(m_isPlaying ? QStyle::SP_MediaStop : QStyle::SP_MediaPlay));
    m_recordBtn->setEnabled(m_isPlaying);
    m_snapshotBtn->setEnabled(m_isPlaying);
}

void StreamTab::resetRecordButton()
{
    m_recordBtn->blockSignals(true);
    m_recordBtn->setChecked(false);
    m_recordBtn->setStyleSheet(QStringLiteral("color:white;border:none;background:transparent;font-size:16px;"));
    m_recordBtn->blockSignals(false);
}

void StreamTab::toggleRecord()
{
    if (!m_recordBtn->isEnabled())
        return;
    m_recordBtn->setChecked(!m_recordBtn->isChecked());
}
