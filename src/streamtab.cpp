#include "streamtab.h"
#include "videoplayer.h"
#include "recorddialog.h"
#include "streamstatemanager.h"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QComboBox>
#include <QLineEdit>
#include <QPushButton>
#include <QFrame>
#include <QStyle>
#include <QDateTime>
#include <QDir>
#include <QRegularExpression>

// ─────────────────────────────────────────────────────────────────────────────
StreamTab::StreamTab(int streamId, QWidget *parent)
    : QWidget(parent), m_streamId(streamId)
{
    auto *mainLay = new QVBoxLayout(this);
    mainLay->setContentsMargins(4, 4, 4, 0);

    // ── Controls bar ────────────────────────────────────────────────
    auto *ctrlLay = new QHBoxLayout;

    m_urlCombo = new QComboBox;
    m_urlCombo->setEditable(true);
    m_urlCombo->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    populateUrlCombo();

    m_removeBtn = new QPushButton(QStringLiteral("Remove"));
    m_removeBtn->setToolTip(QStringLiteral("Remove selected URL from history"));

    m_cameraNameEdit = new QLineEdit;
    m_cameraNameEdit->setMaximumWidth(120);
    m_cameraNameEdit->setPlaceholderText(QStringLiteral("Camera name"));

    // Set initial camera name from state
    StreamState st = StreamStateManager::instance().stateCopy(m_streamId);
    m_cameraNameEdit->setText(st.cameraName);

    auto *sep1 = new QFrame; sep1->setFrameShape(QFrame::VLine); sep1->setFrameShadow(QFrame::Sunken);

    m_playBtn  = new QPushButton;
    m_playBtn->setIcon(style()->standardIcon(QStyle::SP_MediaPlay));
    m_playBtn->setToolTip(QStringLiteral("Play"));

    m_pauseBtn = new QPushButton;
    m_pauseBtn->setIcon(style()->standardIcon(QStyle::SP_MediaPause));
    m_pauseBtn->setCheckable(true);
    m_pauseBtn->setToolTip(QStringLiteral("Pause"));

    m_stopBtn  = new QPushButton;
    m_stopBtn->setIcon(style()->standardIcon(QStyle::SP_MediaStop));
    m_stopBtn->setToolTip(QStringLiteral("Stop"));

    auto *sep2 = new QFrame; sep2->setFrameShape(QFrame::VLine); sep2->setFrameShadow(QFrame::Sunken);

    m_recordBtn = new QPushButton(QStringLiteral("⏺ Record"));
    m_recordBtn->setCheckable(true);
    m_recordBtn->setToolTip(QStringLiteral("Start / stop recording"));

    ctrlLay->addWidget(m_urlCombo, 1);
    ctrlLay->addWidget(m_removeBtn);
    ctrlLay->addWidget(m_cameraNameEdit);
    ctrlLay->addWidget(sep1);
    ctrlLay->addWidget(m_playBtn);
    ctrlLay->addWidget(m_pauseBtn);
    ctrlLay->addWidget(m_stopBtn);
    ctrlLay->addWidget(sep2);
    ctrlLay->addWidget(m_recordBtn);

    mainLay->addLayout(ctrlLay);

    // ── Video player ────────────────────────────────────────────────
    m_player = new VideoPlayer(m_streamId, this);
    mainLay->addWidget(m_player, 1);

    setLayout(mainLay);
    updateButtonStates();

    // ── Connections ─────────────────────────────────────────────────
    connect(m_playBtn,  &QPushButton::clicked,  this, &StreamTab::onPlayClicked);
    connect(m_stopBtn,  &QPushButton::clicked,  this, &StreamTab::onStopClicked);
    connect(m_pauseBtn, &QPushButton::toggled,  this, &StreamTab::onPauseToggled);
    connect(m_recordBtn,&QPushButton::toggled,  this, &StreamTab::onRecordToggled);
    connect(m_removeBtn,&QPushButton::clicked,  this, &StreamTab::onRemoveUrlClicked);

    connect(m_cameraNameEdit, &QLineEdit::textChanged, this, &StreamTab::onCameraNameEdited);

    connect(m_urlCombo->lineEdit(), &QLineEdit::textChanged, this, &StreamTab::onUrlChanged);

    // Player signals
    connect(m_player, &VideoPlayer::errorOccurred, this, [this](const QString &msg) {
        emit statusMessage(QStringLiteral("Stream %1 error: %2").arg(m_streamId).arg(msg));
    });

    connect(m_player, &VideoPlayer::playbackStopped, this, [this]() {
        // If recording, stop it
        m_player->stopRecording();
        m_recordBtn->blockSignals(true);
        m_recordBtn->setChecked(false);
        m_recordBtn->blockSignals(false);
        updateButtonStates();
    });

    connect(m_player, &VideoPlayer::recordingFinished, this, [this](const QString &path) {
        m_recordBtn->blockSignals(true);
        m_recordBtn->setChecked(false);
        m_recordBtn->blockSignals(false);
        updateButtonStates();
        emit statusMessage(QStringLiteral("Recording saved: %1").arg(path));
    });

    connect(m_player, &VideoPlayer::recordingError, this, [this](const QString &msg) {
        m_recordBtn->blockSignals(true);
        m_recordBtn->setChecked(false);
        m_recordBtn->blockSignals(false);
        updateButtonStates();
        emit statusMessage(QStringLiteral("Recording error: %1").arg(msg));
    });

    connect(m_player, &VideoPlayer::autoRecordingStarted, this, [this](const QString &path) {
        m_recordBtn->blockSignals(true);
        m_recordBtn->setChecked(true);
        m_recordBtn->setStyleSheet(QStringLiteral("background-color:#c62828;color:white;"));
        m_recordBtn->blockSignals(false);
        emit statusMessage(QStringLiteral("Auto-recording started: %1").arg(path));
    });

    connect(m_player, &VideoPlayer::autoRecordingStopped, this, [this](const QString &path) {
        m_recordBtn->blockSignals(true);
        m_recordBtn->setChecked(false);
        m_recordBtn->setStyleSheet(QString());
        m_recordBtn->blockSignals(false);
        emit statusMessage(QStringLiteral("Auto-recording saved: %1").arg(path));
    });

    // React to global URL history changes (refresh combo)
    connect(&StreamStateManager::instance(), &StreamStateManager::globalSettingsChanged,
            this, [this]() { populateUrlCombo(); });
}

StreamTab::~StreamTab()
{
    shutDown();
}

// ─────────────────────────────────────────────────────────────────────────────
void StreamTab::shutDown()
{
    m_player->stopRecording();
    m_player->stop();
}

// ─────────────────────────────────────────────────────────────────────────────
// URL combo population
// ─────────────────────────────────────────────────────────────────────────────
void StreamTab::populateUrlCombo()
{
    QString current = m_urlCombo->currentText();
    m_urlCombo->blockSignals(true);
    m_urlCombo->clear();
    for (const auto &e : StreamStateManager::instance().urlHistory())
        m_urlCombo->addItem(e.url);
    // Restore text
    if (!current.isEmpty())
        m_urlCombo->lineEdit()->setText(current);
    m_urlCombo->blockSignals(false);
}

// ─────────────────────────────────────────────────────────────────────────────
// Transport controls
// ─────────────────────────────────────────────────────────────────────────────
void StreamTab::onPlayClicked()
{
    QString url = m_urlCombo->currentText().trimmed();
    if (url.isEmpty()) return;

    // Save to history
    QString cam = m_cameraNameEdit->text().trimmed();
    if (cam.isEmpty()) {
        StreamStateManager::instance().readState(m_streamId, [&](const StreamState &s) {
            cam = s.cameraName;
        });
    }
    StreamStateManager::instance().addUrlToHistory(url, cam);
    StreamStateManager::instance().setLastPlayedUrl(url);

    // Update stream state
    StreamStateManager::instance().modifyState(m_streamId, [&](StreamState &s) {
        s.rtspUrl    = url;
        s.cameraName = cam;
    });

    m_player->play(url);
    m_pauseBtn->blockSignals(true);
    m_pauseBtn->setChecked(false);
    m_pauseBtn->blockSignals(false);
    updateButtonStates();

    emit tabTitleChanged(m_streamId, cam);
}

void StreamTab::onStopClicked()
{
    m_player->stop();
    m_pauseBtn->blockSignals(true);
    m_pauseBtn->setChecked(false);
    m_pauseBtn->blockSignals(false);
    updateButtonStates();
}

void StreamTab::onPauseToggled(bool checked)
{
    m_player->setPaused(checked);
}

void StreamTab::onRecordToggled(bool checked)
{
    if (checked) {
        StreamState st = StreamStateManager::instance().stateCopy(m_streamId);
        QString folder = StreamStateManager::instance().outputFolder();

        if (!folder.isEmpty()) {
            // Auto-generate path
            QDir().mkpath(folder);
            QString ts  = QDateTime::currentDateTime().toString(QStringLiteral("yyyy-MM-dd_HH-mm-ss"));
            QString cam = st.cameraName;
            cam.replace(QRegularExpression(QStringLiteral("[^a-zA-Z0-9_-]")), QStringLiteral("_"));
            QString ext = st.recordFormat;
            QString path = QStringLiteral("%1/%2_%3_recording.%4").arg(folder, ts, cam, ext);
            m_player->startRecording(path, st.recordCodec, st.recordFps);
        } else {
            // Show dialog
            RecordDialog dlg(folder, st.cameraName, this);
            if (dlg.exec() == QDialog::Accepted) {
                m_player->startRecording(dlg.filePath(), dlg.codec(), dlg.fps());
            } else {
                m_recordBtn->blockSignals(true);
                m_recordBtn->setChecked(false);
                m_recordBtn->blockSignals(false);
            }
        }
    } else {
        m_player->stopRecording();
        m_recordBtn->setStyleSheet(QString());
    }
}

void StreamTab::onRemoveUrlClicked()
{
    QString url = m_urlCombo->currentText().trimmed();
    if (url.isEmpty()) return;
    StreamStateManager::instance().removeUrlFromHistory(url);
    m_urlCombo->removeItem(m_urlCombo->currentIndex());
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

// ─────────────────────────────────────────────────────────────────────────────
void StreamTab::updateButtonStates()
{
    StreamState st = StreamStateManager::instance().stateCopy(m_streamId);
    bool playing = (st.playbackState == PlaybackState::Playing ||
                    st.playbackState == PlaybackState::Paused);

    m_playBtn->setEnabled(!playing);
    m_stopBtn->setEnabled(playing);
    m_pauseBtn->setEnabled(playing);
    m_recordBtn->setEnabled(playing);
}
