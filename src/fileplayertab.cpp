#include "fileplayertab.h"
#include "streamstatemanager.h"
#include "videoplayer.h"

#include <QFileDialog>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QResizeEvent>
#include <QSlider>
#include <QStyle>
#include <QTimer>
#include <QUrl>
#include <QVBoxLayout>

// Height of the transport bar overlay in pixels
static constexpr int kOverlayHeight = 52;
// Hide the overlay after this many ms of no mouse movement during playback
static constexpr int kAutoHideMs = 3000;

// -----------------------------------------------------------------------------
FilePlayerTab::FilePlayerTab(int streamId, QWidget *parent)
    : QWidget(parent)
    , m_streamId(streamId)
{
    // VideoPlayer fills the whole tab
    m_player = new VideoPlayer(m_streamId, this);
    m_player->setGeometry(0, 0, width(), height());

    // ── Overlay transport bar ─────────────────────────────────────────
    m_overlay = new QWidget(this);
    m_overlay->setObjectName(QStringLiteral("filePlayerOverlay"));
    m_overlay->setStyleSheet(
        "#filePlayerOverlay {"
        "  background-color: rgba(20,20,20,210);"
        "}"
        "QPushButton {"
        "  background: transparent;"
        "  color: white;"
        "  font-size: 18px;"
        "  border: none;"
        "  padding: 4px 8px;"
        "}"
        "QPushButton:hover { background: rgba(255,255,255,30); border-radius: 4px; }"
        "QLabel { color: #cccccc; font-size: 12px; }"
        "QSlider::groove:horizontal {"
        "  height: 4px;"
        "  background: rgba(255,255,255,60);"
        "  border-radius: 2px;"
        "}"
        "QSlider::sub-page:horizontal {"
        "  background: #42a5f5;"
        "  border-radius: 2px;"
        "}"
        "QSlider::handle:horizontal {"
        "  width: 12px; height: 12px;"
        "  margin: -4px 0;"
        "  background: white;"
        "  border-radius: 6px;"
        "}");
    m_overlay->setAttribute(Qt::WA_TranslucentBackground);

    auto *overlayLay = new QHBoxLayout(m_overlay);
    overlayLay->setContentsMargins(8, 6, 8, 6);
    overlayLay->setSpacing(6);

    // Open file button
    m_openBtn = new QPushButton(QStringLiteral("📁"), m_overlay);
    m_openBtn->setToolTip(QStringLiteral("Open video file…"));
    m_openBtn->setFixedWidth(36);

    // File name
    m_fileLabel = new QLabel(QStringLiteral("No file loaded"), m_overlay);
    m_fileLabel->setStyleSheet(QStringLiteral("color: #aaaaaa; font-size: 11px;"));
    m_fileLabel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
    m_fileLabel->setMaximumWidth(220);

    // Play/pause
    m_playPauseBtn = new QPushButton(m_overlay);
    m_playPauseBtn->setIcon(style()->standardIcon(QStyle::SP_MediaPlay));
    m_playPauseBtn->setToolTip(QStringLiteral("Play / Pause"));
    m_playPauseBtn->setFixedWidth(36);

    // Stop
    m_stopBtn = new QPushButton(m_overlay);
    m_stopBtn->setIcon(style()->standardIcon(QStyle::SP_MediaStop));
    m_stopBtn->setToolTip(QStringLiteral("Stop"));
    m_stopBtn->setFixedWidth(36);

    // Seek slider
    m_seekSlider = new QSlider(Qt::Horizontal, m_overlay);
    m_seekSlider->setRange(0, 0);
    m_seekSlider->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);

    // Time label
    m_timeLbl = new QLabel(QStringLiteral("0:00 / 0:00"), m_overlay);
    m_timeLbl->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    m_timeLbl->setFixedWidth(100);

    overlayLay->addWidget(m_openBtn);
    overlayLay->addWidget(m_fileLabel);
    overlayLay->addWidget(m_playPauseBtn);
    overlayLay->addWidget(m_stopBtn);
    overlayLay->addWidget(m_seekSlider, 1);
    overlayLay->addWidget(m_timeLbl);

    // Auto-hide timer
    m_hideTimer = new QTimer(this);
    m_hideTimer->setSingleShot(true);
    connect(m_hideTimer, &QTimer::timeout, m_overlay, &QWidget::hide);

    // Start visible so the user can open a file immediately
    m_overlay->show();
    m_overlay->raise();

    repositionOverlay();

    // ── Connections ───────────────────────────────────────────────────
    connect(m_openBtn, &QPushButton::clicked, this, &FilePlayerTab::onOpenFile);
    connect(m_playPauseBtn, &QPushButton::clicked, this, &FilePlayerTab::onPlayPause);
    connect(m_stopBtn, &QPushButton::clicked, this, &FilePlayerTab::onStop);

    connect(m_seekSlider, &QSlider::sliderPressed, this, &FilePlayerTab::onSliderPressed);
    connect(m_seekSlider, &QSlider::sliderReleased, this, &FilePlayerTab::onSliderReleased);
    connect(m_seekSlider, &QSlider::sliderMoved, this, &FilePlayerTab::onSeekSliderMoved);

    connect(m_player, &VideoPlayer::positionChanged, this, &FilePlayerTab::onPositionChanged);
    connect(m_player, &VideoPlayer::durationChanged, this, &FilePlayerTab::onDurationChanged);
    connect(m_player, &VideoPlayer::mediaPlaybackStateChanged, this, &FilePlayerTab::onPlaybackStateChanged);

    // Show overlay whenever the mouse moves over the video
    connect(m_player, &VideoPlayer::mouseMoved, this, &FilePlayerTab::showOverlay);
}

FilePlayerTab::~FilePlayerTab()
{
    shutDown();
}

// -----------------------------------------------------------------------------
void FilePlayerTab::shutDown()
{
    if (m_isShutDown)
        return;
    m_isShutDown = true;

    disconnect(m_player, nullptr, this, nullptr);
    m_player->stop();
}

// -----------------------------------------------------------------------------
void FilePlayerTab::resizeEvent(QResizeEvent *event)
{
    QWidget::resizeEvent(event);
    m_player->setGeometry(0, 0, width(), height());
    repositionOverlay();
}

void FilePlayerTab::repositionOverlay()
{
    m_overlay->setGeometry(0, height() - kOverlayHeight, width(), kOverlayHeight);
    m_overlay->raise();
}

// -----------------------------------------------------------------------------
// Overlay show / hide
// -----------------------------------------------------------------------------
void FilePlayerTab::showOverlay()
{
    m_overlay->show();
    m_overlay->raise();
    scheduleHide();
}

void FilePlayerTab::scheduleHide()
{
    // Only auto-hide while playing; keep visible when stopped/paused
    if (m_player->playbackState() == QMediaPlayer::PlayingState)
        m_hideTimer->start(kAutoHideMs);
    else
        m_hideTimer->stop();
}

// -----------------------------------------------------------------------------
// Transport slots
// -----------------------------------------------------------------------------
void FilePlayerTab::onOpenFile()
{
    const QString path = QFileDialog::getOpenFileName(this,
                                                      QStringLiteral("Open Video File"),
                                                      QString(),
                                                      QStringLiteral("Video Files (*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm *.ts *.m4v *.mpeg *.mpg);;"
                                                                     "All Files (*)"));

    if (path.isEmpty())
        return;

    m_player->stop();
    m_player->play(QUrl::fromLocalFile(path).toString());

    // Show just the file name in the label
    const QString name = path.section(QLatin1Char('/'), -1);
    m_fileLabel->setText(name);
    m_fileLabel->setToolTip(path);
    emit statusMessage(QStringLiteral("Opened: %1").arg(path));
}

void FilePlayerTab::onPlayPause()
{
    switch (m_player->playbackState()) {
    case QMediaPlayer::PlayingState:
        m_player->pause();
        break;
    case QMediaPlayer::PausedState:
        m_player->play(QString()); // resume (no URL needed – player keeps source)
        break;
    default:
        break;
    }
}

void FilePlayerTab::onStop()
{
    m_player->stop();
    m_seekSlider->setValue(0);
}

// -----------------------------------------------------------------------------
// Seek slider
// -----------------------------------------------------------------------------
void FilePlayerTab::onSliderPressed()
{
    m_seekDragging = true;
    m_hideTimer->stop(); // keep overlay visible while scrubbing
}

void FilePlayerTab::onSliderReleased()
{
    m_seekDragging = false;
    if (m_seekSlider->maximum() > 0) {
        const qint64 pos = static_cast<qint64>(m_seekSlider->value()) * m_player->duration() / m_seekSlider->maximum();
        m_player->seekTo(pos);
    }
    scheduleHide();
}

void FilePlayerTab::onSeekSliderMoved(int value)
{
    if (m_player->duration() > 0 && m_seekSlider->maximum() > 0) {
        const qint64 pos = static_cast<qint64>(value) * m_player->duration() / m_seekSlider->maximum();
        m_timeLbl->setText(QStringLiteral("%1 / %2").arg(formatTime(pos)).arg(formatTime(m_player->duration())));
    }
}

// -----------------------------------------------------------------------------
// Player state → UI
// -----------------------------------------------------------------------------
void FilePlayerTab::onPositionChanged(qint64 ms)
{
    if (m_seekDragging)
        return;

    const qint64 dur = m_player->duration();
    m_timeLbl->setText(QStringLiteral("%1 / %2").arg(formatTime(ms)).arg(formatTime(dur)));

    if (dur > 0 && m_seekSlider->maximum() > 0) {
        m_seekSlider->blockSignals(true);
        m_seekSlider->setValue(static_cast<int>(ms * m_seekSlider->maximum() / dur));
        m_seekSlider->blockSignals(false);
    }
}

void FilePlayerTab::onDurationChanged(qint64 ms)
{
    m_seekSlider->setRange(0, ms > 0 ? 10000 : 0);
    m_timeLbl->setText(QStringLiteral("0:00 / %1").arg(formatTime(ms)));
}

void FilePlayerTab::onPlaybackStateChanged(QMediaPlayer::PlaybackState state)
{
    switch (state) {
    case QMediaPlayer::PlayingState:
        m_playPauseBtn->setIcon(style()->standardIcon(QStyle::SP_MediaPause));
        scheduleHide();
        break;
    case QMediaPlayer::PausedState:
        m_playPauseBtn->setIcon(style()->standardIcon(QStyle::SP_MediaPlay));
        m_overlay->show();
        m_overlay->raise();
        m_hideTimer->stop();
        break;
    case QMediaPlayer::StoppedState:
        m_playPauseBtn->setIcon(style()->standardIcon(QStyle::SP_MediaPlay));
        m_overlay->show();
        m_overlay->raise();
        m_hideTimer->stop();
        break;
    }
}

// -----------------------------------------------------------------------------
// Static helper
// -----------------------------------------------------------------------------
QString FilePlayerTab::formatTime(qint64 ms)
{
    if (ms < 0)
        ms = 0;
    const int totalSec = static_cast<int>(ms / 1000);
    const int h = totalSec / 3600;
    const int m = (totalSec % 3600) / 60;
    const int s = totalSec % 60;
    if (h > 0)
        return QString::asprintf("%d:%02d:%02d", h, m, s);
    return QString::asprintf("%d:%02d", m, s);
}
