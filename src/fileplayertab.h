#pragma once

#include <QMediaPlayer>
#include <QWidget>

class VideoPlayer;
class QLabel;
class QPushButton;
class QSlider;
class QTimer;
class QResizeEvent;

/// Tab for playing local video files (mp4, avi, mov, …).
/// Wraps a full-area VideoPlayer (with effects + zoom/pan) and provides
/// an auto-hiding overlay transport bar at the bottom.
class FilePlayerTab : public QWidget
{
    Q_OBJECT

public:
    explicit FilePlayerTab(int streamId, QWidget *parent = nullptr);
    ~FilePlayerTab() override;

    int streamId() const
    {
        return m_streamId;
    }
    VideoPlayer *videoPlayer() const
    {
        return m_player;
    }

    /// Stop playback (called before the tab is closed).
    void shutDown();

signals:
    void statusMessage(const QString &msg);

protected:
    void resizeEvent(QResizeEvent *event) override;

private slots:
    void onOpenFile();
    void onPlayPause();
    void onStop();
    void onSliderPressed();
    void onSliderReleased();
    void onSeekSliderMoved(int value);

    void onPositionChanged(qint64 ms);
    void onDurationChanged(qint64 ms);
    void onPlaybackStateChanged(QMediaPlayer::PlaybackState state);

    void showOverlay();
    void scheduleHide();

private:
    void repositionOverlay();
    static QString formatTime(qint64 ms);

    int m_streamId;
    bool m_isShutDown = false;
    bool m_seekDragging = false;

    VideoPlayer *m_player = nullptr;

    // ---------- overlay widgets ----------
    QWidget *m_overlay = nullptr;
    QLabel *m_fileLabel = nullptr;
    QPushButton *m_openBtn = nullptr;
    QPushButton *m_playPauseBtn = nullptr;
    QPushButton *m_stopBtn = nullptr;
    QSlider *m_seekSlider = nullptr;
    QLabel *m_timeLbl = nullptr;
    QPushButton *m_muteBtn = nullptr;
    QSlider *m_volumeSlider = nullptr;
    QTimer *m_hideTimer = nullptr;
};
