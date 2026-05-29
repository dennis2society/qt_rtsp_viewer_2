#pragma once

#include <QWidget>

class VideoPlayer;
class QComboBox;
class QLineEdit;
class QPushButton;
class QSlider;
class QTimer;
class QEvent;
class QResizeEvent;

/// One tab per stream.  Contains a URL bar at the top and a VideoPlayer that
/// fills the rest.  Transport controls live in a semi-transparent overlay at
/// the bottom of the video area (auto-hides during playback).
class StreamTab : public QWidget
{
    Q_OBJECT

public:
    explicit StreamTab(int streamId, QWidget *parent = nullptr);
    ~StreamTab() override;

    int streamId() const
    {
        return m_streamId;
    }
    VideoPlayer *videoPlayer() const
    {
        return m_player;
    }

    /// Stop playback + recording (called when tab is about to be closed).
    void shutDown();

    /// Toggle recording programmatically (e.g. from a keyboard shortcut).
    void toggleRecord();

signals:
    void tabTitleChanged(int streamId, const QString &title);
    void statusMessage(const QString &msg);
    void closeTabRequested();

public slots:
    void onPlayStopClicked();

private slots:
    void onRecordToggled(bool checked);
    void onRemoveUrlClicked();
    void onCameraNameEdited(const QString &name);
    void onUrlChanged(const QString &url);

private:
    void populateUrlCombo();
    void updateOverlayButtons();
    void repositionOverlay();
    void showOverlay();
    void scheduleHideOverlay();
    /// Uncheck and un-highlight the record button without emitting its toggled signal.
    void resetRecordButton();

    bool eventFilter(QObject *obj, QEvent *ev) override;
    void resizeEvent(QResizeEvent *ev) override;

    int m_streamId;
    bool m_isShutDown = false;
    bool m_isPlaying = false;

    // UI - top bar
    QComboBox *m_urlCombo = nullptr;
    QPushButton *m_removeBtn = nullptr;
    QLineEdit *m_cameraNameEdit = nullptr;

    // UI - overlay
    QWidget *m_overlay = nullptr;
    QPushButton *m_playStopBtn = nullptr;
    QPushButton *m_recordBtn = nullptr;
    QPushButton *m_snapshotBtn = nullptr;
    QPushButton *m_muteBtn = nullptr;
    QSlider *m_volumeSlider = nullptr;

    VideoPlayer *m_player = nullptr;

    // Timers
    QTimer *m_hideTimer = nullptr;
    QTimer *m_reconnectTimer = nullptr;
};
