#pragma once

#include <QWidget>

class VideoPlayer;
class QComboBox;
class QLineEdit;
class QPushButton;

/// One tab per stream.  Contains the URL bar, transport controls,
/// record button, and a VideoPlayer widget.
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

signals:
    void tabTitleChanged(int streamId, const QString &title);
    void statusMessage(const QString &msg);
    void closeTabRequested();

private slots:
    void onPlayClicked();
    void onStopClicked();
    void onPauseToggled(bool checked);
    void onRecordToggled(bool checked);
    void onRemoveUrlClicked();
    void onCameraNameEdited(const QString &name);
    void onUrlChanged(const QString &url);

private:
    void populateUrlCombo();
    void updateButtonStates();

    int m_streamId;
    bool m_isShutDown = false;

    // UI
    QComboBox *m_urlCombo = nullptr;
    QPushButton *m_removeBtn = nullptr;
    QLineEdit *m_cameraNameEdit = nullptr;
    QPushButton *m_playBtn = nullptr;
    QPushButton *m_pauseBtn = nullptr;
    QPushButton *m_stopBtn = nullptr;
    QPushButton *m_recordBtn = nullptr;
    VideoPlayer *m_player = nullptr;
};
