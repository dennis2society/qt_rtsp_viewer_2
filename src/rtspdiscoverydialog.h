#pragma once

#include <QDialog>
#include <QList>

class QLabel;
class QLineEdit;
class QProgressBar;
class QPushButton;
class QSpinBox;
class QTableWidget;
class QTcpSocket;

/// Scans an IPv4 range for hosts with open RTSP ports and confirms them by
/// sending an RTSP OPTIONS request.
class RtspDiscoveryDialog : public QDialog
{
    Q_OBJECT

public:
    explicit RtspDiscoveryDialog(QWidget *parent = nullptr);
    ~RtspDiscoveryDialog() override;

signals:
    void openUrlRequested(const QString &url);

private slots:
    void startScan();
    void stopScan();
    void openSelected();
    void copySelectedUrl();
    void testSelectedHost();
    void openSelectedTestUrl();

private:
    struct Target {
        quint32 ip;
        quint16 port;
    };

    struct UrlCandidate {
        QString path; // may include "?query"
        QString user;
        QString pass;
    };

    QList<UrlCandidate> buildCandidates(const QString &host) const;
    void launchUrlTests();
    void runDescribe(int index, const QByteArray &authorization);
    void finishUrlTest(int index, int code, const QString &reason, bool authSent);
    void stopUrlTests();

    void prefillLocalSubnet();
    void launchProbes();
    void startProbe(const Target &t);
    void finishProbe(QTcpSocket *sock, bool open, const QByteArray &reply);
    void addResult(const QString &host, quint16 port, bool rtsp, const QString &server);
    void setScanning(bool scanning);
    QString urlForRow(int row) const;

    QLineEdit *m_startIpEdit = nullptr;
    QLineEdit *m_endIpEdit = nullptr;
    QLineEdit *m_portsEdit = nullptr;
    QSpinBox *m_timeoutSpin = nullptr;
    QSpinBox *m_parallelSpin = nullptr;
    QLineEdit *m_pathEdit = nullptr;
    QLineEdit *m_userEdit = nullptr;
    QLineEdit *m_passEdit = nullptr;
    QPushButton *m_scanBtn = nullptr;
    QPushButton *m_stopBtn = nullptr;
    QPushButton *m_openBtn = nullptr;
    QPushButton *m_copyBtn = nullptr;
    QProgressBar *m_progress = nullptr;
    QLabel *m_statusLabel = nullptr;
    QTableWidget *m_table = nullptr;

    QPushButton *m_testBtn = nullptr;
    QPushButton *m_openTestBtn = nullptr;
    QLabel *m_testStatusLabel = nullptr;
    QTableWidget *m_testTable = nullptr;
    QObject *m_testRoot = nullptr; // parent of URL-test sockets, kept apart from scan sockets

    QString m_testHost;
    quint16 m_testPort = 0;
    QList<UrlCandidate> m_candidates;
    int m_nextCandidate = 0;
    int m_testActive = 0;
    int m_testDone = 0;
    int m_testOk = 0;

    QList<Target> m_targets;
    int m_nextTarget = 0;
    int m_active = 0;
    int m_done = 0;
    bool m_scanning = false;
};
