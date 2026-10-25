#include "rtspdiscoverydialog.h"
#include "streamstatemanager.h"

#include <QAbstractSocket>
#include <QApplication>
#include <QBrush>
#include <QClipboard>
#include <QColor>
#include <QCryptographicHash>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QHostAddress>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QNetworkInterface>
#include <QProgressBar>
#include <QPushButton>
#include <QRandomGenerator>
#include <QRegularExpression>
#include <QSet>
#include <QSpinBox>
#include <QTableWidget>
#include <QTcpSocket>
#include <QTimer>
#include <QUrl>
#include <QVBoxLayout>

#include <memory>
#include <utility>

namespace
{
constexpr quint32 kMaxHosts = 65536; // a /16
constexpr int kMaxPorts = 16;
constexpr int kMaxReplyBytes = 8192;
constexpr int kParallelUrlTests = 4;

// Well-known stream paths of common camera vendors / firmwares.
const char *const kCommonPaths[] = {
    "/",
    "/live",
    "/live.sdp",
    "/live/main",
    "/live/sub",
    "/live/ch00_0",
    "/live/ch00_1",
    "/live/0/MAIN",
    "/stream",
    "/stream1",
    "/stream2",
    "/h264",
    "/h264_stream",
    "/ch0_0.h264",
    "/11",
    "/12",
    "/1",
    "/0",
    "/av0_0",
    "/onvif1",
    "/onvif2",
    "/media/video1",
    "/video1",
    "/mpeg4",
    "/Streaming/Channels/101", // Hikvision
    "/Streaming/Channels/102", // Hikvision
    "/cam/realmonitor?channel=1&subtype=0", // Dahua / Amcrest
    "/cam/realmonitor?channel=1&subtype=1", // Dahua / Amcrest
    "/h264Preview_01_main", // Reolink
    "/h264Preview_01_sub", // Reolink
    "/videoMain", // Foscam
    "/videoSub", // Foscam
    "/axis-media/media.amp", // Axis
    "/unicast/c1/s0/live", // Uniview
    "/unicast/c1/s1/live", // Uniview
    "/MediaInput/h264",
};

QUrl buildRtspUrl(const QString &host, quint16 port, const QString &pathAndQuery, const QString &user = QString(), const QString &pass = QString())
{
    QUrl url;
    url.setScheme(QStringLiteral("rtsp"));
    url.setHost(host);
    url.setPort(port);
    QString path = pathAndQuery.trimmed();
    const int q = path.indexOf(QLatin1Char('?'));
    if (q >= 0) {
        url.setQuery(path.mid(q + 1));
        path.truncate(q);
    }
    if (!path.startsWith(QLatin1Char('/')))
        path.prepend(QLatin1Char('/'));
    url.setPath(path);
    if (!user.isEmpty()) {
        url.setUserName(user);
        if (!pass.isEmpty())
            url.setPassword(pass);
    }
    return url;
}

QByteArray md5Hex(const QByteArray &data)
{
    return QCryptographicHash::hash(data, QCryptographicHash::Md5).toHex();
}

QByteArray sanitizeHeaderValue(const QString &s)
{
    QByteArray b = s.toUtf8();
    b.replace('\r', "").replace('\n', "").replace('"', "");
    return b;
}

/// Build an Authorization header value from a 401 reply (Digest preferred, else Basic).
QByteArray buildAuthorization(const QByteArray &headers, const QByteArray &method, const QByteArray &uri, const QString &user, const QString &pass)
{
    QByteArray digest;
    bool basic = false;
    for (const QByteArray &rawLine : headers.split('\n')) {
        const QByteArray line = rawLine.trimmed();
        if (!line.toLower().startsWith("www-authenticate:"))
            continue;
        const QByteArray value = line.mid(17).trimmed();
        const QByteArray lower = value.toLower();
        if (lower.startsWith("digest") && digest.isEmpty())
            digest = value.mid(6).trimmed();
        else if (lower.startsWith("basic"))
            basic = true;
    }

    const QByteArray u = sanitizeHeaderValue(user);
    const QByteArray p = pass.toUtf8();

    if (!digest.isEmpty()) {
        QMap<QByteArray, QByteArray> params;
        static const QRegularExpression re(QStringLiteral(R"re((\w+)\s*=\s*(?:"([^"]*)"|([^,\s]+)))re"));
        auto it = re.globalMatch(QString::fromLatin1(digest));
        while (it.hasNext()) {
            const auto m = it.next();
            const QString val = m.captured(2).isNull() ? m.captured(3) : m.captured(2);
            params.insert(m.captured(1).toLower().toLatin1(), val.toLatin1());
        }
        const QByteArray realm = params.value("realm");
        const QByteArray nonce = params.value("nonce");
        const QByteArray algorithm = params.value("algorithm", "MD5");
        if (nonce.isEmpty() || algorithm.toUpper() != "MD5")
            return QByteArray();

        const QByteArray ha1 = md5Hex(u + ':' + realm + ':' + p);
        const QByteArray ha2 = md5Hex(method + ':' + uri);
        QByteArray header = "Digest username=\"" + u + "\", realm=\"" + realm + "\", nonce=\"" + nonce + "\", uri=\"" + uri + "\"";

        const bool qopAuth = params.value("qop").split(',').contains("auth");
        if (qopAuth) {
            const QByteArray nc = "00000001";
            const QByteArray cnonce = QByteArray::number(QRandomGenerator::global()->generate64(), 16);
            header += ", qop=auth, nc=" + nc + ", cnonce=\"" + cnonce + "\", response=\"" + md5Hex(ha1 + ':' + nonce + ':' + nc + ':' + cnonce + ":auth:" + ha2)
                + "\"";
        } else {
            header += ", response=\"" + md5Hex(ha1 + ':' + nonce + ':' + ha2) + "\"";
        }
        if (params.contains("opaque"))
            header += ", opaque=\"" + params.value("opaque") + "\"";
        return header;
    }

    if (basic)
        return "Basic " + (u + ':' + p).toBase64();
    return QByteArray();
}

enum Column {
    ColHost,
    ColPort,
    ColStatus,
    ColServer,
    ColCount
};

bool parseIPv4(const QString &text, quint32 &out)
{
    QHostAddress addr;
    if (!addr.setAddress(text.trimmed()) || addr.protocol() != QAbstractSocket::IPv4Protocol)
        return false;
    bool ok = false;
    out = addr.toIPv4Address(&ok);
    return ok;
}
} // namespace

// -----------------------------------------------------------------------------
RtspDiscoveryDialog::RtspDiscoveryDialog(QWidget *parent)
    : QDialog(parent)
{
    setWindowTitle(QStringLiteral("Discover RTSP Cameras"));
    resize(680, 720);

    auto *mainLay = new QVBoxLayout(this);

    // -- Scan settings -----------------------------------------------
    auto *scanGroup = new QGroupBox(QStringLiteral("Scan Range"));
    auto *scanLay = new QGridLayout(scanGroup);

    scanLay->addWidget(new QLabel(QStringLiteral("Start IP:")), 0, 0);
    m_startIpEdit = new QLineEdit;
    m_startIpEdit->setPlaceholderText(QStringLiteral("e.g. 192.168.1.1"));
    scanLay->addWidget(m_startIpEdit, 0, 1);

    scanLay->addWidget(new QLabel(QStringLiteral("End IP:")), 0, 2);
    m_endIpEdit = new QLineEdit;
    m_endIpEdit->setPlaceholderText(QStringLiteral("e.g. 192.168.1.254"));
    scanLay->addWidget(m_endIpEdit, 0, 3);

    scanLay->addWidget(new QLabel(QStringLiteral("Ports:")), 1, 0);
    m_portsEdit = new QLineEdit(QStringLiteral("554, 8554"));
    m_portsEdit->setToolTip(QStringLiteral("Comma-separated list of TCP ports to probe"));
    scanLay->addWidget(m_portsEdit, 1, 1);

    auto *tuneLay = new QHBoxLayout;
    tuneLay->addWidget(new QLabel(QStringLiteral("Timeout:")));
    m_timeoutSpin = new QSpinBox;
    m_timeoutSpin->setRange(100, 10000);
    m_timeoutSpin->setSingleStep(100);
    m_timeoutSpin->setValue(800);
    m_timeoutSpin->setSuffix(QStringLiteral(" ms"));
    tuneLay->addWidget(m_timeoutSpin);
    tuneLay->addWidget(new QLabel(QStringLiteral("Parallel:")));
    m_parallelSpin = new QSpinBox;
    m_parallelSpin->setRange(1, 256);
    m_parallelSpin->setValue(64);
    tuneLay->addWidget(m_parallelSpin);
    scanLay->addLayout(tuneLay, 1, 2, 1, 2);

    mainLay->addWidget(scanGroup);

    // -- URL template ------------------------------------------------
    auto *urlGroup = new QGroupBox(QStringLiteral("Stream URL (used when opening a result)"));
    auto *urlLay = new QGridLayout(urlGroup);

    urlLay->addWidget(new QLabel(QStringLiteral("Path:")), 0, 0);
    m_pathEdit = new QLineEdit;
    m_pathEdit->setPlaceholderText(QStringLiteral("e.g. /stream1 or /h264Preview_01_main"));
    urlLay->addWidget(m_pathEdit, 0, 1, 1, 3);

    urlLay->addWidget(new QLabel(QStringLiteral("Username:")), 1, 0);
    m_userEdit = new QLineEdit;
    urlLay->addWidget(m_userEdit, 1, 1);

    urlLay->addWidget(new QLabel(QStringLiteral("Password:")), 1, 2);
    m_passEdit = new QLineEdit;
    m_passEdit->setEchoMode(QLineEdit::Password);
    urlLay->addWidget(m_passEdit, 1, 3);

    mainLay->addWidget(urlGroup);

    // -- Controls ----------------------------------------------------
    auto *ctrlLay = new QHBoxLayout;
    m_scanBtn = new QPushButton(QStringLiteral("Scan"));
    m_stopBtn = new QPushButton(QStringLiteral("Stop"));
    m_stopBtn->setEnabled(false);
    m_progress = new QProgressBar;
    m_progress->setRange(0, 1);
    m_progress->setValue(0);
    ctrlLay->addWidget(m_scanBtn);
    ctrlLay->addWidget(m_stopBtn);
    ctrlLay->addWidget(m_progress, 1);
    mainLay->addLayout(ctrlLay);

    m_statusLabel = new QLabel(QStringLiteral("Idle"));
    m_statusLabel->setStyleSheet(QStringLiteral("color:gray;"));
    mainLay->addWidget(m_statusLabel);

    // -- Results -----------------------------------------------------
    m_table = new QTableWidget(0, ColCount);
    m_table->setHorizontalHeaderLabels({QStringLiteral("Host"), QStringLiteral("Port"), QStringLiteral("Status"), QStringLiteral("Server")});
    m_table->setSelectionBehavior(QAbstractItemView::SelectRows);
    m_table->setSelectionMode(QAbstractItemView::SingleSelection);
    m_table->setEditTriggers(QAbstractItemView::NoEditTriggers);
    m_table->verticalHeader()->setVisible(false);
    m_table->horizontalHeader()->setStretchLastSection(true);
    mainLay->addWidget(m_table, 1);

    auto *resLay = new QHBoxLayout;
    m_openBtn = new QPushButton(QStringLiteral("Open in New Tab"));
    m_copyBtn = new QPushButton(QStringLiteral("Copy URL"));
    m_openBtn->setEnabled(false);
    m_copyBtn->setEnabled(false);
    auto *closeBtn = new QPushButton(QStringLiteral("Close"));
    resLay->addWidget(m_openBtn);
    resLay->addWidget(m_copyBtn);
    resLay->addStretch();
    resLay->addWidget(closeBtn);
    mainLay->addLayout(resLay);

    // -- URL test ----------------------------------------------------
    auto *testGroup = new QGroupBox(QStringLiteral("Stream URL Test (selected host)"));
    auto *testLay = new QVBoxLayout(testGroup);

    auto *testCtrlLay = new QHBoxLayout;
    m_testBtn = new QPushButton(QStringLiteral("Test URLs"));
    m_testBtn->setToolTip(
        QStringLiteral("Send RTSP DESCRIBE for the Path field, previously used URLs and "
                       "common vendor paths to the selected host"));
    m_testBtn->setEnabled(false);
    m_openTestBtn = new QPushButton(QStringLiteral("Open Selected URL"));
    m_openTestBtn->setEnabled(false);
    m_testStatusLabel = new QLabel;
    m_testStatusLabel->setStyleSheet(QStringLiteral("color:gray;"));
    testCtrlLay->addWidget(m_testBtn);
    testCtrlLay->addWidget(m_openTestBtn);
    testCtrlLay->addWidget(m_testStatusLabel, 1);
    testLay->addLayout(testCtrlLay);

    m_testTable = new QTableWidget(0, 2);
    m_testTable->setHorizontalHeaderLabels({QStringLiteral("Path"), QStringLiteral("Result")});
    m_testTable->setSelectionBehavior(QAbstractItemView::SelectRows);
    m_testTable->setSelectionMode(QAbstractItemView::SingleSelection);
    m_testTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
    m_testTable->verticalHeader()->setVisible(false);
    m_testTable->horizontalHeader()->setStretchLastSection(true);
    testLay->addWidget(m_testTable, 1);

    mainLay->addWidget(testGroup, 1);

    m_testRoot = new QObject(this);

    connect(m_scanBtn, &QPushButton::clicked, this, &RtspDiscoveryDialog::startScan);
    connect(m_stopBtn, &QPushButton::clicked, this, &RtspDiscoveryDialog::stopScan);
    connect(m_openBtn, &QPushButton::clicked, this, &RtspDiscoveryDialog::openSelected);
    connect(m_copyBtn, &QPushButton::clicked, this, &RtspDiscoveryDialog::copySelectedUrl);
    connect(closeBtn, &QPushButton::clicked, this, &QDialog::close);
    connect(m_table, &QTableWidget::cellDoubleClicked, this, &RtspDiscoveryDialog::openSelected);
    connect(m_table, &QTableWidget::itemSelectionChanged, this, [this]() {
        const bool has = m_table->currentRow() >= 0;
        m_openBtn->setEnabled(has);
        m_copyBtn->setEnabled(has);
        m_testBtn->setEnabled(has);
    });
    connect(m_testBtn, &QPushButton::clicked, this, &RtspDiscoveryDialog::testSelectedHost);
    connect(m_openTestBtn, &QPushButton::clicked, this, &RtspDiscoveryDialog::openSelectedTestUrl);
    connect(m_testTable, &QTableWidget::cellDoubleClicked, this, &RtspDiscoveryDialog::openSelectedTestUrl);
    connect(m_testTable, &QTableWidget::itemSelectionChanged, this, [this]() {
        m_openTestBtn->setEnabled(m_testTable->currentRow() >= 0);
    });

    prefillLocalSubnet();
}

RtspDiscoveryDialog::~RtspDiscoveryDialog()
{
    // Probe lambdas reference this dialog; sever them before children are destroyed.
    m_scanning = false;
    const auto sockets = findChildren<QTcpSocket *>();
    for (QTcpSocket *sock : sockets) {
        sock->disconnect();
        sock->abort();
    }
}

// -----------------------------------------------------------------------------
void RtspDiscoveryDialog::prefillLocalSubnet()
{
    const auto ifaces = QNetworkInterface::allInterfaces();
    for (const QNetworkInterface &iface : ifaces) {
        const auto flags = iface.flags();
        if (!(flags & QNetworkInterface::IsUp) || !(flags & QNetworkInterface::IsRunning) || (flags & QNetworkInterface::IsLoopBack))
            continue;
        for (const QNetworkAddressEntry &entry : iface.addressEntries()) {
            const QHostAddress ip = entry.ip();
            if (ip.protocol() != QAbstractSocket::IPv4Protocol)
                continue;
            // Default to the local /24 to keep the initial scan short.
            const quint32 base = ip.toIPv4Address() & 0xFFFFFF00u;
            m_startIpEdit->setText(QHostAddress(base | 1u).toString());
            m_endIpEdit->setText(QHostAddress(base | 254u).toString());
            return;
        }
    }
}

// -----------------------------------------------------------------------------
void RtspDiscoveryDialog::startScan()
{
    if (m_scanning)
        return;

    quint32 startIp = 0, endIp = 0;
    if (!parseIPv4(m_startIpEdit->text(), startIp) || !parseIPv4(m_endIpEdit->text(), endIp)) {
        QMessageBox::warning(this, windowTitle(), QStringLiteral("Please enter valid IPv4 start and end addresses."));
        return;
    }
    if (startIp > endIp)
        std::swap(startIp, endIp);
    if (endIp - startIp >= kMaxHosts) {
        QMessageBox::warning(this, windowTitle(), QStringLiteral("Range too large (max %1 addresses).").arg(kMaxHosts));
        return;
    }

    QList<quint16> ports;
    const QStringList parts = m_portsEdit->text().split(QRegularExpression(QStringLiteral("[,;\\s]+")), Qt::SkipEmptyParts);
    for (const QString &p : parts) {
        bool ok = false;
        const uint v = p.toUInt(&ok);
        if (!ok || v == 0 || v > 65535) {
            QMessageBox::warning(this, windowTitle(), QStringLiteral("Invalid port: %1").arg(p));
            return;
        }
        if (!ports.contains(static_cast<quint16>(v)))
            ports.append(static_cast<quint16>(v));
    }
    if (ports.isEmpty() || ports.size() > kMaxPorts) {
        QMessageBox::warning(this, windowTitle(), QStringLiteral("Specify between 1 and %1 ports.").arg(kMaxPorts));
        return;
    }

    m_targets.clear();
    for (quint64 ip = startIp; ip <= endIp; ++ip)
        for (quint16 port : ports)
            m_targets.append({static_cast<quint32>(ip), port});

    m_table->setRowCount(0);
    m_nextTarget = 0;
    m_done = 0;
    m_progress->setRange(0, m_targets.size());
    m_progress->setValue(0);
    m_statusLabel->setText(QStringLiteral("Scanning %1 endpoints…").arg(m_targets.size()));
    setScanning(true);
    launchProbes();
}

void RtspDiscoveryDialog::stopScan()
{
    if (!m_scanning)
        return;
    m_scanning = false;
    m_nextTarget = m_targets.size();
    const auto sockets = findChildren<QTcpSocket *>(QString(), Qt::FindDirectChildrenOnly);
    for (QTcpSocket *sock : sockets)
        finishProbe(sock, false, QByteArray());
    setScanning(false);
    m_statusLabel->setText(QStringLiteral("Stopped – %1 found").arg(m_table->rowCount()));
}

void RtspDiscoveryDialog::setScanning(bool scanning)
{
    m_scanning = scanning;
    m_scanBtn->setEnabled(!scanning);
    m_stopBtn->setEnabled(scanning);
    m_startIpEdit->setEnabled(!scanning);
    m_endIpEdit->setEnabled(!scanning);
    m_portsEdit->setEnabled(!scanning);
    m_timeoutSpin->setEnabled(!scanning);
    m_parallelSpin->setEnabled(!scanning);
}

// -----------------------------------------------------------------------------
void RtspDiscoveryDialog::launchProbes()
{
    while (m_scanning && m_active < m_parallelSpin->value() && m_nextTarget < m_targets.size())
        startProbe(m_targets.at(m_nextTarget++));

    if (m_scanning && m_active == 0 && m_nextTarget >= m_targets.size()) {
        setScanning(false);
        m_statusLabel->setText(QStringLiteral("Done – %1 found").arg(m_table->rowCount()));
    }
}

void RtspDiscoveryDialog::startProbe(const Target &t)
{
    auto *sock = new QTcpSocket(this);
    auto *timer = new QTimer(sock);
    auto buffer = std::make_shared<QByteArray>();
    const QString host = QHostAddress(t.ip).toString();
    const quint16 port = t.port;
    sock->setProperty("host", host);
    sock->setProperty("port", port);
    ++m_active;

    timer->setSingleShot(true);
    connect(timer, &QTimer::timeout, sock, [this, sock, buffer]() {
        finishProbe(sock, sock->state() == QAbstractSocket::ConnectedState, *buffer);
    });
    connect(sock, &QTcpSocket::connected, sock, [sock, host, port]() {
        const QByteArray req =
            "OPTIONS rtsp://" + host.toLatin1() + ':' + QByteArray::number(port) + "/ RTSP/1.0\r\nCSeq: 1\r\nUser-Agent: QtRtspViewer\r\n\r\n";
        sock->write(req);
    });
    connect(sock, &QTcpSocket::readyRead, sock, [this, sock, buffer]() {
        buffer->append(sock->read(kMaxReplyBytes - buffer->size()));
        if (buffer->contains("\r\n\r\n") || buffer->size() >= kMaxReplyBytes)
            finishProbe(sock, true, *buffer);
    });
    connect(sock, &QTcpSocket::errorOccurred, sock, [this, sock, buffer](QAbstractSocket::SocketError) {
        // A peer closing after replying still counts as an open port.
        finishProbe(sock, !buffer->isEmpty(), *buffer);
    });

    sock->connectToHost(host, port);
    timer->start(m_timeoutSpin->value());
}

void RtspDiscoveryDialog::finishProbe(QTcpSocket *sock, bool open, const QByteArray &reply)
{
    if (sock->property("done").toBool())
        return;
    sock->setProperty("done", true);
    sock->disconnect();
    if (auto *timer = sock->findChild<QTimer *>())
        timer->stop();

    const QString host = sock->property("host").toString();
    const quint16 port = static_cast<quint16>(sock->property("port").toUInt());
    sock->abort();
    sock->deleteLater();

    --m_active;
    ++m_done;
    m_progress->setValue(m_done);

    if (open) {
        const bool rtsp = reply.startsWith("RTSP/");
        QString server;
        if (rtsp) {
            const QList<QByteArray> lines = reply.split('\n');
            for (const QByteArray &line : lines) {
                if (line.toLower().startsWith("server:")) {
                    server = QString::fromLatin1(line.mid(7).trimmed());
                    break;
                }
            }
        }
        addResult(host, port, rtsp, server);
    }

    launchProbes();
}

void RtspDiscoveryDialog::addResult(const QString &host, quint16 port, bool rtsp, const QString &server)
{
    const int row = m_table->rowCount();
    m_table->insertRow(row);
    m_table->setItem(row, ColHost, new QTableWidgetItem(host));
    m_table->setItem(row, ColPort, new QTableWidgetItem(QString::number(port)));
    m_table->setItem(row, ColStatus, new QTableWidgetItem(rtsp ? QStringLiteral("RTSP") : QStringLiteral("Port open (no RTSP reply)")));
    m_table->setItem(row, ColServer, new QTableWidgetItem(server));
    m_table->resizeColumnsToContents();
}

// -----------------------------------------------------------------------------
QString RtspDiscoveryDialog::urlForRow(int row) const
{
    return buildRtspUrl(m_table->item(row, ColHost)->text(),
                        static_cast<quint16>(m_table->item(row, ColPort)->text().toUInt()),
                        m_pathEdit->text(),
                        m_userEdit->text(),
                        m_passEdit->text())
        .toString();
}

void RtspDiscoveryDialog::openSelected()
{
    const int row = m_table->currentRow();
    if (row >= 0)
        emit openUrlRequested(urlForRow(row));
}

void RtspDiscoveryDialog::copySelectedUrl()
{
    const int row = m_table->currentRow();
    if (row >= 0)
        QApplication::clipboard()->setText(urlForRow(row));
}

// -----------------------------------------------------------------------------
// Stream URL testing
QList<RtspDiscoveryDialog::UrlCandidate> RtspDiscoveryDialog::buildCandidates(const QString &host) const
{
    QList<UrlCandidate> list;
    QSet<QString> seen;
    const QString fieldUser = m_userEdit->text();
    const QString fieldPass = m_passEdit->text();

    auto add = [&](QString path, const QString &user, const QString &pass) {
        path = path.trimmed();
        if (!path.startsWith(QLatin1Char('/')))
            path.prepend(QLatin1Char('/'));
        if (seen.contains(path))
            return;
        seen.insert(path);
        list.append({path, user, pass});
    };

    if (!m_pathEdit->text().trimmed().isEmpty())
        add(m_pathEdit->text(), fieldUser, fieldPass);

    QStringList used;
    const QString last = StreamStateManager::instance().lastPlayedUrl();
    if (!last.isEmpty())
        used.append(last);
    const auto history = StreamStateManager::instance().urlHistory();
    for (const auto &e : history)
        used.append(e.url);

    // Previously used URLs: same host first (reusing their stored credentials), then other hosts' paths.
    for (int round = 0; round < 2; ++round) {
        for (const QString &s : std::as_const(used)) {
            const QUrl u(s.trimmed());
            if (!u.isValid() || u.scheme().compare(QStringLiteral("rtsp"), Qt::CaseInsensitive) != 0)
                continue;
            const bool sameHost = u.host().compare(host, Qt::CaseInsensitive) == 0;
            if (sameHost != (round == 0))
                continue;
            QString path = u.path();
            if (u.hasQuery())
                path += QLatin1Char('?') + u.query(QUrl::FullyEncoded);
            const bool useStored = fieldUser.isEmpty() && sameHost;
            add(path, useStored ? u.userName() : fieldUser, useStored ? u.password() : fieldPass);
        }
    }

    for (const char *p : kCommonPaths)
        add(QString::fromLatin1(p), fieldUser, fieldPass);
    return list;
}

void RtspDiscoveryDialog::testSelectedHost()
{
    const int row = m_table->currentRow();
    if (row < 0)
        return;

    stopUrlTests();
    m_testHost = m_table->item(row, ColHost)->text();
    m_testPort = static_cast<quint16>(m_table->item(row, ColPort)->text().toUInt());
    m_candidates = buildCandidates(m_testHost);
    m_nextCandidate = 0;
    m_testActive = 0;
    m_testDone = 0;
    m_testOk = 0;
    m_testTable->setRowCount(0);
    m_testStatusLabel->setText(QStringLiteral("Testing %1 URLs on %2:%3…").arg(m_candidates.size()).arg(m_testHost).arg(m_testPort));
    launchUrlTests();
}

void RtspDiscoveryDialog::stopUrlTests()
{
    const auto sockets = m_testRoot->findChildren<QTcpSocket *>(QString(), Qt::FindDirectChildrenOnly);
    for (QTcpSocket *sock : sockets) {
        sock->disconnect();
        sock->abort();
        sock->deleteLater();
    }
    m_nextCandidate = m_candidates.size();
    m_testActive = 0;
}

void RtspDiscoveryDialog::launchUrlTests()
{
    while (m_testActive < kParallelUrlTests && m_nextCandidate < m_candidates.size()) {
        ++m_testActive;
        runDescribe(m_nextCandidate++, QByteArray());
    }
}

void RtspDiscoveryDialog::runDescribe(int index, const QByteArray &authorization)
{
    const UrlCandidate c = m_candidates.at(index);
    const QByteArray uri = buildRtspUrl(m_testHost, m_testPort, c.path).toEncoded();
    auto *sock = new QTcpSocket(m_testRoot);
    auto *timer = new QTimer(sock);
    auto buffer = std::make_shared<QByteArray>();

    auto finish = [this, sock, timer, index, c, uri, authorization](int code, const QString &reason, const QByteArray &headers) {
        sock->disconnect();
        timer->stop();
        sock->abort();
        sock->deleteLater();
        if (code == 401 && authorization.isEmpty() && !c.user.isEmpty()) {
            const QByteArray auth = buildAuthorization(headers, "DESCRIBE", uri, c.user, c.pass);
            if (!auth.isEmpty()) {
                runDescribe(index, auth);
                return;
            }
        }
        finishUrlTest(index, code, reason, !authorization.isEmpty());
    };

    timer->setSingleShot(true);
    connect(timer, &QTimer::timeout, sock, [finish]() {
        finish(-1, QStringLiteral("No response (timeout)"), QByteArray());
    });
    connect(sock, &QTcpSocket::connected, sock, [sock, uri, authorization]() {
        QByteArray req =
            "DESCRIBE " + uri + " RTSP/1.0\r\nCSeq: " + (authorization.isEmpty() ? "2" : "3") + "\r\nAccept: application/sdp\r\nUser-Agent: QtRtspViewer\r\n";
        if (!authorization.isEmpty())
            req += "Authorization: " + authorization + "\r\n";
        req += "\r\n";
        sock->write(req);
    });
    connect(sock, &QTcpSocket::readyRead, sock, [sock, buffer, finish]() {
        buffer->append(sock->read(kMaxReplyBytes - buffer->size()));
        const int end = buffer->indexOf("\r\n\r\n");
        if (end < 0) {
            if (buffer->size() >= kMaxReplyBytes)
                finish(-1, QStringLiteral("Reply too large"), QByteArray());
            return;
        }
        const QByteArray headers = buffer->left(end);
        const int eol = headers.indexOf("\r\n");
        const QByteArray statusLine = eol < 0 ? headers : headers.left(eol);
        if (!statusLine.startsWith("RTSP/")) {
            finish(-1, QStringLiteral("Not an RTSP reply"), headers);
            return;
        }
        const QList<QByteArray> parts = statusLine.split(' ');
        const int code = parts.size() > 1 ? parts.at(1).toInt() : -1;
        finish(code, QString::fromLatin1(parts.mid(2).join(' ')), headers);
    });
    connect(sock, &QTcpSocket::errorOccurred, sock, [sock, finish](QAbstractSocket::SocketError) {
        finish(-1, sock->errorString(), QByteArray());
    });

    sock->connectToHost(m_testHost, m_testPort);
    timer->start(qMax(2000, m_timeoutSpin->value() * 3));
}

void RtspDiscoveryDialog::finishUrlTest(int index, int code, const QString &reason, bool authSent)
{
    --m_testActive;
    ++m_testDone;
    const UrlCandidate &c = m_candidates.at(index);

    QString result;
    QColor color(Qt::gray);
    switch (code) {
    case 200:
        result = QStringLiteral("OK – stream available");
        color = QColor(0, 150, 0);
        ++m_testOk;
        break;
    case 401:
        if (authSent)
            result = QStringLiteral("Unauthorized – wrong credentials");
        else if (c.user.isEmpty())
            result = QStringLiteral("Auth required – enter username/password");
        else
            result = QStringLiteral("Unauthorized – unsupported auth scheme");
        color = QColor(200, 120, 0);
        break;
    case 404:
        result = QStringLiteral("Not found");
        break;
    default:
        if (code > 0) {
            result = QStringLiteral("RTSP %1 %2").arg(code).arg(reason);
        } else {
            result = reason;
            color = QColor(180, 0, 0);
        }
        break;
    }

    // Working URLs go to the top so they are easy to spot.
    const int row = (code == 200) ? 0 : m_testTable->rowCount();
    m_testTable->insertRow(row);
    auto *pathItem = new QTableWidgetItem(c.path);
    pathItem->setData(Qt::UserRole, buildRtspUrl(m_testHost, m_testPort, c.path, c.user, c.pass).toString());
    auto *resultItem = new QTableWidgetItem(result);
    resultItem->setForeground(QBrush(color));
    m_testTable->setItem(row, 0, pathItem);
    m_testTable->setItem(row, 1, resultItem);
    m_testTable->resizeColumnToContents(0);

    if (m_testDone >= m_candidates.size())
        m_testStatusLabel->setText(
            QStringLiteral("Done – %1 of %2 URLs working on %3:%4").arg(m_testOk).arg(m_candidates.size()).arg(m_testHost).arg(m_testPort));
    else
        launchUrlTests();
}

void RtspDiscoveryDialog::openSelectedTestUrl()
{
    const int row = m_testTable->currentRow();
    if (row >= 0)
        emit openUrlRequested(m_testTable->item(row, 0)->data(Qt::UserRole).toString());
}
