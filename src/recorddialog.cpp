#include "recorddialog.h"

#include <QComboBox>
#include <QDateTime>
#include <QDialogButtonBox>
#include <QDir>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QRegularExpression>
#include <QVBoxLayout>

// -----------------------------------------------------------------------------
const QList<RecordDialog::Preset> RecordDialog::s_presets = {
    {QStringLiteral("MP4 - H.264 (NVENC)"), QStringLiteral("mp4"), QStringLiteral("h264_nvenc")},
    {QStringLiteral("MP4 - H.265/HEVC (NVENC)"), QStringLiteral("mp4"), QStringLiteral("hevc_nvenc")},
    {QStringLiteral("MP4 - H.264 (auto)"), QStringLiteral("mp4"), QStringLiteral("libx264")},
    {QStringLiteral("MP4 - H.265/HEVC (auto)"), QStringLiteral("mp4"), QStringLiteral("libx265")},
    {QStringLiteral("MKV - H.264 (NVENC)"), QStringLiteral("mkv"), QStringLiteral("h264_nvenc")},
    {QStringLiteral("MKV - H.265/HEVC (NVENC)"), QStringLiteral("mkv"), QStringLiteral("hevc_nvenc")},
    {QStringLiteral("MKV - H.264 (auto)"), QStringLiteral("mkv"), QStringLiteral("libx264")},
    {QStringLiteral("MKV - H.265/HEVC (auto)"), QStringLiteral("mkv"), QStringLiteral("libx265")},
    {QStringLiteral("AVI - H.264 (NVENC)"), QStringLiteral("avi"), QStringLiteral("h264_nvenc")},
    {QStringLiteral("AVI - H.264 (auto)"), QStringLiteral("avi"), QStringLiteral("libx264")},
};

// -----------------------------------------------------------------------------
RecordDialog::RecordDialog(const QString &defaultDir, const QString &cameraName, QWidget *parent)
    : QDialog(parent)
{
    setWindowTitle(QStringLiteral("Recording Settings"));
    setMinimumWidth(480);
    setupUI(defaultDir, cameraName);
}

void RecordDialog::setupUI(const QString &defaultDir, const QString &cameraName)
{
    auto *form = new QFormLayout;

    // Preset (codec + container)
    m_presetCombo = new QComboBox;
    for (const auto &p : s_presets)
        m_presetCombo->addItem(p.label);
    form->addRow(QStringLiteral("Format / Codec:"), m_presetCombo);

    // Output path
    auto *pathRow = new QHBoxLayout;
    m_pathEdit = new QLineEdit;
    auto *browseBtn = new QPushButton(QStringLiteral("Browse..."));
    pathRow->addWidget(m_pathEdit, 1);
    pathRow->addWidget(browseBtn);
    form->addRow(QStringLiteral("Output file:"), pathRow);

    // Generate default path
    QString ts = QDateTime::currentDateTime().toString(QStringLiteral("yyyy-MM-dd_HH-mm-ss"));
    QString cam = cameraName;
    cam.replace(QRegularExpression(QStringLiteral("[^a-zA-Z0-9_-]")), QStringLiteral("_"));
    QString dir = defaultDir.isEmpty() ? QDir::homePath() : defaultDir;
    QString ext = s_presets[0].ext;
    m_pathEdit->setText(QStringLiteral("%1/%2_%3_recording.%4").arg(dir, ts, cam, ext));

    // Update extension when preset changes
    connect(m_presetCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int idx) {
        if (idx < 0 || idx >= s_presets.size())
            return;
        QString path = m_pathEdit->text();
        int dot = path.lastIndexOf(QLatin1Char('.'));
        if (dot > 0)
            path = path.left(dot + 1) + s_presets[idx].ext;
        m_pathEdit->setText(path);
    });

    connect(browseBtn, &QPushButton::clicked, this, &RecordDialog::onBrowse);

    // FPS
    m_fpsSpin = new QDoubleSpinBox;
    m_fpsSpin->setRange(1.0, 120.0);
    m_fpsSpin->setValue(25.0);
    m_fpsSpin->setDecimals(1);
    form->addRow(QStringLiteral("Frame rate:"), m_fpsSpin);

    // Buttons
    auto *buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
    connect(buttons, &QDialogButtonBox::accepted, this, &QDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);

    auto *mainLayout = new QVBoxLayout(this);
    mainLayout->addLayout(form);
    mainLayout->addWidget(buttons);
    setLayout(mainLayout);
}

void RecordDialog::onBrowse()
{
    int idx = m_presetCombo->currentIndex();
    QString ext = (idx >= 0 && idx < s_presets.size()) ? s_presets[idx].ext : QStringLiteral("mp4");
    QString filter = QStringLiteral("Video (*.%1);;All files (*)").arg(ext);
    QString path = QFileDialog::getSaveFileName(this, QStringLiteral("Save recording"), m_pathEdit->text(), filter);
    if (!path.isEmpty())
        m_pathEdit->setText(path);
}

QString RecordDialog::filePath() const
{
    return m_pathEdit->text();
}

QString RecordDialog::codec() const
{
    int idx = m_presetCombo->currentIndex();
    return (idx >= 0 && idx < s_presets.size()) ? s_presets[idx].codec : QStringLiteral("libx264");
}

double RecordDialog::fps() const
{
    return m_fpsSpin->value();
}
