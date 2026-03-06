#pragma once

#include <QDialog>

class QComboBox;
class QLineEdit;
class QDoubleSpinBox;

/// Modal dialog for choosing recording parameters (manual recording).
class RecordDialog : public QDialog {
    Q_OBJECT

public:
    explicit RecordDialog(const QString &defaultDir,
                          const QString &cameraName,
                          QWidget *parent = nullptr);

    QString filePath() const;
    QString codec()    const;
    double  fps()      const;

private:
    void setupUI(const QString &defaultDir, const QString &cameraName);
    void onBrowse();

    QComboBox      *m_presetCombo  = nullptr;
    QLineEdit      *m_pathEdit     = nullptr;
    QDoubleSpinBox *m_fpsSpin      = nullptr;

    struct Preset {
        QString label;
        QString ext;
        QString codec;
    };
    static const QList<Preset> s_presets;
};
