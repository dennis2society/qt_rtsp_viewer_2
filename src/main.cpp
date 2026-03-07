#include "mainwindow.h"
#include "streamstatemanager.h"

#include <QApplication>
#include <QIcon>

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    app.setApplicationName(QStringLiteral("QtRtspViewer"));
    app.setOrganizationName(QStringLiteral("QtRtspViewer"));
    app.setWindowIcon(QIcon(QStringLiteral(":/qt_rtsp_viewer_icon.png")));

    // Ensure the singleton is alive and settings are loaded
    StreamStateManager::instance();

    MainWindow w;
    w.show();

    return app.exec();
}
