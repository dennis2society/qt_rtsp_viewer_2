#include "mainwindow.h"
#include "streamstatemanager.h"

#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    app.setApplicationName(QStringLiteral("QtRtspViewer"));
    app.setOrganizationName(QStringLiteral("QtRtspViewer"));

    // Ensure the singleton is alive and settings are loaded
    StreamStateManager::instance();

    MainWindow w;
    w.show();

    return app.exec();
}
