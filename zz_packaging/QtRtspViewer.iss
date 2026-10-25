; Compiled by build_windows_installer.ps1, which passes AppVersion, StageDir, SourceRoot and OutDir.
#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef StageDir
  #error StageDir must be defined (ISCC /DStageDir=...)
#endif
#ifndef SourceRoot
  #define SourceRoot ".."
#endif
#ifndef OutDir
  #define OutDir "dist"
#endif

#define AppName "Qt RTSP Viewer"
#define AppExe "QtRtspViewer.exe"

[Setup]
AppId={{8F3B2C1A-5D4E-4A7B-9C6D-2E1F0A3B4C5D}
AppName={#AppName}
AppVersion={#AppVersion}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
UninstallDisplayIcon={app}\{#AppExe}
LicenseFile={#SourceRoot}\LICENSE
OutputDir={#OutDir}
OutputBaseFilename=QtRtspViewer-{#AppVersion}-win64-setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequiredOverridesAllowed=dialog

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "{#StageDir}\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent
