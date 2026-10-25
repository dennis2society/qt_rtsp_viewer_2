#!/usr/bin/env bash
# Package QtRtspViewer as a Linux AppImage using linuxdeploy + linuxdeploy-plugin-qt.
# Does not build: it packages the existing Release build in <repo>/build.
#
# Usage: zz_packaging/build_appimage.sh [--clean]
#   --clean       remove the AppImage work directory first
#
# Env overrides: RELEASE_BUILD_DIR (default <repo>/build), BUILD_DIR (work dir), QMAKE
#
# NOTE: an AppImage does not bundle glibc. Build on the OLDEST distro you want to
# support (e.g. Ubuntu 22.04 / glibc 2.35); an image built on a newer host will
# not start on older systems.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${BUILD_DIR:-$SCRIPT_DIR/build-appimage}"
RELEASE_BUILD_DIR="${RELEASE_BUILD_DIR:-$SRC_DIR/build}"
TOOLS_DIR="$SCRIPT_DIR/tools"
APPDIR="$BUILD_DIR/AppDir"
ARCH="${ARCH:-$(uname -m)}"
APP_NAME="QtRtspViewer"

CLEAN=0
for arg in "$@"; do
    case "$arg" in
        --clean)      CLEAN=1 ;;
        -h|--help)    sed -n '2,8p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown option: $arg" >&2; exit 1 ;;
    esac
done

VERSION="$(sed -n 's/^project([A-Za-z0-9_]* VERSION \([0-9.]*\).*/\1/p' "$SRC_DIR/CMakeLists.txt")"
VERSION="${VERSION:-0.0.0}"

for cmd in wget file; do
    command -v "$cmd" >/dev/null || { echo "Missing required tool: $cmd" >&2; exit 1; }
done

QMAKE="${QMAKE:-$(command -v qmake6 || command -v qmake || true)}"
[[ -n "$QMAKE" ]] || { echo "qmake6 not found; set QMAKE=/path/to/qmake6" >&2; exit 1; }

HOST_GLIBC="$(ldd --version | head -n1 | grep -oE '[0-9]+\.[0-9]+$' || true)"
echo "==> Host glibc: ${HOST_GLIBC:-unknown} (the AppImage will require at least this version)"

[[ $CLEAN -eq 1 ]] && rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR" "$TOOLS_DIR"

# ── Existing Release build ───────────────────────────────────────────────────
BIN="$RELEASE_BUILD_DIR/bin/$APP_NAME"
[[ -x "$BIN" ]] || { echo "Binary not found: $BIN (build it first: cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build)" >&2; exit 1; }
if ! grep -q '^CMAKE_BUILD_TYPE:STRING=Release$' "$RELEASE_BUILD_DIR/CMakeCache.txt" 2>/dev/null; then
    echo "WARNING: $RELEASE_BUILD_DIR is not configured as a Release build" >&2
fi
echo "==> Packaging $BIN"
ldd "$BIN" | grep -q libavformat || echo "WARNING: binary is not linked against FFmpeg; recording will be unavailable" >&2

# ── Packaging tools ──────────────────────────────────────────────────────────
fetch_tool() {
    local name="$1" url="$2"
    if [[ ! -x "$TOOLS_DIR/$name" ]]; then
        echo "==> Downloading $name"
        wget -q --show-progress -O "$TOOLS_DIR/$name.tmp" "$url"
        chmod +x "$TOOLS_DIR/$name.tmp"
        mv "$TOOLS_DIR/$name.tmp" "$TOOLS_DIR/$name"
    fi
}
fetch_tool "linuxdeploy-$ARCH.AppImage" \
    "https://github.com/linuxdeploy/linuxdeploy/releases/download/continuous/linuxdeploy-$ARCH.AppImage"
fetch_tool "linuxdeploy-plugin-qt-$ARCH.AppImage" \
    "https://github.com/linuxdeploy/linuxdeploy-plugin-qt/releases/download/continuous/linuxdeploy-plugin-qt-$ARCH.AppImage"

# ── AppDir ───────────────────────────────────────────────────────────────────
echo "==> Preparing AppDir"
rm -rf "$APPDIR"
mkdir -p "$APPDIR/usr/bin" "$APPDIR/usr/share/applications" \
         "$APPDIR/usr/share/icons/hicolor/128x128/apps"

install -m755 "$BIN" "$APPDIR/usr/bin/$APP_NAME"
# The app loads the cascade from <exe dir>/opencv/
cp -r "$SRC_DIR/opencv" "$APPDIR/usr/bin/opencv"
install -m644 "$SRC_DIR/res/qt_rtsp_viewer_icon.png" \
    "$APPDIR/usr/share/icons/hicolor/128x128/apps/$APP_NAME.png"

cat > "$APPDIR/usr/share/applications/$APP_NAME.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=Qt RTSP Viewer
Comment=Multi-stream RTSP viewer with motion detection and recording
Exec=$APP_NAME
Icon=$APP_NAME
Categories=AudioVideo;Video;Player;
Terminal=false
EOF

# ── Deploy + create AppImage ─────────────────────────────────────────────────
echo "==> Running linuxdeploy"
export APPIMAGE_EXTRACT_AND_RUN=1          # no FUSE needed to run the tools
export PATH="$TOOLS_DIR:$PATH"             # lets linuxdeploy find the qt plugin
export QMAKE
export NO_STRIP=1                          # bundled strip chokes on modern .relr.dyn sections
export EXTRA_QT_PLUGINS="multimedia;tls;networkinformation"
# Wayland plugin naming differs between Qt versions (libqwayland.so vs. -egl/-generic).
QT_PLUGINS_DIR="$("$QMAKE" -query QT_INSTALL_PLUGINS)"
WAYLAND_PLUGINS=()
for p in libqwayland.so libqwayland-egl.so libqwayland-generic.so; do
    [[ -f "$QT_PLUGINS_DIR/platforms/$p" ]] && WAYLAND_PLUGINS+=("$p")
done
if [[ ${#WAYLAND_PLUGINS[@]} -gt 0 ]]; then
    export EXTRA_PLATFORM_PLUGINS="$(IFS=';'; echo "${WAYLAND_PLUGINS[*]}")"
    EXTRA_QT_PLUGINS+=";wayland-shell-integration;wayland-graphics-integration-client;wayland-decoration-client"
else
    echo "WARNING: qt6-wayland not found; the AppImage will be X11/XWayland only" >&2
fi
export LINUXDEPLOY_OUTPUT_VERSION="$VERSION"
export ARCH

cd "$BUILD_DIR"
"$TOOLS_DIR/linuxdeploy-$ARCH.AppImage" \
    --appdir "$APPDIR" \
    --executable "$APPDIR/usr/bin/$APP_NAME" \
    --desktop-file "$APPDIR/usr/share/applications/$APP_NAME.desktop" \
    --icon-file "$APPDIR/usr/share/icons/hicolor/128x128/apps/$APP_NAME.png" \
    --plugin qt \
    --output appimage

OUT_FILE="$(ls -t "$BUILD_DIR"/*.AppImage | head -n1)"
FINAL="$SRC_DIR/$APP_NAME-$VERSION-$ARCH.AppImage"
mv -f "$OUT_FILE" "$FINAL"
echo "==> Done: $FINAL"
