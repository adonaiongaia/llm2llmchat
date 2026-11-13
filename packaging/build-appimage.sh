#!/usr/bin/env bash
# Build a self-contained AppImage for LLM2LLMChat.
#
# Requirements:
#   - A virtualenv with the app and PyInstaller installed:
#       python3 -m venv .venv
#       . .venv/bin/activate
#       python -m pip install -e '.[keyring,build]'
#
# Usage:
#   ./packaging/build-appimage.sh
#
# The finished bundle is written to dist/LLM2LLMChat-<version>-x86_64.AppImage.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
APP_NAME="LLM2LLMChat"
BINARY="llm2llmchat"
ARCH="$(uname -m)"
BUILD_DIR="$ROOT/build/appimage"
APPDIR="$BUILD_DIR/AppDir"
DIST_DIR="$ROOT/dist"
CACHE_DIR="$ROOT/packaging/.cache"
APPIMAGETOOL="$CACHE_DIR/appimagetool-$ARCH.AppImage"
APPIMAGETOOL_URL="https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-$ARCH.AppImage"

if [ ! -x "$PYTHON" ]; then
    echo "error: no Python interpreter at $PYTHON" >&2
    echo "       create the project virtualenv first, or set PYTHON=/path/to/python" >&2
    exit 1
fi
if ! "$PYTHON" -c 'import PyInstaller' >/dev/null 2>&1; then
    echo "error: PyInstaller is not installed for $PYTHON" >&2
    echo "       run: $PYTHON -m pip install -e '.[keyring,build]'" >&2
    exit 1
fi

VERSION="$("$PYTHON" -c 'import sys; sys.path.insert(0, "src"); import llm2llmchat; print(llm2llmchat.__version__)')"
echo "Building $APP_NAME $VERSION ($ARCH)"

rm -rf "$APPDIR"
mkdir -p "$APPDIR/usr/bin" \
         "$APPDIR/usr/share/applications" \
         "$APPDIR/usr/share/icons/hicolor/256x256/apps" \
         "$BUILD_DIR/pyinstaller" \
         "$DIST_DIR"

export PYINSTALLER_CONFIG_DIR="$BUILD_DIR/pyinstaller-cache"

echo "Bundling Python and Qt with PyInstaller..."
"$PYTHON" -m PyInstaller \
    --noconfirm \
    --clean \
    --windowed \
    --name "$BINARY" \
    --paths "$ROOT/src" \
    --distpath "$APPDIR/usr/bin" \
    --workpath "$BUILD_DIR/pyinstaller/work" \
    --specpath "$BUILD_DIR/pyinstaller" \
    --copy-metadata keyring \
    --copy-metadata SecretStorage \
    --hidden-import keyring.backends.chainer \
    --hidden-import keyring.backends.SecretService \
    --hidden-import keyring.backends.libsecret \
    --hidden-import keyring.backends.kwallet \
    --hidden-import keyring.backends.fail \
    --hidden-import keyring.backends.null \
    --collect-submodules secretstorage \
    --exclude-module PySide6.QtWebEngineCore \
    --exclude-module PySide6.QtWebEngineWidgets \
    --exclude-module PySide6.QtWebEngineQuick \
    --exclude-module PySide6.QtWebChannel \
    --exclude-module PySide6.QtQml \
    --exclude-module PySide6.QtQuick \
    --exclude-module PySide6.QtQuickWidgets \
    --exclude-module PySide6.QtQuick3D \
    --exclude-module PySide6.Qt3DCore \
    --exclude-module PySide6.Qt3DRender \
    --exclude-module PySide6.Qt3DInput \
    --exclude-module PySide6.Qt3DLogic \
    --exclude-module PySide6.Qt3DAnimation \
    --exclude-module PySide6.Qt3DExtras \
    --exclude-module PySide6.QtMultimedia \
    --exclude-module PySide6.QtMultimediaWidgets \
    --exclude-module PySide6.QtCharts \
    --exclude-module PySide6.QtDataVisualization \
    --exclude-module PySide6.QtGraphs \
    --exclude-module PySide6.QtPdf \
    --exclude-module PySide6.QtPdfWidgets \
    --exclude-module PySide6.QtDesigner \
    "$ROOT/packaging/entrypoint.py"

echo "Assembling AppDir..."
install -m 755 "$ROOT/packaging/AppRun" "$APPDIR/AppRun"
install -m 644 "$ROOT/packaging/llm2llmchat.desktop" "$APPDIR/usr/share/applications/llm2llmchat.desktop"
install -m 644 "$ROOT/packaging/llm2llmchat.desktop" "$APPDIR/llm2llmchat.desktop"
"$PYTHON" "$ROOT/packaging/make_icon.py" "$ROOT/packaging/llm2llmchat.svg" "$APPDIR/llm2llmchat.png" 256
install -m 644 "$APPDIR/llm2llmchat.png" "$APPDIR/usr/share/icons/hicolor/256x256/apps/llm2llmchat.png"
ln -sfn llm2llmchat.png "$APPDIR/.DirIcon"

if command -v desktop-file-validate >/dev/null 2>&1; then
    desktop-file-validate "$APPDIR/usr/share/applications/llm2llmchat.desktop"
fi

if [ ! -x "$APPIMAGETOOL" ]; then
    echo "Downloading appimagetool..."
    mkdir -p "$CACHE_DIR"
    curl -fL --retry 3 -o "$APPIMAGETOOL" "$APPIMAGETOOL_URL"
    chmod +x "$APPIMAGETOOL"
fi

OUTPUT="$DIST_DIR/${APP_NAME}-${VERSION}-${ARCH}.AppImage"
echo "Packing AppImage..."
rm -f "$OUTPUT"
ARCH="$ARCH" APPIMAGE_EXTRACT_AND_RUN=1 "$APPIMAGETOOL" --no-appstream "$APPDIR" "$OUTPUT"
chmod +x "$OUTPUT"

echo
echo "Built $OUTPUT ($(du -h "$OUTPUT" | cut -f1))"
