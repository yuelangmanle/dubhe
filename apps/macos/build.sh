#!/bin/bash
# 把 Dubhe 打成 macOS App 包（.app），不依赖 Xcode 工程。
#
#   ./apps/macos/build.sh               # 出 build/Dubhe.app
#   ./apps/macos/build.sh --run         # 出完直接打开
#   ./apps/macos/build.sh --dmg         # 再打一个 build/Dubhe.dmg
#   DUBHE_SIGN_IDENTITY='Developer ID Application: …' \
#     DUBHE_NOTARY_PROFILE=dubhe ./apps/macos/build.sh --dmg --notarize
#
# 产物结构：
#   build/Dubhe.app/Contents/MacOS/Dubhe            外壳（swiftc 编出来，~200 KB）
#   build/Dubhe.app/Contents/Helpers/dubhed         引擎
#   build/Dubhe.app/Contents/Resources/web/         Web UI
set -euo pipefail
cd "$(dirname "$0")/../.."

OUT_DIR=${BUILD_DIR:-build}
APP="$OUT_DIR/Dubhe.app"
IMAGE="$OUT_DIR/Dubhe.dmg"
PLIST=apps/macos/Resources/Info.plist
NAME=Dubhe
SIGN_IDENTITY=${DUBHE_SIGN_IDENTITY:-}
NOTARY_PROFILE=${DUBHE_NOTARY_PROFILE:-}
LLAMA_RUNTIME=${DUBHE_LLAMA_RUNTIME_DIR:-}
RUN=0
DMG=0
NOTARIZE=0
for a in "$@"; do
  case "$a" in
    --run) RUN=1 ;;
    --dmg) DMG=1 ;;
    --notarize) NOTARIZE=1 ;;
    *) echo "未知参数: $a" >&2; exit 2 ;;
  esac
done

# ---------------------------------------------------------------- 前置检查
if [[ -n "$SIGN_IDENTITY" && "$SIGN_IDENTITY" != "Developer ID Application: "* && "$SIGN_IDENTITY" != "Apple Development: "* ]]; then
  echo "DUBHE_SIGN_IDENTITY 应为 Developer ID Application 或 Apple Development 证书的完整名称。" >&2
  exit 2
fi
if [[ "$NOTARIZE" = 1 && ( "$SIGN_IDENTITY" != "Developer ID Application: "* || -z "$NOTARY_PROFILE" ) ]]; then
  echo "--notarize 需要设置 DUBHE_SIGN_IDENTITY 和 DUBHE_NOTARY_PROFILE。" >&2
  exit 2
fi
if [ ! -x "$OUT_DIR/dubhed" ]; then
  echo "找不到 ${OUT_DIR}/dubhed，请先用 make app 或 make dmg 构建。" >&2
  exit 1
fi
if [ ! -f web/index.html ]; then
  echo "缺 web/index.html —— 前端资源不完整" >&2
  exit 1
fi
if ! command -v swiftc >/dev/null 2>&1; then
  echo "缺 swiftc —— 装 Xcode 或 Command Line Tools" >&2
  exit 1
fi

MIN_OS=$(/usr/libexec/PlistBuddy -c 'Print :LSMinimumSystemVersion' "$PLIST")
verify_min_os() {
  local binary=$1
  local versions
  versions=$(xcrun vtool -show-build "$binary" | awk '$1 == "minos" { print $2 }')
  if [ -z "$versions" ] || ! awk -v expected="$MIN_OS" '$0 != expected { exit 1 }' <<< "$versions"; then
    echo "$binary 的最低系统版本为 ${versions:-未知}，包声明为 ${MIN_OS}。请用 make app 或 make dmg 重编。" >&2
    return 1
  fi
  echo "   $(basename "$binary") 最低支持 macOS $MIN_OS"
}
verify_min_os "$OUT_DIR/dubhed" >/dev/null

STAGE=$(mktemp -d "$OUT_DIR/.dubhe-app.XXXXXX")
trap 'rm -rf "$STAGE"' EXIT
STAGED_APP="$STAGE/$NAME.app"
CONTENTS="$STAGED_APP/Contents"
STAGED_IMAGE="$STAGE/$NAME.dmg"

# ---------------------------------------------------------------- 编译外壳
mkdir -p "$CONTENTS/MacOS" "$CONTENTS/Helpers" "$CONTENTS/Resources"

echo "==> 编译外壳"
swiftc -O -whole-module-optimization \
       -target "$(uname -m)-apple-macos$MIN_OS" \
       -o "$CONTENTS/MacOS/$NAME" \
       apps/macos/Sources/main.swift apps/macos/Sources/ReleaseUpdates.swift \
       apps/macos/Sources/TranslationPDF.swift apps/macos/Sources/TranslationExport.swift \
       apps/macos/Sources/OneShotScreenCapture.swift \
       -framework Cocoa -framework Carbon -framework WebKit -framework PDFKit -framework CoreText -framework ScreenCaptureKit

echo "==> 组装包内容"
cp "$PLIST" "$CONTENTS/Info.plist"
cp CHANGELOG.md "$CONTENTS/Resources/"
cp "$OUT_DIR/dubhed" "$CONTENTS/Helpers/dubhed"
rsync -a --exclude 'node_modules' --exclude '*.map' web/ "$CONTENTS/Resources/web/"
mkdir -p "$CONTENTS/Resources/backends"
cp tools/model_backend.py tools/mlx_lfm3.py tools/translation_docx.py "$CONTENTS/Resources/backends/"
if [[ -n "$LLAMA_RUNTIME" ]]; then
  if [[ ! -x "$LLAMA_RUNTIME/llama-server" || ! -f "$LLAMA_RUNTIME/LICENSE" ]]; then
    echo "DUBHE_LLAMA_RUNTIME_DIR 必须包含官方 llama-server、动态库和 LICENSE。" >&2
    exit 1
  fi
  TRANSLATION_HELPERS="$CONTENTS/Helpers/llama"
  mkdir -p "$TRANSLATION_HELPERS"
  rsync -a --include 'libllama-server-impl.dylib' --exclude '*-impl.dylib' \
        --include '*.dylib' --include '/llama-server' \
        --exclude '*' "$LLAMA_RUNTIME/" "$TRANSLATION_HELPERS/"
  cp "$LLAMA_RUNTIME/LICENSE" "$CONTENTS/Resources/llama.cpp-LICENSE.txt"
  "$TRANSLATION_HELPERS/llama-server" --version
fi
verify_min_os "$CONTENTS/MacOS/$NAME"
verify_min_os "$CONTENTS/Helpers/dubhed"

# ---------------------------------------------------------------- 图标
# 1024 的源图已经是 macOS 那套圆角 + 留边的样式，直接切各档尺寸即可。
# iconset 的档位是固定的（16/32/128/256/512 各一档 1x 一档 2x），少一档
# iconutil 会报错。
ICON_SRC=assets/branding/app-icon-1024.png
if [ -f "$ICON_SRC" ] && command -v iconutil >/dev/null 2>&1; then
  SET="$STAGE/Dubhe.iconset"
  mkdir -p "$SET"
  for s in 16 32 128 256 512; do
    sips -z $s $s "$ICON_SRC" --out "$SET/icon_${s}x${s}.png" >/dev/null 2>&1
    sips -z $((s*2)) $((s*2)) "$ICON_SRC" --out "$SET/icon_${s}x${s}@2x.png" >/dev/null 2>&1
  done
  iconutil -c icns "$SET" -o "$CONTENTS/Resources/AppIcon.icns" 2>/dev/null \
    && echo "   图标 $(du -h "$CONTENTS/Resources/AppIcon.icns" | cut -f1)" \
    || echo "   图标生成失败（不影响运行，会显示系统默认图标）"
else
  echo "   跳过图标（缺 $ICON_SRC 或 iconutil）"
fi

# PkgInfo 不是必需的，但有些老工具靠它认类型
printf 'APPL????' > "$CONTENTS/PkgInfo"

# 版本号从 dubhe.h 的宏里读，不去跑 `dubhed --version` —— 引擎没有这个参数，
# 未知参数会被忽略然后照常起服务，脚本就永远卡在那儿了。
VER=$(awk '$1 == "#define" && $2 == "DUBHE_VERSION_MAJOR" {a=$3}
           $1 == "#define" && $2 == "DUBHE_VERSION_MINOR" {b=$3}
           $1 == "#define" && $2 == "DUBHE_VERSION_PATCH" {c=$3}
           END {if (a == "" || b == "" || c == "") exit 1; print a"."b"."c}' \
      engine/include/dubhe/dubhe.h)
/usr/libexec/PlistBuddy -c "Set :CFBundleShortVersionString $VER" "$CONTENTS/Info.plist"
/usr/libexec/PlistBuddy -c "Set :CFBundleVersion $VER" "$CONTENTS/Info.plist"
echo "   版本 $VER"

# ---------------------------------------------------------------- 签名
if [[ "$SIGN_IDENTITY" == "Developer ID Application: "* ]]; then
  echo "==> Developer ID 签名"
  SIGN_FLAGS=(--options runtime --timestamp --sign "$SIGN_IDENTITY")
elif [ -n "$SIGN_IDENTITY" ]; then
  echo "==> Apple Development 签名"
  SIGN_FLAGS=(--timestamp=none --sign "$SIGN_IDENTITY")
else
  echo "==> 本地 ad-hoc 签名"
  SIGN_FLAGS=(--timestamp=none --sign -)
fi
codesign --force "${SIGN_FLAGS[@]}" "$CONTENTS/Helpers/dubhed"
if [[ -n "$LLAMA_RUNTIME" ]]; then
  for binary in "$TRANSLATION_HELPERS/"*.dylib "$TRANSLATION_HELPERS/llama-server"; do
    if [[ ! -L "$binary" ]]; then codesign --force "${SIGN_FLAGS[@]}" "$binary"; fi
  done
fi
codesign --force "${SIGN_FLAGS[@]}" "$STAGED_APP"
codesign --verify --strict "$CONTENTS/Helpers/dubhed"
codesign --verify --deep --strict "$STAGED_APP"

notarize() {
  local archive=$1
  local product=$2
  local result="$STAGE/notary-result.plist"
  echo "==> 提交 Apple 公证"
  xcrun notarytool submit "$archive" --keychain-profile "$NOTARY_PROFILE" --wait \
        --output-format plist > "$result"
  cat "$result"
  if [ "$(plutil -extract status raw -o - "$result")" != Accepted ]; then
    echo "Apple 公证未通过。" >&2
    return 1
  fi
  xcrun stapler staple "$product"
  xcrun stapler validate "$product"
}

# ---------------------------------------------------------------- DMG
if [ "$DMG" = "1" ]; then
  echo "==> 打包 DMG"
  VOLUME="$STAGE/volume"
  mkdir "$VOLUME"
  ditto "$STAGED_APP" "$VOLUME/$NAME.app"
  ln -s /Applications "$VOLUME/Applications"
  hdiutil create -quiet -volname "$NAME" -srcfolder "$VOLUME" -format UDZO "$STAGED_IMAGE"
  if [[ "$SIGN_IDENTITY" == "Developer ID Application: "* ]]; then
    codesign --force --timestamp --sign "$SIGN_IDENTITY" --identifier com.dubhe.desktop.dmg "$STAGED_IMAGE"
    codesign --verify --strict "$STAGED_IMAGE"
  fi
  if [ "$NOTARIZE" = 1 ]; then
    notarize "$STAGED_IMAGE" "$STAGED_IMAGE"
  fi
elif [ "$NOTARIZE" = 1 ]; then
  ZIP="$STAGE/$NAME.zip"
  ditto -c -k --keepParent "$STAGED_APP" "$ZIP"
  notarize "$ZIP" "$STAGED_APP"
fi

rm -rf "$APP"
mv "$STAGED_APP" "$APP"
echo "   $APP ($(du -sh "$APP" | cut -f1))"
if [ "$DMG" = 1 ]; then
  mv -f "$STAGED_IMAGE" "$IMAGE"
  echo "   $IMAGE ($(du -sh "$IMAGE" | cut -f1))"
fi

if [ "$RUN" = "1" ]; then
  echo "==> 启动"
  open "$APP"
fi
