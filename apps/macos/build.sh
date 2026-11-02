#!/bin/bash
# 把 Dubhe 打成 macOS App 包（.app），不依赖 Xcode 工程。
#
#   ./apps/macos/build.sh               # 出 build/Dubhe.app
#   ./apps/macos/build.sh --run         # 出完直接打开
#   ./apps/macos/build.sh --dmg         # 再打一个 build/Dubhe.dmg
#
# 产物结构：
#   build/Dubhe.app/Contents/MacOS/Dubhe            外壳（swiftc 编出来，~200 KB）
#   build/Dubhe.app/Contents/Resources/dubhed       引擎
#   build/Dubhe.app/Contents/Resources/web/         Web UI
#
# 引擎和 web/ 都塞进包里，是为了「拷到别的机器双击就能开」——外壳里
# resolveEngine() 先找 Bundle.main.resourcePath，所以包里那份一定优先。
set -euo pipefail
cd "$(dirname "$0")/../.."

APP=build/Dubhe.app
NAME=Dubhe
RUN=0
DMG=0
for a in "$@"; do
  case "$a" in
    --run) RUN=1 ;;
    --dmg) DMG=1 ;;
    *) echo "未知参数: $a"; exit 2 ;;
  esac
done

# ---------------------------------------------------------------- 前置检查
if [ ! -x build/dubhed ]; then
  echo "缺 build/dubhed —— 先 make all" >&2
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

# ---------------------------------------------------------------- 编译外壳
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

echo "==> 编译外壳"
swiftc -O -whole-module-optimization \
       -target "$(uname -m)-apple-macos14.0" \
       -o "$APP/Contents/MacOS/$NAME" \
       apps/macos/Sources/main.swift \
       -framework Cocoa -framework WebKit

echo "==> 组装包内容"
cp apps/macos/Resources/Info.plist "$APP/Contents/Info.plist"
cp build/dubhed "$APP/Contents/Resources/dubhed"
# web/ 只有 1700 行前端，整份拷进去；--exclude 掉以后可能出现的 node_modules
rsync -a --exclude 'node_modules' --exclude '*.map' web/ "$APP/Contents/Resources/web/"

# ---------------------------------------------------------------- 图标
# 1024 的源图已经是 macOS 那套圆角 + 留边的样式，直接切各档尺寸即可。
# iconset 的档位是固定的（16/32/128/256/512 各一档 1x 一档 2x），少一档
# iconutil 会报错。
ICON_SRC=assets/branding/app-icon-1024.png
if [ -f "$ICON_SRC" ] && command -v iconutil >/dev/null 2>&1; then
  SET=$(mktemp -d)/Dubhe.iconset
  mkdir -p "$SET"
  for s in 16 32 128 256 512; do
    sips -z $s $s "$ICON_SRC" --out "$SET/icon_${s}x${s}.png" >/dev/null 2>&1
    sips -z $((s*2)) $((s*2)) "$ICON_SRC" --out "$SET/icon_${s}x${s}@2x.png" >/dev/null 2>&1
  done
  iconutil -c icns "$SET" -o "$APP/Contents/Resources/AppIcon.icns" 2>/dev/null \
    && echo "   图标 $(du -h "$APP/Contents/Resources/AppIcon.icns" | cut -f1)" \
    || echo "   图标生成失败（不影响运行，会显示系统默认图标）"
  rm -rf "$(dirname "$SET")"
else
  echo "   跳过图标（缺 $ICON_SRC 或 iconutil）"
fi

# PkgInfo 不是必需的，但有些老工具靠它认类型
printf 'APPL????' > "$APP/Contents/PkgInfo"

# 版本号从 dubhe.h 的宏里读，不去跑 `dubhed --version` —— 引擎没有这个参数，
# 未知参数会被忽略然后照常起服务，脚本就永远卡在那儿了。
VER=$(awk '/DUBHE_VERSION_MAJOR/{a=$3} /DUBHE_VERSION_MINOR/{b=$3} /DUBHE_VERSION_PATCH/{c=$3} END{print a"."b"."c}' \
      engine/include/dubhe/dubhe.h 2>/dev/null || true)
if [ -n "${VER:-}" ] && [ "$VER" != ".." ]; then
  /usr/libexec/PlistBuddy -c "Set :CFBundleShortVersionString $VER" "$APP/Contents/Info.plist" 2>/dev/null || true
  /usr/libexec/PlistBuddy -c "Set :CFBundleVersion $VER" "$APP/Contents/Info.plist" 2>/dev/null || true
  echo "   版本 $VER"
fi

# ---------------------------------------------------------------- 签名
# 本地自用用 ad-hoc（-）即可：能过 Gatekeeper 的"未识别开发者"提示要求
# 用户右键打开，但本机跑没有任何问题。要分发给别人就换成：
#   codesign --force --deep --options runtime --sign "Developer ID Application: XXX" "$APP"
echo "==> 签名（ad-hoc）"
codesign --force --deep --sign - "$APP" >/dev/null 2>&1 || \
  echo "   签名失败（不影响本机运行）"

# ---------------------------------------------------------------- 校验
echo "==> 校验"
codesign --verify --deep --strict "$APP" 2>/dev/null && echo "   签名 OK" || echo "   签名未通过（本机仍可运行）"
SIZE=$(du -sh "$APP" | cut -f1)
echo "   $APP  ($SIZE)"
ls -1 "$APP/Contents/Resources"

# ---------------------------------------------------------------- DMG
if [ "$DMG" = "1" ]; then
  echo "==> 打包 DMG"
  rm -f build/Dubhe.dmg
  STAGE=$(mktemp -d)
  cp -R "$APP" "$STAGE/"
  ln -s /Applications "$STAGE/Applications"
  hdiutil create -quiet -volname "Dubhe" -srcfolder "$STAGE" -ov -format UDZO build/Dubhe.dmg
  rm -rf "$STAGE"
  echo "   build/Dubhe.dmg ($(du -sh build/Dubhe.dmg | cut -f1))"
fi

if [ "$RUN" = "1" ]; then
  echo "==> 启动"
  open "$APP"
fi
