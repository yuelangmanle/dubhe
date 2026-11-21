#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  printf '%s\n' '用法 bash tools/quality_android.sh 序列号 test [调试APK 测试APK]' \
    '     bash tools/quality_android.sh 序列号 upgrade 旧APK 新APK' >&2
  exit 2
fi

serial=$1
mode=$2
shift 2
root="$(cd "$(dirname "$0")/.." && pwd)"
sdk="${ANDROID_HOME:-${ANDROID_SDK_ROOT:-$HOME/Library/Android/sdk}}"
device=("${ADB:-$sdk/platform-tools/adb}" -s "$serial")
package=dev.dubhe.mobile

if [[ ! -x "${device[0]}" ]]; then
  device[0]="${ADB:-adb}"
fi
if [[ "$("${device[@]}" get-state)" != device ]]; then
  printf '设备未连接或尚未授权 ADB。\n' >&2
  exit 1
fi
if ! "${device[@]}" shell getprop ro.product.cpu.abilist | tr -d '\r' | grep -Eq '(^|,)arm64-v8a(,|$)'; then
  printf '需要能运行 arm64-v8a APK 的模拟器或手机。\n' >&2
  exit 1
fi

install() {
  if [[ ! -f "$1" ]]; then
    printf '找不到 APK %s\n' "$1" >&2
    exit 1
  fi
  "${device[@]}" install -r "$1"
}

version() {
  "${device[@]}" shell dumpsys package "$package" | awk -F= '/versionName=/ && !seen { print $2; seen=1 }' | tr -d '\r'
}

launch() {
  "${device[@]}" shell am force-stop "$package"
  local started
  started=$("${device[@]}" shell am start -W -n "$package/.MainActivity")
  printf '%s\n' "$started"
  grep -q '^Status: ok' <<<"$started"
}

case "$mode" in
  test)
    if [[ $# -ne 0 && $# -ne 2 ]]; then exit 2; fi
    install "${1:-$root/apps/android/app/build/outputs/apk/debug/app-debug.apk}"
    install "${2:-$root/apps/android/app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk}"
    launch
    result=$("${device[@]}" shell am instrument -w "$package.test/androidx.test.runner.AndroidJUnitRunner")
    printf '%s\n' "$result"
    grep -Eq '^OK \([1-9][0-9]* tests?\)' <<<"$result"
    ;;
  upgrade)
    if [[ $# -ne 2 ]]; then exit 2; fi
    [[ -f "$1" && -f "$2" ]]
    install "$1"
    launch
    before="$(version)"
    install "$2"
    launch
    after="$(version)"
    if [[ -z "$before" || -z "$after" || "$before" == "$after" ]]; then
      printf '未观察到版本变化。\n' >&2
      exit 1
    fi
    printf 'Android 覆盖安装 %s → %s\n' "$before" "$after"
    ;;
  *) printf '模式必须是 test 或 upgrade。\n' >&2; exit 2 ;;
esac
