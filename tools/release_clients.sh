#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

mode=${1:-build}
if [[ "$mode" != build && "$mode" != publish ]]; then
  echo "用法：$0 [build|publish]" >&2
  exit 2
fi
repo=yuelangmanle/dubhe
version=$(awk '$2 == "DUBHE_VERSION_MAJOR" {major=$3}
               $2 == "DUBHE_VERSION_MINOR" {minor=$3}
               $2 == "DUBHE_VERSION_PATCH" {patch=$3}
               END {if (major=="" || minor=="" || patch=="") exit 1; print major "." minor "." patch}' engine/include/dubhe/dubhe.h)
tag="v$version"

if [[ "$mode" == publish ]]; then
  if [[ -n "$(git status --porcelain)" || "$(git branch --show-current)" != main ]]; then
    echo "请先将本次版本和更新日志提交到干净的 main 分支。" >&2
    exit 2
  fi
  case "$(git remote get-url --push origin)" in
    "https://github.com/$repo.git"|"git@github.com:$repo.git") ;;
    *) echo "origin 的推送地址应为 $repo。" >&2; exit 2 ;;
  esac
fi

output="$PWD/build/releases/$tag"
mkdir -p "$output"
notes="$output/CHANGELOG-section.md"
awk -v header="## [$version]" '
  /^## \[/ { if (found || seen++) exit; found = index($0, header) == 1; next }
  found { print; if ($0 ~ /^- /) entries++ }
  END { if (!found || !entries) exit 1 }
' CHANGELOG.md > "$notes" || {
  echo "请把 $version 的更新日志放在 CHANGELOG.md 的第一节，并写明本次变化。" >&2
  exit 2
}

support="$HOME/Library/Application Support/Dubhe"
export DUBHE_LLAMA_RUNTIME_DIR=${DUBHE_LLAMA_RUNTIME_DIR:-"$support/BuildRuntime/llama-macos-arm64"}
signing=${DUBHE_ANDROID_SIGNING_DIR:-"$support/AndroidSigning"}
export ANDROID_HOME=${ANDROID_HOME:-"$HOME/Library/Android/sdk"}
export JAVA_HOME=${JAVA_HOME:-"/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home"}
unset ANDROID_SDK_ROOT
android_tools="$ANDROID_HOME/build-tools/36.0.0"
android_certificate=a6f43cd30e0f4f9ee7fe981f315e54485bc4124cbc57fb249f66a3caa7b28bfb
if [[ ! -x "$DUBHE_LLAMA_RUNTIME_DIR/llama-server" || ! -x "$android_tools/apksigner" ||
      ! -f "$signing/android-release.p12" || ! -f "$signing/android-release.pass" ]]; then
  echo "发布需要现有的 macOS 翻译运行时、Android SDK 36 和 Dubhe Android 发布签名。" >&2
  exit 2
fi

mac_name="Dubhe-$version-macOS-arm64.dmg"
android_name="Dubhe-$version-Android-arm64.apk"
mac="$output/$mac_name"
android="$output/$android_name"
make -j 4 BUILD=build/verify-release dmg
hdiutil verify -quiet build/verify-release/Dubhe.dmg
/usr/bin/codesign --verify --deep --strict build/verify-release/Dubhe.app
cp build/verify-release/Dubhe.dmg "$mac"
./apps/android/gradlew -p apps/android :app:assembleRelease
"$android_tools/apksigner" sign --ks "$signing/android-release.p12" --ks-key-alias dubhe-release \
  --ks-pass "file:$signing/android-release.pass" --v4-signing-enabled false --out "$android" \
  apps/android/app/build/outputs/apk/release/app-release-unsigned.apk
certificate=$("$android_tools/apksigner" verify --print-certs "$android" |
  awk '/^Signer #1 certificate SHA-256 digest:/ { print $NF }')
if [[ "$certificate" != "$android_certificate" ]]; then
  echo "Android APK 的证书与 Dubhe 发布签名不匹配，无法保证覆盖升级。" >&2
  exit 2
fi
"$android_tools/zipalign" -c -P 16 4 "$android"
badging=$("$android_tools/aapt" dump badging "$android")
[[ "$badging" == *"versionName='$version'"* && "$badging" == *"native-code: 'arm64-v8a'"* ]]
cmp CHANGELOG.md <(unzip -p "$android" assets/CHANGELOG.md)
cmp CHANGELOG.md build/verify-release/Dubhe.app/Contents/Resources/CHANGELOG.md
printf '%s\n' "$mac" "$android"

if [[ "$mode" == publish ]]; then
  if git show-ref --verify --quiet "refs/tags/$tag"; then
    [[ "$(git rev-parse "$tag^{commit}")" == "$(git rev-parse HEAD)" ]]
  else
    git tag -a "$tag" -m "Dubhe $version"
  fi
  git push --atomic origin HEAD:refs/heads/main "refs/tags/$tag"
  if draft=$(gh release view "$tag" --repo "$repo" --json isDraft --jq .isDraft 2>/dev/null); then
    if [[ "$draft" != true ]]; then
      echo "$tag 已发布，请提升版本号后再发布。" >&2
      exit 2
    fi
  else
    gh release create "$tag" --repo "$repo" --verify-tag --draft --title "Dubhe $version" --notes-file "$notes"
  fi
  gh release upload "$tag" --repo "$repo" --clobber "$mac" "$android"
  assets=$(gh release view "$tag" --repo "$repo" --json assets --jq '.assets[].name')
  if [[ "$assets" != *"$mac_name"* || "$assets" != *"$android_name"* ]]; then
    echo "GitHub 草稿中缺少安装包，请重试。" >&2
    exit 1
  fi
  gh release edit "$tag" --repo "$repo" --draft=false --latest
fi
