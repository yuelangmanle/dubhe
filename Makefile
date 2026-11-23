# Dubhe 构建（零外部依赖，只用系统 clang）
#
# 关键：-MMD -MP 生成头文件依赖。改 internal.h 会改结构体布局，
# 任何一个 .o 漏重编都会造成 ABI 错位（表现为诡异段错误）。
CXX      ?= clang++
CXXSTD   := -std=c++20
OPT      ?= -O3
WARN     := -Wall -Wextra -Wno-unused-parameter
DEFS     := -DNDEBUG
INCS     := -Iengine/include -Iengine/src
CFLAGS   := $(OPT) $(WARN) $(DEFS) $(INCS) -MMD -MP
CXXFLAGS := $(CXXSTD) $(CFLAGS)
BUILD_INPUTS := Makefile

UNAME_S  := $(shell uname -s)

# PLATFORM 决定选哪套平台源文件和链接参数。默认跟随本机，交叉编译时显式指定：
#   make PLATFORM=android ANDROID_ABI=arm64-v8a
PLATFORM ?= $(UNAME_S)

ENGINE_SRCS := engine/src/gguf.cpp engine/src/quant.cpp engine/src/quant_q8.cpp \
               engine/src/tokenizer.cpp engine/src/model.cpp engine/src/engine.cpp \
               engine/src/mmproj.cpp engine/src/mel.cpp engine/src/audiomm.cpp

# Metal 后端只在 Apple 平台编译。着色器在运行时编译（newLibraryWithSource:），
# 所以不需要离线 Metal Toolchain。
ifeq ($(PLATFORM),Darwin)
  MACOS_PLIST := apps/macos/Resources/Info.plist
  MACOS_MIN := $(shell /usr/libexec/PlistBuddy -c 'Print :LSMinimumSystemVersion' $(MACOS_PLIST))
  MACOS_FLAGS := -mmacosx-version-min=$(MACOS_MIN)
  BUILD_INPUTS += $(MACOS_PLIST)
  ENGINE_SRCS += engine/src/metal.mm engine/src/image.mm engine/src/audiofile.mm \
                 engine/src/vit_gpu.mm
  CFLAGS      += $(MACOS_FLAGS)
  CXXFLAGS    += $(MACOS_FLAGS)
  OBJCXXFLAGS := $(CXXFLAGS) -fobjc-arc
  LDFLAGS     := $(MACOS_FLAGS) -framework Metal -framework Foundation -framework ImageIO \
                 -framework CoreGraphics -framework CoreFoundation \
                 -framework AVFoundation -framework CoreMedia
else ifeq ($(PLATFORM),android)
  # ---- NDK 交叉编译
  # 目标：先把纯文本 CPU 路径跑起来。Metal 换成 stub，图像/音频解码也还是
  # stub（Android 侧要接 ImageDecoder / MediaCodec，那是客户端那一轮的事）。
  ANDROID_ABI ?= arm64-v8a
  ANDROID_API ?= 26
  ANDROID_NDK ?= $(firstword $(ANDROID_NDK_HOME) $(ANDROID_NDK_ROOT) \
                             $(wildcard $(HOME)/Library/Android/sdk/ndk/*) \
                             $(wildcard $(HOME)/Android/Sdk/ndk/*) \
                             $(if $(ANDROID_HOME),$(wildcard $(ANDROID_HOME)/ndk/*)) \
                             $(if $(ANDROID_SDK_ROOT),$(wildcard $(ANDROID_SDK_ROOT)/ndk/*)))
  ifeq ($(ANDROID_NDK),)
    $(error 找不到 NDK。装一个：sdkmanager "ndk;27.2.12479018"，或指定 ANDROID_NDK=<path>)
  endif
  NDK_HOST := $(shell uname -s | tr 'A-Z' 'a-z')-x86_64
  NDK_BIN  := $(ANDROID_NDK)/toolchains/llvm/prebuilt/$(NDK_HOST)/bin

  ifeq ($(ANDROID_ABI),arm64-v8a)
    ANDROID_TARGET := aarch64-linux-android$(ANDROID_API)
    ANDROID_ARCH_FLAGS :=
  else ifeq ($(ANDROID_ABI),armeabi-v7a)
    ANDROID_TARGET := armv7a-linux-androideabi$(ANDROID_API)
    ANDROID_ARCH_FLAGS := -mfpu=neon -mfloat-abi=softfp
  else ifeq ($(ANDROID_ABI),x86_64)
    ANDROID_TARGET := x86_64-linux-android$(ANDROID_API)
    ANDROID_ARCH_FLAGS :=
  else
    $(error 不支持的 ANDROID_ABI: $(ANDROID_ABI))
  endif

  CXX := $(NDK_BIN)/clang++
  CC  := $(NDK_BIN)/clang
  AR  := $(NDK_BIN)/llvm-ar
  ANDROID_FLAGS := --target=$(ANDROID_TARGET) $(ANDROID_ARCH_FLAGS) -fPIC -fno-omit-frame-pointer
  CFLAGS   += $(ANDROID_FLAGS)
  CXXFLAGS += $(ANDROID_FLAGS)
  # 静态链 libc++：Android 系统里没有 libc++.so，动态链会装不上
  LDFLAGS  := --target=$(ANDROID_TARGET) -static-libstdc++ -Wl,--gc-sections -lm
  ENGINE_SRCS += engine/src/metal_stub.cpp engine/src/image_stub.cpp engine/src/audiofile_stub.cpp \
                 engine/src/vit_gpu_stub.cpp
  BUILD := build-android-$(ANDROID_ABI)
  ifeq ($(ENABLE_VULKAN),1)
    BUILD := build-android-$(ANDROID_ABI)-vulkan
    GLSLC ?= $(ANDROID_NDK)/shader-tools/$(NDK_HOST)/glslc
    ENGINE_SRCS += engine/src/vulkan_matvec.cpp
    CXXFLAGS += -DDUBHE_VULKAN=1 -I$(BUILD)/generated
    LDFLAGS  += -lvulkan
    VULKAN_TEST := $(BUILD)/test_vulkan
  endif
else
  ENGINE_SRCS += engine/src/metal_stub.cpp engine/src/image_stub.cpp engine/src/audiofile_stub.cpp \
                 engine/src/vit_gpu_stub.cpp
  CFLAGS   += -pthread
  CXXFLAGS += -pthread
  LDFLAGS   := -pthread
endif

# BUILD 必须在平台分支之后再展开：交叉编译时 BUILD 会被改成 build-android-<abi>，
# 如果 BIN 在上面就展开，Android 产物会直接盖掉本机的 build/dubhe（踩过）。
BUILD ?= build
BIN   := $(BUILD)/dubhe
SRV   := $(BUILD)/dubhed
LIB   := $(BUILD)/libdubhe.a

ENGINE_OBJS := $(patsubst %.cpp,$(BUILD)/%.o,$(patsubst %.mm,$(BUILD)/%.o,$(ENGINE_SRCS)))
CLI_OBJ     := $(BUILD)/cli/main.o
SRV_OBJ     := $(BUILD)/server/dubhed.o
PORTABLE_TOOLS := $(addprefix $(BUILD)/,selftest ggufdump tokprobe vlprobe audprobe batchcheck)
ifeq ($(PLATFORM),Darwin)
  METAL_TOOLS := $(addprefix $(BUILD)/,optest mma_probe mm2_check mvbench)
endif

MODEL ?= $(HOME)/.dubhe/models/qwen2.5-0.5b-instruct/qwen2.5-0.5b-instruct-q4_k_m.gguf

.PHONY: all clean library tools run inspect bench serve check test test-metal integration-test \
        gpu optest shader-check mm2 engine batchcheck membw mvbench app app-run dmg

all: $(BIN) $(SRV) tools

# 别名。注意：仓库里有个同名目录 engine/，所以 make 会把 "engine" 当成一个
# 已存在、无先决条件的文件目标而直接报 "Nothing to be done"——它**不会**重编
# 任何东西。这个坑让一次参数 A/B 悄悄比较了两个完全相同的二进制。别名显式
# 依赖 $(BIN) 之后，"make engine" 才真的会重编。
engine: $(BIN)

tools: $(PORTABLE_TOOLS) $(METAL_TOOLS)

library: $(LIB)

$(LIB): $(ENGINE_OBJS)
	@rm -f "$@.tmp"
	$(AR) rcs "$@.tmp" $^
	@mv "$@.tmp" "$@"
	@echo "✓ $@"

# 分词 + 贪心生成探针：同一模型不同量化档输出不一致时，用它区分
# "分词差异" 和 "前向差异"（对比打印出的 token id 序列即可）。
$(BUILD)/tokprobe: $(BUILD)/tools/tokprobe.o $(LIB)
	$(CXX) $^ -o $@ $(LDFLAGS)
	@echo "✓ $@"

$(BIN): $(CLI_OBJ) $(LIB)
	$(CXX) $^ -o $@ $(LDFLAGS)
	@echo "✓ $@"

$(SRV): $(SRV_OBJ) $(LIB)
	$(CXX) $^ -o $@ $(LDFLAGS)
	@echo "✓ $@"

$(BUILD)/selftest: $(BUILD)/tools/selftest.o $(LIB)
	$(CXX) $^ -o $@ $(LDFLAGS)
	@echo "✓ $@"

$(BUILD)/ggufdump: $(BUILD)/engine/src/gguf.o $(BUILD)/engine/src/quant.o $(BUILD)/tools/ggufdump.o
	$(CXX) $^ -o $@ $(LDFLAGS)
	@echo "✓ $@"

$(BUILD)/%.o: %.cpp $(BUILD_INPUTS)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/%.o: %.mm $(BUILD_INPUTS)
	@mkdir -p $(dir $@)
	$(CXX) $(OBJCXXFLAGS) -c $< -o $@

$(BUILD)/%.o: %.c $(BUILD_INPUTS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD)

-include $(ENGINE_OBJS:.o=.d) $(CLI_OBJ:.o=.d) $(SRV_OBJ:.o=.d) \
         $(wildcard $(BUILD)/tools/*.d) $(wildcard $(BUILD)/tests/*.d)

inspect: $(BIN)
	"$(BIN)" inspect -m "$(MODEL)"

run: $(BIN)
	"$(BIN)" run -m "$(MODEL)" -p "你好，介绍一下你自己" -n 128 --stats

bench: $(BIN)
	"$(BIN)" bench -m "$(MODEL)" -n 64

serve: $(SRV)
	"$(SRV)" -m "$(MODEL)" --port 7777

$(BUILD)/vlprobe: $(BUILD)/tools/vlprobe.o $(LIB)
	$(CXX) $^ -o $@ $(LDFLAGS)
	@echo "✓ $@"

# 多序列批处理对拍（连续批处理的正确性 + 吞吐）
$(BUILD)/batchcheck: $(BUILD)/tools/batchcheck.o $(LIB)
	$(CXX) $^ -o $@ $(LDFLAGS)
	@echo "✓ $@"

batchcheck: $(BUILD)/batchcheck
	"$(BUILD)/batchcheck" "$(MODEL)" 3 8

$(BUILD)/audprobe: $(BUILD)/tools/audprobe.o $(LIB)
	$(CXX) $^ -o $@ $(LDFLAGS)
	@echo "✓ $@"

$(BUILD)/test_quant: $(BUILD)/tests/test_quant.o $(LIB)
	$(CXX) $^ -o $@ $(LDFLAGS)
	@echo "✓ $@"

$(BUILD)/test_abi: $(BUILD)/tests/test_abi.o $(LIB)
	$(CXX) $^ -o $@ $(LDFLAGS)
	@echo "✓ $@"

ifneq ($(VULKAN_TEST),)
$(BUILD)/generated/vulkan_q4_k.inc: engine/src/vulkan_q4_k.comp Makefile
	@mkdir -p $(dir $@)
	"$(GLSLC)" --target-env=vulkan1.0 -O -mfmt=num "$<" -o "$@.tmp"
	@mv "$@.tmp" "$@"

$(BUILD)/engine/src/vulkan_matvec.o: $(BUILD)/generated/vulkan_q4_k.inc

$(VULKAN_TEST): $(BUILD)/tests/test_vulkan.o $(LIB)
	$(CXX) $^ -o $@ $(LDFLAGS)
	@echo "✓ $@"
endif

ifeq ($(PLATFORM),android)
test: $(BUILD)/test_quant $(BUILD)/test_abi $(VULKAN_TEST)
	@echo "Android 测试已编译，可在对应 ABI 的设备或模拟器运行。"

check integration-test:
	@echo "$@ 需要可在本机运行的构建。" >&2
	@exit 1
else
$(BUILD)/test_store: $(BUILD)/tests/test_store.o
	$(CXX) $^ -o $@ $(LDFLAGS)

test: $(BUILD)/test_quant $(BUILD)/test_abi $(BUILD)/test_store
	"$(BUILD)/test_quant"
	"$(BUILD)/test_abi"
	"$(BUILD)/test_store"

integration-test: $(SRV)
	python3 tools/gen_model_catalog.py --check
	python3 tests/test_server.py "$(SRV)"
	python3 tests/test_model_backend.py
	python3 tests/test_translation_docx.py
	node tests/test_client.mjs
ifeq ($(PLATFORM),Darwin)
	swiftc apps/macos/Sources/TranslationPDF.swift apps/macos/Sources/TranslationExport.swift tests/test_translation_pdf.swift -o "$(BUILD)/test_translation_pdf" -framework PDFKit -framework CoreText
	"$(BUILD)/test_translation_pdf"
	swiftc apps/macos/Sources/ScanPDF.swift apps/macos/Sources/ScanSearchablePDF.swift apps/macos/Sources/ScanOCR.swift tests/test_scan_pdf.swift -o "$(BUILD)/test_scan_pdf" -framework PDFKit -framework CoreText
	"$(BUILD)/test_scan_pdf"
	swiftc apps/macos/Sources/ReleaseUpdates.swift tests/test_release_updates.swift -o "$(BUILD)/test_release_updates"
	"$(BUILD)/test_release_updates"
endif

# 一条命令自检（ADR-0007）：逐条运行并校验所用后端。
check: $(BUILD)/selftest
ifeq ($(PLATFORM),Darwin)
	"$(BUILD)/selftest" --backend metal "$(MODEL)" "你好，请用一句话介绍你自己。"
endif
	"$(BUILD)/selftest" --backend cpu "$(MODEL)" "你好，请用一句话介绍你自己。"
endif

ifeq ($(PLATFORM),Darwin)
app: $(SRV)
	BUILD_DIR="$(BUILD)" ./apps/macos/build.sh $(MACOS_APP_ARGS)

app-run: $(SRV)
	BUILD_DIR="$(BUILD)" ./apps/macos/build.sh --run $(MACOS_APP_ARGS)

dmg: $(SRV)
	BUILD_DIR="$(BUILD)" ./apps/macos/build.sh --dmg $(MACOS_APP_ARGS)

# 逐个算子对拍：只在 Apple 上编译 Metal 工具。
$(BUILD)/optest: $(BUILD)/tools/gpu_optest.o
	$(CXX) $^ -o $@ $(LDFLAGS)
	@echo "✓ $@"

$(BUILD)/mma_probe: $(BUILD)/tools/mma_probe.o
	$(CXX) $^ -o $@ $(LDFLAGS)
	@echo "✓ $@"

$(BUILD)/mm2_check: $(BUILD)/tools/mm2_check.o
	$(CXX) $^ -o $@ $(LDFLAGS)
	@echo "✓ $@"

# 着色器是 C++ 字符串里的源码，编译器看不到它 —— 只有运行时才报错，
# 表现为"静默回退到 CPU"（实测踩过一次：prefill 数字全失真）。
# 这个目标把字符串抠出来用离线 Metal 工具链编一遍，几秒钟就能抓到。
shader-check:
	@python3 -c "import re,sys; s=open('engine/src/metal_kernels.h').read(); \
	  m=re.search(r'R\"MSL\\((.*?)\\)MSL\"', s, re.S); \
	  sys.exit('找不到 MSL 字符串') if not m else open('/tmp/dubhe_k.metal','w').write(m.group(1))"
	@xcrun -sdk macosx metal -c /tmp/dubhe_k.metal -o /tmp/dubhe_k.air
	@echo "✓ 着色器编译通过"

optest: $(BUILD)/optest
	"$(BUILD)/optest"

# CI 的虚拟机可能没有 Metal 设备。仅在显式允许、且探针确认无设备时跳过。
test: test-metal
test-metal: $(BUILD)/optest $(BUILD)/mm2_check
	@status=0; "$(BUILD)/optest" || status=$$?; \
	  if [ "$$status" = 77 ] && [ "$(TEST_ALLOW_NO_METAL)" = 1 ]; then \
	    echo "没有 Metal 设备，跳过 GPU 测试。"; exit 0; \
	  fi; \
	  [ "$$status" = 0 ] || exit "$$status"; \
	  "$(BUILD)/mm2_check"

# 分块 matmul 与旧内核逐元素对拍 + 裸吞吐基准
mm2: $(BUILD)/mm2_check
	"$(BUILD)/mm2_check"
	"$(BUILD)/mm2_check" --bench

# 只读带宽天花板（按访问形状分）：解码优化前先量清楚天花板在哪
$(BUILD)/membw: $(BUILD)/tools/membw.o
	$(CXX) $^ -o $@ $(LDFLAGS)
	@echo "✓ $@"

# 单个 mv_* 内核的吞吐 + 对拍。秒级出数，不用加载整个模型。
$(BUILD)/mvbench: $(BUILD)/tools/mvbench.o $(BUILD)/engine/src/quant.o $(BUILD)/engine/src/gguf.o
	$(CXX) $^ -o $@ $(LDFLAGS)
	@echo "✓ $@"

membw: $(BUILD)/membw
	"$(BUILD)/membw" 1073741824 15

mvbench: $(BUILD)/mvbench
	"$(BUILD)/mvbench" q4_k 8192 2048 10
	"$(BUILD)/mvbench" q4_k 2048 8192 10

# GPU / CPU 对比
gpu: $(BIN)
	@echo "── CPU ──"
	"$(BIN)" bench -m "$(MODEL)" -n 64 --no-gpu
	@echo "── GPU ──"
	"$(BIN)" bench -m "$(MODEL)" -n 64
else
app app-run dmg gpu optest shader-check mm2 test-metal membw mvbench:
	@echo "$@ 需要 macOS / Metal。" >&2
	@exit 1
endif
