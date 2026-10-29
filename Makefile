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
CXXFLAGS := $(CXXSTD) $(OPT) $(WARN) $(DEFS) $(INCS) -MMD -MP

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
  ENGINE_SRCS += engine/src/metal.mm engine/src/image.mm engine/src/audiofile.mm
  OBJCXXFLAGS := $(CXXFLAGS) -fobjc-arc
  LDFLAGS     := -framework Metal -framework Foundation -framework ImageIO \
                 -framework CoreGraphics -framework CoreFoundation \
                 -framework AVFoundation -framework CoreMedia
else ifeq ($(PLATFORM),android)
  # ---- NDK 交叉编译
  # 目标：先把纯文本 CPU 路径跑起来。Metal 换成 stub，图像/音频解码也还是
  # stub（Android 侧要接 ImageDecoder / MediaCodec，那是客户端那一轮的事）。
  ANDROID_ABI ?= arm64-v8a
  ANDROID_API ?= 26
  ANDROID_NDK ?= $(firstword $(wildcard $(HOME)/Library/Android/sdk/ndk/*) \
                             $(wildcard $(HOME)/Android/Sdk/ndk/*))
  ifeq ($(ANDROID_NDK),)
    $(error 找不到 NDK。装一个：sdkmanager "ndk;27.2.12479018"，或指定 ANDROID_NDK=<path>)
  endif
  NDK_BIN := $(ANDROID_NDK)/toolchains/llvm/prebuilt/$(shell uname -s | tr 'A-Z' 'a-z')-x86_64/bin

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
  CXXFLAGS += --target=$(ANDROID_TARGET) $(ANDROID_ARCH_FLAGS) -fPIC -fno-omit-frame-pointer
  # 静态链 libc++：Android 系统里没有 libc++.so，动态链会装不上
  LDFLAGS  := --target=$(ANDROID_TARGET) -static-libstdc++ -Wl,--gc-sections -lm
  ENGINE_SRCS += engine/src/metal_stub.cpp engine/src/image_stub.cpp engine/src/audiofile_stub.cpp
  BUILD := build-android-$(ANDROID_ABI)
else
  ENGINE_SRCS += engine/src/metal_stub.cpp engine/src/image_stub.cpp engine/src/audiofile_stub.cpp
  LDFLAGS     :=
endif

# BUILD 必须在平台分支之后再展开：交叉编译时 BUILD 会被改成 build-android-<abi>，
# 如果 BIN 在上面就展开，Android 产物会直接盖掉本机的 build/dubhe（踩过）。
BUILD ?= build
BIN   := $(BUILD)/dubhe
SRV   := $(BUILD)/dubhed

ENGINE_OBJS := $(patsubst %.cpp,$(BUILD)/%.o,$(patsubst %.mm,$(BUILD)/%.o,$(ENGINE_SRCS)))
CLI_OBJ     := $(BUILD)/cli/main.o
SRV_OBJ     := $(BUILD)/server/dubhed.o
TOOL_OBJS   := $(BUILD)/tools/selftest.o $(BUILD)/tools/ggufdump.o

MODEL ?= $(HOME)/.dubhe/models/qwen2.5-0.5b-instruct/qwen2.5-0.5b-instruct-q4_k_m.gguf

.PHONY: all clean tools run inspect bench serve check gpu optest shader-check mm2 engine

all: $(BIN) $(SRV) tools

# 别名。注意：仓库里有个同名目录 engine/，所以 make 会把 "engine" 当成一个
# 已存在、无先决条件的文件目标而直接报 "Nothing to be done"——它**不会**重编
# 任何东西。这个坑让一次参数 A/B 悄悄比较了两个完全相同的二进制。别名显式
# 依赖 $(BIN) 之后，"make engine" 才真的会重编。
engine: $(BIN)

tools: $(BUILD)/selftest $(BUILD)/ggufdump $(BUILD)/tokprobe \
       $(BUILD)/mma_probe $(BUILD)/mm2_check $(BUILD)/vlprobe $(BUILD)/audprobe

# 分词 + 贪心生成探针：同一模型不同量化档输出不一致时，用它区分
# "分词差异" 和 "前向差异"（对比打印出的 token id 序列即可）。
$(BUILD)/tokprobe: $(ENGINE_OBJS) $(BUILD)/tools/tokprobe.o
	$(CXX) $^ -o $@ $(LDFLAGS)
	@echo "✓ $@"

$(BIN): $(ENGINE_OBJS) $(CLI_OBJ)
	@mkdir -p $(dir $@)
	$(CXX) $(ENGINE_OBJS) $(CLI_OBJ) -o $@ $(LDFLAGS)
	@echo "✓ $@"

$(SRV): $(ENGINE_OBJS) $(SRV_OBJ)
	@mkdir -p $(dir $@)
	$(CXX) $(ENGINE_OBJS) $(SRV_OBJ) -o $@ $(LDFLAGS)
	@echo "✓ $@"

$(BUILD)/selftest: $(ENGINE_OBJS) $(BUILD)/tools/selftest.o
	$(CXX) $(ENGINE_OBJS) $(BUILD)/tools/selftest.o -o $@ $(LDFLAGS)
	@echo "✓ $@"

$(BUILD)/ggufdump: $(BUILD)/engine/src/gguf.o $(BUILD)/engine/src/quant.o $(BUILD)/tools/ggufdump.o
	$(CXX) $^ -o $@ $(LDFLAGS)
	@echo "✓ $@"

# 融合算子逐个对拍：整个前向是 400 多个 dispatch 编进一个 command buffer，
# 错一个的症状只有"输出乱码"，从外面看不出是哪个。这个工具几秒定位。
$(BUILD)/optest: tools/gpu_optest.mm engine/src/metal_kernels.h
	@mkdir -p $(dir $@)
	$(CXX) $(OBJCXXFLAGS) $< -o $@ $(LDFLAGS)
	@echo "✓ $@"

$(BUILD)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/%.o: %.mm
	@mkdir -p $(dir $@)
	$(CXX) $(OBJCXXFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD)

-include $(shell find $(BUILD) -name '*.d' 2>/dev/null)

inspect: $(BIN)
	./$(BIN) inspect -m $(MODEL)

run: $(BIN)
	./$(BIN) run -m $(MODEL) -p "你好，介绍一下你自己" -n 128 --stats

bench: $(BIN)
	./$(BIN) bench -m $(MODEL) -n 64

serve: $(SRV)
	./$(SRV) -m $(MODEL) --port 7777

# 一条命令自检（ADR-0007 要求）：GPU 与 CPU 两条路径都跑一遍
check: all
	./$(BUILD)/selftest $(MODEL) "你好，请用一句话介绍你自己。"
	@echo "── CPU 路径 ──"
	@DUBHE_NO_GPU=1 ./$(BUILD)/selftest $(MODEL) "你好，请用一句话介绍你自己。" 2>&1 | tail -3

# 逐个算子对拍（仅 Apple）
$(BUILD)/mma_probe: tools/mma_probe.mm
	@mkdir -p $(dir $@)
	$(CXX) $(OBJCXXFLAGS) $< -o $@ $(LDFLAGS)
	@echo "✓ $@"

$(BUILD)/vlprobe: $(ENGINE_OBJS) $(BUILD)/tools/vlprobe.o
	$(CXX) $^ -o $@ $(LDFLAGS)
	@echo "✓ $@"

$(BUILD)/audprobe: $(ENGINE_OBJS) $(BUILD)/tools/audprobe.o
	$(CXX) $^ -o $@ $(LDFLAGS)
	@echo "✓ $@"

$(BUILD)/mm2_check: tools/mm2_check.mm engine/src/metal_kernels.h
	@mkdir -p $(dir $@)
	$(CXX) $(OBJCXXFLAGS) $< -o $@ $(LDFLAGS)
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
	./$(BUILD)/optest

# 分块 matmul 与旧内核逐元素对拍 + 裸吞吐基准
mm2: $(BUILD)/mm2_check
	./$(BUILD)/mm2_check
	./$(BUILD)/mm2_check --bench

# GPU / CPU 对比
gpu: $(BIN)
	@echo "── CPU ──"
	./$(BIN) bench -m $(MODEL) -n 64 --no-gpu
	@echo "── GPU ──"
	./$(BIN) bench -m $(MODEL) -n 64
