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

BUILD    := build
BIN      := $(BUILD)/dubhe
SRV      := $(BUILD)/dubhed

UNAME_S  := $(shell uname -s)

ENGINE_SRCS := engine/src/gguf.cpp engine/src/quant.cpp engine/src/quant_q8.cpp \
               engine/src/tokenizer.cpp engine/src/model.cpp engine/src/engine.cpp

# Metal 后端只在 Apple 平台编译。着色器在运行时编译（newLibraryWithSource:），
# 所以不需要离线 Metal Toolchain。
ifeq ($(UNAME_S),Darwin)
  ENGINE_SRCS += engine/src/metal.mm
  OBJCXXFLAGS := $(CXXFLAGS) -fobjc-arc
  LDFLAGS     := -framework Metal -framework Foundation
else
  ENGINE_SRCS += engine/src/metal_stub.cpp
  LDFLAGS     :=
endif

ENGINE_OBJS := $(patsubst %.cpp,$(BUILD)/%.o,$(patsubst %.mm,$(BUILD)/%.o,$(ENGINE_SRCS)))
CLI_OBJ     := $(BUILD)/cli/main.o
SRV_OBJ     := $(BUILD)/server/dubhed.o
TOOL_OBJS   := $(BUILD)/tools/selftest.o $(BUILD)/tools/ggufdump.o

MODEL ?= $(HOME)/.dubhe/models/qwen2.5-0.5b-instruct/qwen2.5-0.5b-instruct-q4_k_m.gguf

.PHONY: all clean tools run inspect bench serve check gpu optest

all: $(BIN) $(SRV) tools

tools: $(BUILD)/selftest $(BUILD)/ggufdump

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
optest: $(BUILD)/optest
	./$(BUILD)/optest

# GPU / CPU 对比
gpu: $(BIN)
	@echo "── CPU ──"
	./$(BIN) bench -m $(MODEL) -n 64 --no-gpu
	@echo "── GPU ──"
	./$(BIN) bench -m $(MODEL) -n 64
