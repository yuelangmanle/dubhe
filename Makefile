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
LDFLAGS  :=

BUILD    := build
BIN      := $(BUILD)/dubhe
SRV      := $(BUILD)/dubhed

ENGINE_SRCS := engine/src/gguf.cpp engine/src/quant.cpp engine/src/quant_q8.cpp \
               engine/src/tokenizer.cpp engine/src/model.cpp engine/src/engine.cpp
ENGINE_OBJS := $(patsubst %.cpp,$(BUILD)/%.o,$(ENGINE_SRCS))
CLI_OBJ     := $(BUILD)/cli/main.o
SRV_OBJ     := $(BUILD)/server/dubhed.o
TOOL_OBJS   := $(BUILD)/tools/selftest.o $(BUILD)/tools/ggufdump.o

MODEL ?= $(HOME)/.dubhe/models/qwen2.5-0.5b-instruct/qwen2.5-0.5b-instruct-q4_k_m.gguf

.PHONY: all clean tools run inspect bench serve check

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

$(BUILD)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

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

# 一条命令自检（ADR-0007 要求）
check: all
	./$(BUILD)/selftest $(MODEL) "你好，请用一句话介绍你自己。"
