LOCAL_PATH := $(call my-dir)
DUBHE_ENGINE := ../../../../../engine

include $(CLEAR_VARS)
LOCAL_MODULE := dubhe_jni
LOCAL_SRC_FILES := \
    jni_bridge.cpp android_media.cpp \
    $(addprefix $(DUBHE_ENGINE)/src/, \
        gguf.cpp quant.cpp quant_q8.cpp tokenizer.cpp model.cpp engine.cpp \
        mmproj.cpp mel.cpp audiomm.cpp metal_stub.cpp vit_gpu_stub.cpp)
LOCAL_C_INCLUDES := $(addprefix $(LOCAL_PATH)/$(DUBHE_ENGINE)/,include src)
LOCAL_CPPFLAGS := -std=c++20 -O3 -DNDEBUG -fvisibility=hidden
LOCAL_CPP_FEATURES := exceptions rtti
LOCAL_LDFLAGS := -Wl,-z,max-page-size=16384
LOCAL_LDLIBS := -ljnigraphics
ifeq ($(DUBHE_ENABLE_VULKAN),1)
LOCAL_SRC_FILES += $(DUBHE_ENGINE)/src/vulkan_matvec.cpp
LOCAL_C_INCLUDES += $(DUBHE_VULKAN_SHADER_DIR)
LOCAL_CPPFLAGS += -DDUBHE_VULKAN=1
LOCAL_LDLIBS += -lvulkan
endif
include $(BUILD_SHARED_LIBRARY)
