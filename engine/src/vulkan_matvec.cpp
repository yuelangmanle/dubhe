#include "vulkan_matvec.h"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <vector>

namespace dubhe::vulkan {
namespace {

constexpr uint32_t kShader[] = {
#include "vulkan_q4_k.inc"
};
constexpr uint32_t kBindings = 3;

template <typename T>
T vk_info(VkStructureType type) {
    T info{};
    info.sType = type;
    return info;
}

struct Dimensions {
    uint32_t rows;
    uint32_t columns;

    std::array<VkDeviceSize, kBindings> sizes() const {
        return {VkDeviceSize(columns / 256) * 144 * rows,
                VkDeviceSize(columns) * sizeof(float), VkDeviceSize(rows) * sizeof(float)};
    }
};

struct Buffer {
    VkBuffer handle = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mapped = nullptr;
    VkDeviceSize capacity = 0;
};

class Matvec {
public:
    Matvec() : ready_(initialize()) {}
    ~Matvec();

    const char* device_name() const { return ready_ ? properties_.deviceName : nullptr; }
    bool supports(const Dimensions& shape) const;
    bool run(const void* weights, const float* input, float* output, const Dimensions& shape);

private:
    bool initialize();
    bool select_device();
    bool create_pipeline();
    bool allocate_commands();
    bool ensure_buffer(Buffer& buffer, VkDeviceSize size);
    void release(Buffer& buffer);

    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_ = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties properties_{};
    VkPhysicalDeviceMemoryProperties memory_{};
    uint32_t family_ = 0;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout bindings_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;
    VkDescriptorSet descriptors_ = VK_NULL_HANDLE;
    VkCommandPool command_pool_ = VK_NULL_HANDLE;
    VkCommandBuffer command_ = VK_NULL_HANDLE;
    std::array<Buffer, kBindings> buffers_{};
    std::mutex mutex_;
    const bool ready_;
};

Matvec::~Matvec() {
    if (device_) {
        for (auto& buffer : buffers_) release(buffer);
        vkDestroyCommandPool(device_, command_pool_, nullptr);
        vkDestroyDescriptorPool(device_, descriptor_pool_, nullptr);
        vkDestroyPipeline(device_, pipeline_, nullptr);
        vkDestroyPipelineLayout(device_, layout_, nullptr);
        vkDestroyDescriptorSetLayout(device_, bindings_, nullptr);
        vkDestroyDevice(device_, nullptr);
    }
    if (instance_) vkDestroyInstance(instance_, nullptr);
}

bool Matvec::initialize() {
    auto app = vk_info<VkApplicationInfo>(VK_STRUCTURE_TYPE_APPLICATION_INFO);
    app.pApplicationName = "Dubhe";
    app.apiVersion = VK_API_VERSION_1_0;
    auto info = vk_info<VkInstanceCreateInfo>(VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO);
    info.pApplicationInfo = &app;
    if (vkCreateInstance(&info, nullptr, &instance_) != VK_SUCCESS || !select_device()) return false;

    const float priority = 1.f;
    auto queue_info = vk_info<VkDeviceQueueCreateInfo>(VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO);
    queue_info.queueFamilyIndex = family_;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;
    auto device_info = vk_info<VkDeviceCreateInfo>(VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO);
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    if (vkCreateDevice(physical_, &device_info, nullptr, &device_) != VK_SUCCESS) return false;
    vkGetDeviceQueue(device_, family_, 0, &queue_);
    vkGetPhysicalDeviceMemoryProperties(physical_, &memory_);
    return create_pipeline() && allocate_commands();
}

bool Matvec::select_device() {
    uint32_t count = 0;
    if (vkEnumeratePhysicalDevices(instance_, &count, nullptr) != VK_SUCCESS || !count) return false;
    std::vector<VkPhysicalDevice> devices(count);
    if (vkEnumeratePhysicalDevices(instance_, &count, devices.data()) != VK_SUCCESS) return false;
    devices.resize(count);
    for (VkPhysicalDevice device : devices) {
        uint32_t families = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &families, nullptr);
        if (!families) continue;
        std::vector<VkQueueFamilyProperties> queues(families);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &families, queues.data());
        for (uint32_t family = 0; family < families; ++family) {
            if (!queues[family].queueCount || !(queues[family].queueFlags & VK_QUEUE_COMPUTE_BIT)) continue;
            VkPhysicalDeviceProperties candidate{};
            vkGetPhysicalDeviceProperties(device, &candidate);
            // 优先选 GPU，也允许只有软件 Vulkan 的模拟器跑数值测试。
            if (!physical_ || candidate.deviceType != VK_PHYSICAL_DEVICE_TYPE_CPU) {
                physical_ = device;
                properties_ = candidate;
                family_ = family;
            }
            break;
        }
        if (physical_ && properties_.deviceType != VK_PHYSICAL_DEVICE_TYPE_CPU) break;
    }
    return physical_ != VK_NULL_HANDLE;
}

bool Matvec::create_pipeline() {
    std::array<VkDescriptorSetLayoutBinding, kBindings> bindings{};
    for (uint32_t i = 0; i < bindings.size(); ++i) {
        bindings[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    }
    auto binding_info = vk_info<VkDescriptorSetLayoutCreateInfo>(VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO);
    binding_info.bindingCount = kBindings;
    binding_info.pBindings = bindings.data();
    if (vkCreateDescriptorSetLayout(device_, &binding_info, nullptr, &bindings_) != VK_SUCCESS) return false;

    const VkPushConstantRange dimensions{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Dimensions)};
    auto layout_info = vk_info<VkPipelineLayoutCreateInfo>(VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO);
    layout_info.setLayoutCount = 1;
    layout_info.pSetLayouts = &bindings_;
    layout_info.pushConstantRangeCount = 1;
    layout_info.pPushConstantRanges = &dimensions;
    if (vkCreatePipelineLayout(device_, &layout_info, nullptr, &layout_) != VK_SUCCESS) return false;

    VkShaderModule shader = VK_NULL_HANDLE;
    auto shader_info = vk_info<VkShaderModuleCreateInfo>(VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO);
    shader_info.codeSize = sizeof(kShader);
    shader_info.pCode = kShader;
    if (vkCreateShaderModule(device_, &shader_info, nullptr, &shader) != VK_SUCCESS) return false;
    auto pipeline_info = vk_info<VkComputePipelineCreateInfo>(VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO);
    pipeline_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_info.stage.module = shader;
    pipeline_info.stage.pName = "main";
    pipeline_info.layout = layout_;
    const VkResult result = vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline_);
    vkDestroyShaderModule(device_, shader, nullptr);
    return result == VK_SUCCESS;
}

bool Matvec::allocate_commands() {
    const VkDescriptorPoolSize storage{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kBindings};
    auto descriptor_pool_info = vk_info<VkDescriptorPoolCreateInfo>(VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO);
    descriptor_pool_info.maxSets = 1;
    descriptor_pool_info.poolSizeCount = 1;
    descriptor_pool_info.pPoolSizes = &storage;
    if (vkCreateDescriptorPool(device_, &descriptor_pool_info, nullptr, &descriptor_pool_) != VK_SUCCESS) return false;
    auto descriptor_info = vk_info<VkDescriptorSetAllocateInfo>(VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO);
    descriptor_info.descriptorPool = descriptor_pool_;
    descriptor_info.descriptorSetCount = 1;
    descriptor_info.pSetLayouts = &bindings_;
    if (vkAllocateDescriptorSets(device_, &descriptor_info, &descriptors_) != VK_SUCCESS) return false;

    auto command_pool_info = vk_info<VkCommandPoolCreateInfo>(VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO);
    command_pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    command_pool_info.queueFamilyIndex = family_;
    if (vkCreateCommandPool(device_, &command_pool_info, nullptr, &command_pool_) != VK_SUCCESS) return false;
    auto command_info = vk_info<VkCommandBufferAllocateInfo>(VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO);
    command_info.commandPool = command_pool_;
    command_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_info.commandBufferCount = 1;
    return vkAllocateCommandBuffers(device_, &command_info, &command_) == VK_SUCCESS;
}

void Matvec::release(Buffer& buffer) {
    if (buffer.mapped) vkUnmapMemory(device_, buffer.memory);
    vkDestroyBuffer(device_, buffer.handle, nullptr);
    vkFreeMemory(device_, buffer.memory, nullptr);
    buffer = {};
}

bool Matvec::ensure_buffer(Buffer& buffer, VkDeviceSize size) {
    if (buffer.capacity >= size) return true;
    release(buffer);
    auto info = vk_info<VkBufferCreateInfo>(VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO);
    info.size = size;
    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(device_, &info, nullptr, &buffer.handle) != VK_SUCCESS) return false;

    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(device_, buffer.handle, &requirements);
    constexpr auto coherent = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    for (uint32_t i = 0; i < memory_.memoryTypeCount; ++i) {
        if (!(requirements.memoryTypeBits & (1u << i)) ||
            (memory_.memoryTypes[i].propertyFlags & coherent) != coherent) continue;
        auto alloc = vk_info<VkMemoryAllocateInfo>(VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO);
        alloc.allocationSize = requirements.size;
        alloc.memoryTypeIndex = i;
        if (vkAllocateMemory(device_, &alloc, nullptr, &buffer.memory) != VK_SUCCESS) continue;
        if (vkBindBufferMemory(device_, buffer.handle, buffer.memory, 0) != VK_SUCCESS ||
            vkMapMemory(device_, buffer.memory, 0, VK_WHOLE_SIZE, 0, &buffer.mapped) != VK_SUCCESS) break;
        buffer.capacity = size;
        return true;
    }
    release(buffer);
    return false;
}

bool Matvec::supports(const Dimensions& shape) const {
    if (!ready_) return false;
    const auto& limits = properties_.limits;
    const uint64_t groups = (uint64_t)limits.maxComputeWorkGroupCount[0] * limits.maxComputeWorkGroupCount[1];
    const auto sizes = shape.sizes();
    return shape.rows <= groups && *std::max_element(sizes.begin(), sizes.end()) <= limits.maxStorageBufferRange;
}

bool Matvec::run(const void* weights, const float* input, float* output, const Dimensions& shape) {
    std::lock_guard lock(mutex_);
    const auto sizes = shape.sizes();
    std::array<VkDescriptorBufferInfo, kBindings> buffers{};
    std::array<VkWriteDescriptorSet, kBindings> writes{};
    for (uint32_t i = 0; i < buffers_.size(); ++i) {
        if (!ensure_buffer(buffers_[i], sizes[i])) return false;
        buffers[i] = {buffers_[i].handle, 0, sizes[i]};
        auto& write = writes[i];
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = descriptors_;
        write.dstBinding = i;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        write.pBufferInfo = &buffers[i];
    }
    std::memcpy(buffers_[0].mapped, weights, static_cast<size_t>(sizes[0]));
    std::memcpy(buffers_[1].mapped, input, static_cast<size_t>(sizes[1]));
    vkUpdateDescriptorSets(device_, kBindings, writes.data(), 0, nullptr);

    auto begin = vk_info<VkCommandBufferBeginInfo>(VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO);
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkResetCommandPool(device_, command_pool_, 0) != VK_SUCCESS ||
        vkBeginCommandBuffer(command_, &begin) != VK_SUCCESS) return false;
    vkCmdBindPipeline(command_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
    vkCmdBindDescriptorSets(command_, VK_PIPELINE_BIND_POINT_COMPUTE, layout_, 0, 1, &descriptors_, 0, nullptr);
    vkCmdPushConstants(command_, layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(shape), &shape);
    const uint32_t width = std::min(shape.rows, properties_.limits.maxComputeWorkGroupCount[0]);
    vkCmdDispatch(command_, width, 1 + (shape.rows - 1) / width, 1);
    auto visibility = vk_info<VkMemoryBarrier>(VK_STRUCTURE_TYPE_MEMORY_BARRIER);
    visibility.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    visibility.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(command_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0,
                         1, &visibility, 0, nullptr, 0, nullptr);
    if (vkEndCommandBuffer(command_) != VK_SUCCESS) return false;

    auto submit = vk_info<VkSubmitInfo>(VK_STRUCTURE_TYPE_SUBMIT_INFO);
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command_;
    if (vkQueueSubmit(queue_, 1, &submit, VK_NULL_HANDLE) != VK_SUCCESS || vkQueueWaitIdle(queue_) != VK_SUCCESS) {
        return false;
    }
    std::memcpy(output, buffers_[2].mapped, static_cast<size_t>(sizes[2]));
    return true;
}

Matvec& context() {
    static Matvec instance;
    return instance;
}

}  // namespace

const char* device_name() { return context().device_name(); }

bool supports(int rows, int columns) {
    return rows > 0 && columns > 0 && columns % 256 == 0 &&
           context().supports({static_cast<uint32_t>(rows), static_cast<uint32_t>(columns)});
}

bool matvec_q4_k(const void* weights, const float* input, float* output, int rows, int columns) {
    if (!weights || !input || !output || !supports(rows, columns)) return false;
    return context().run(weights, input, output, {static_cast<uint32_t>(rows), static_cast<uint32_t>(columns)});
}

}  // namespace dubhe::vulkan
