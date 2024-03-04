#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>
#include <list>
#include <map>
#include <vector>
#include <chrono>


#include <vulkan/vulkan.hpp>
#define NUM 1048576
#define THREAD_PER_WARP 32
using namespace std;

class PipelineComponents {
public:
  vk::DescriptorSetLayout descriptorSetLayout;
  vk::PipelineLayout pipelineLayout;
  vk::PipelineCache pipelineCache;
  vk::Pipeline computePipeline;
};

class CommandComponents {
public:
  vk::CommandPool cmdPool;
  vk::CommandBuffer cmdBuffer;
  vk::Fence fence;
  vk::Queue queue;
};

vk::Instance getVulkanInstance() {
  vk::ApplicationInfo AppInfo{
      "VulkanCompute",   // Application Name
      1,                 // Application Version
      nullptr,           // Engine Name or nullptr
      0,                 // Engine Version
      VK_API_VERSION_1_1 // Vulkan API version
  };

  const std::vector<const char *> Layers = {"VK_LAYER_KHRONOS_validation"};
  vk::InstanceCreateInfo InstanceCreateInfo(vk::InstanceCreateFlags(), // Flags
                                            &AppInfo,      // Application Info
                                            Layers.size(), // Layers count
                                            Layers.data()  // Layers
  );
  vk::Instance instance = vk::createInstance(InstanceCreateInfo);

  return instance;
}

vk::PhysicalDevice getPhysicalDevice(vk::Instance &instance) {
  vk::PhysicalDevice physicalDevice =
      instance.enumeratePhysicalDevices().front();
  vk::PhysicalDeviceProperties deviceProps = physicalDevice.getProperties();
  std::cout << "Device Name    : " << deviceProps.deviceName << std::endl;
  const uint32_t ApiVersion = deviceProps.apiVersion;
  std::cout << "Vulkan Version : " << VK_VERSION_MAJOR(ApiVersion) << "."
            << VK_VERSION_MINOR(ApiVersion) << "."
            << VK_VERSION_PATCH(ApiVersion) << std::endl;
  vk::PhysicalDeviceLimits deviceLimits = deviceProps.limits;
  std::cout << "Max Compute Shared Memory Size: "
            << deviceLimits.maxComputeSharedMemorySize / 1024 << " KB"
            << std::endl;
  return physicalDevice;
}

uint32_t getComputeQueueFamilyIndex(vk::PhysicalDevice &physicalDevice) {
  vector<vk::QueueFamilyProperties> queueFamilyProps =
      physicalDevice.getQueueFamilyProperties();
  auto PropIt =
      std::find_if(queueFamilyProps.begin(), queueFamilyProps.end(),
                   [](const vk::QueueFamilyProperties &Prop) {
                     return Prop.queueFlags & vk::QueueFlagBits::eCompute;
                   });
  uint32_t computeQueueFamilyIndex = distance(queueFamilyProps.begin(), PropIt);
  cout << "Compute Queue Family Index: " << computeQueueFamilyIndex << endl;
  return computeQueueFamilyIndex;
}

vk::Device getDevice(uint32_t computeQueueFamilyIndex,
                     vk::PhysicalDevice &physicalDevice) {
  float queuePriorities = 1.0f;
  vk::DeviceQueueCreateInfo DeviceQueueCreateInfo(
      vk::DeviceQueueCreateFlags(), // Flags
      computeQueueFamilyIndex,      // Queue Family Index
      1,                            // Number of Queues
      &queuePriorities);
  vk::DeviceCreateInfo DeviceCreateInfo(
      vk::DeviceCreateFlags(), // Flags
      1,
      &DeviceQueueCreateInfo // Device Queue Create Info struct
  );
  vk::Device device = physicalDevice.createDevice(DeviceCreateInfo);
  return device;
}

map<string, vector<pair<vk::Buffer, vk::DeviceMemory>>>
getBuffers(vk::Device &device, vk::PhysicalDevice &physicalDevice,
           uint32_t computeQueueFamilyIndex) {
  const uint32_t NumElements = NUM;
  const uint32_t BufferSize = NumElements * sizeof(float);

  vk::BufferCreateInfo bufferCreateInfo{
      vk::BufferCreateFlags(),                 // Flags
      BufferSize,                              // Size
      vk::BufferUsageFlagBits::eStorageBuffer, // Usage
      vk::SharingMode::eExclusive,             // Sharing mode
      1,                                       // Number of queue family indices
      &computeQueueFamilyIndex                 // List of queue family indices
  };
  vk::Buffer inBuffer = device.createBuffer(bufferCreateInfo);
  vk::Buffer outBuffer = device.createBuffer(bufferCreateInfo);

  // Memory req
  vk::MemoryRequirements inBufferMemoryRequirements =
      device.getBufferMemoryRequirements(inBuffer);
  vk::MemoryRequirements outBufferMemoryRequirements =
      device.getBufferMemoryRequirements(outBuffer);

  // query
  vk::PhysicalDeviceMemoryProperties memoryProperties =
      physicalDevice.getMemoryProperties();

  uint32_t memoryTypeIndex = uint32_t(~0);
  vk::DeviceSize memoryHeapSize = uint32_t(~0);
  for (uint32_t currentMemoryTypeIndex = 0;
       currentMemoryTypeIndex < memoryProperties.memoryTypeCount;
       ++currentMemoryTypeIndex) {
    vk::MemoryType memoryType =
        memoryProperties.memoryTypes[currentMemoryTypeIndex];
    if ((vk::MemoryPropertyFlagBits::eHostVisible & memoryType.propertyFlags) &&
        (vk::MemoryPropertyFlagBits::eHostCoherent &
         memoryType.propertyFlags)) {
      memoryHeapSize = memoryProperties.memoryHeaps[memoryType.heapIndex].size;
      memoryTypeIndex = currentMemoryTypeIndex;
      break;
    }
  }

  std::cout << "Memory Type Index: " << memoryTypeIndex << std::endl;
  std::cout << "Memory Heap Size : " << memoryHeapSize / 1024 / 1024 / 1024
            << " GB" << std::endl;

  // Allocate memory
  vk::MemoryAllocateInfo inBufferMemoryAllocateInfo(
      inBufferMemoryRequirements.size, memoryTypeIndex);
  vk::MemoryAllocateInfo outBufferMemoryAllocateInfo(
      outBufferMemoryRequirements.size, memoryTypeIndex);
  vk::DeviceMemory inBufferMemory =
      device.allocateMemory(inBufferMemoryAllocateInfo);
  vk::DeviceMemory outBufferMemory =
      device.allocateMemory(outBufferMemoryAllocateInfo);

  // Map memory and write
  float *inBufferPtr =
      static_cast<float *>(device.mapMemory(inBufferMemory, 0, BufferSize));
  for (uint32_t k = 0; k < NumElements; ++k) {
    inBufferPtr[k] = (float)k -(NumElements/2);
  }
  device.unmapMemory(inBufferMemory);

  // Bind buffers to memory
  device.bindBufferMemory(inBuffer, inBufferMemory, 0);
  device.bindBufferMemory(outBuffer, outBufferMemory, 0);

  vector<pair<vk::Buffer, vk::DeviceMemory>> inputVectors = {
      make_pair(inBuffer, inBufferMemory)};
  vector<pair<vk::Buffer, vk::DeviceMemory>> outputVectors = {
      make_pair(outBuffer, outBufferMemory)};
  map<string, vector<pair<vk::Buffer, vk::DeviceMemory>>> ret;
  ret["input"] = inputVectors;
  ret["output"] = outputVectors;

  return ret;
}

void cleanup(vk::Device &device, vk::Instance &instance,
             map<string, vector<pair<vk::Buffer, vk::DeviceMemory>>> &buffers,
             PipelineComponents &pComponents, vk::ShaderModule &shaderModule,
             pair<vk::DescriptorPool, vk::DescriptorSet> &descriptorObjects, CommandComponents& cmdComponents) {
  printf("Starting cleanup.\n");

  device.resetCommandPool(cmdComponents.cmdPool, vk::CommandPoolResetFlags());
  device.destroyFence(cmdComponents.fence);


  device.destroyDescriptorPool(descriptorObjects.first);

  device.destroyDescriptorSetLayout(pComponents.descriptorSetLayout);
  device.destroyPipelineLayout(pComponents.pipelineLayout);
  device.destroyPipelineCache(pComponents.pipelineCache);
  device.destroyPipeline(pComponents.computePipeline);
  device.destroyShaderModule(shaderModule);
  device.destroyCommandPool(cmdComponents.cmdPool);

  vector<pair<vk::Buffer, vk::DeviceMemory>> inputs = buffers["input"];
  vector<pair<vk::Buffer, vk::DeviceMemory>> outputs = buffers["output"];
  for (auto item : inputs) {
    device.freeMemory(item.second);
    device.destroyBuffer(item.first);
  }
  for (auto item : outputs) {
    device.freeMemory(item.second);
    device.destroyBuffer(item.first);
  }
  device.destroy();
  instance.destroy();
}

vk::ShaderModule getPipeline(vk::Device &device,
                             PipelineComponents &pipelineComponents) {
  std::vector<char> shaderContents;
  if (std::ifstream shaderFile{"shaders/relu.spv",
                               std::ios::binary | std::ios::ate}) {
    const size_t fileSize =
        shaderFile.tellg(); // The file opened at the end. Getting the position
                            // here is the filesize
    shaderFile.seekg(0);    // Resetting file pointer to the front.
    shaderContents.resize(fileSize, '\0');
    shaderFile.read(shaderContents.data(), fileSize);
  }

  vk::ShaderModuleCreateInfo shaderModuleCreateInfo(
      vk::ShaderModuleCreateFlags(),                              // Flags
      shaderContents.size(),                                      // Code size
      reinterpret_cast<const uint32_t *>(shaderContents.data())); // Code
  vk::ShaderModule shaderModule =
      device.createShaderModule(shaderModuleCreateInfo);

  const std::vector<vk::DescriptorSetLayoutBinding> descriptorSetLayoutBinding =
      {{0, vk::DescriptorType::eStorageBuffer, 1,
        vk::ShaderStageFlagBits::eCompute},
       {1, vk::DescriptorType::eStorageBuffer, 1,
        vk::ShaderStageFlagBits::eCompute}};
  vk::DescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo(
      vk::DescriptorSetLayoutCreateFlags(), descriptorSetLayoutBinding);
  vk::DescriptorSetLayout descriptorSetLayout =
      device.createDescriptorSetLayout(descriptorSetLayoutCreateInfo);

  // Pipeline Layout
  vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo(
      vk::PipelineLayoutCreateFlags(), descriptorSetLayout);
  vk::PipelineLayout pipelineLayout =
      device.createPipelineLayout(pipelineLayoutCreateInfo);
  vk::PipelineCache pipelineCache =
      device.createPipelineCache(vk::PipelineCacheCreateInfo());

  // Compute Pipeline
  vk::PipelineShaderStageCreateInfo pipelineShaderCreateInfo(
      vk::PipelineShaderStageCreateFlags(), // Flags
      vk::ShaderStageFlagBits::eCompute,    // Stage
      shaderModule,                         // Shader Module
      "main"                                // Shader Entry Point
  );
  vk::ComputePipelineCreateInfo computePipelineCreateInfo(
      vk::PipelineCreateFlags(), // Flags
      pipelineShaderCreateInfo,  // Shader Create Info struct
      pipelineLayout             // Pipeline Layout
  );
  vk::Pipeline computePipeline =
      device.createComputePipeline(pipelineCache, computePipelineCreateInfo)
          .value;

  pipelineComponents.computePipeline = computePipeline;
  pipelineComponents.descriptorSetLayout = descriptorSetLayout;
  pipelineComponents.pipelineCache = pipelineCache;
  pipelineComponents.pipelineLayout = pipelineLayout;

  return shaderModule;
}

pair<vk::DescriptorPool, vk::DescriptorSet>
getDescriptors(vk::Device &device, vk::DescriptorSetLayout &descriptorSetLayout,
               vk::Buffer &inBuffer, vk::Buffer &outBuffer,
               uint32_t numElements) {
  // Descriptor sets must be allocated in a vk::DescriptorPool, so we need to
  // create one first
  vk::DescriptorPoolSize descriptorPoolSize(vk::DescriptorType::eStorageBuffer,
                                            2);
  vk::DescriptorPoolCreateInfo descriptorPoolCreateInfo(
      vk::DescriptorPoolCreateFlags(), 1, descriptorPoolSize);
  vk::DescriptorPool descriptorPool =
      device.createDescriptorPool(descriptorPoolCreateInfo);

  // Allocate descriptor sets, update them to use buffers:
  vk::DescriptorSetAllocateInfo descriptorSetAllocInfo(descriptorPool, 1,
                                                       &descriptorSetLayout);
  const std::vector<vk::DescriptorSet> descriptorSets =
      device.allocateDescriptorSets(descriptorSetAllocInfo);
  vk::DescriptorSet descriptorSet = descriptorSets.front();
  vk::DescriptorBufferInfo inBufferInfo(inBuffer, 0,
                                        numElements * sizeof(int32_t));
  vk::DescriptorBufferInfo outBufferInfo(outBuffer, 0,
                                         numElements * sizeof(int32_t));

  const std::vector<vk::WriteDescriptorSet> writeDescriptorSets = {
      {descriptorSet, 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr,
       &inBufferInfo},
      {descriptorSet, 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr,
       &outBufferInfo},
  };
  device.updateDescriptorSets(writeDescriptorSets, {});
  return make_pair(descriptorPool, descriptorSet);
}

CommandComponents getCommandComponents(vk::Device &device,
                                       uint32_t computeQueueFamilyIndex,
                                       vk::DescriptorSet &descriptorset,
                                       PipelineComponents &pipelineComponents) {
  vk::CommandPoolCreateInfo commandPoolCreateInfo(vk::CommandPoolCreateFlags(),
                                                  computeQueueFamilyIndex);
  vk::CommandPool commandPool = device.createCommandPool(commandPoolCreateInfo);
  // Allocate Command buffer from Pool
  vk::CommandBufferAllocateInfo commandBufferAllocInfo(
      commandPool,                      // Command Pool
      vk::CommandBufferLevel::ePrimary, // Level
      1);                               // Num Command Buffers
  const std::vector<vk::CommandBuffer> cmdBuffers =
      device.allocateCommandBuffers(commandBufferAllocInfo);
  vk::CommandBuffer cmdBuffer = cmdBuffers.front();

  // Record commands
  vk::CommandBufferBeginInfo cmdBufferBeginInfo(
      vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
  cmdBuffer.begin(cmdBufferBeginInfo);
  cmdBuffer.bindPipeline(vk::PipelineBindPoint::eCompute,
                         pipelineComponents.computePipeline);
  cmdBuffer.bindDescriptorSets(
      vk::PipelineBindPoint::eCompute,   // Bind point
      pipelineComponents.pipelineLayout, // Pipeline Layout
      0,                                 // First descriptor set
      {descriptorset},                   // List of descriptor sets
      {});                               // Dynamic offsets
  cmdBuffer.dispatch(NUM/THREAD_PER_WARP, 1, 1);
  cmdBuffer.end();

  // Fence and submit
  vk::Queue queue = device.getQueue(computeQueueFamilyIndex, 0);
  vk::Fence fence = device.createFence(vk::FenceCreateInfo());

  CommandComponents commandComponents;
  commandComponents.cmdBuffer = cmdBuffer;
  commandComponents.cmdPool = commandPool;
  commandComponents.queue = queue;
  commandComponents.fence = fence;

  return commandComponents;
}

void execute(CommandComponents &cmdComponents, vk::Device &device) {
  vk::SubmitInfo submitInfo(0, nullptr, nullptr, 1, &(cmdComponents.cmdBuffer));
  cmdComponents.queue.submit({submitInfo}, cmdComponents.fence);
  (void)device.waitForFences({cmdComponents.fence}, true, uint64_t(-1));
}

void showResult(vk::DeviceMemory& inBufferMemory , vk::DeviceMemory& outBufferMemory, vk::Device& device){
  float *inBufferPtr = static_cast<float *>(device.mapMemory(inBufferMemory, 0 , NUM * sizeof(float)));
  float *outBufferPtr = static_cast<float *>(
      device.mapMemory(outBufferMemory, 0, NUM * sizeof(int32_t)));

  for(int k = 0;k<NUM;k++){
    if(outBufferPtr[k] != max(0.0f , inBufferPtr[k])){
        printf("Mismatch at %d , %f %f\n", k, inBufferPtr[k], outBufferPtr[k]);
    }
  }

  printf("Mismatch check complete.\n");

  device.unmapMemory(inBufferMemory);
  device.unmapMemory(outBufferMemory);

}

int main(int argc, char const *argv[]) {
  // Create a bunch of stuff.
  vk::Instance currentInstance = getVulkanInstance();
  vk::PhysicalDevice physicalDevice = getPhysicalDevice(currentInstance);
  uint32_t computeQueueFamilyIndex = getComputeQueueFamilyIndex(physicalDevice);
  vk::Device device = getDevice(computeQueueFamilyIndex, physicalDevice);
  map<string, vector<pair<vk::Buffer, vk::DeviceMemory>>> buffers =
      getBuffers(device, physicalDevice, computeQueueFamilyIndex);

  PipelineComponents pComponents;
  vk::ShaderModule shaderModule = getPipeline(device, pComponents);
  auto descriptorObjects = getDescriptors(
      device, pComponents.descriptorSetLayout, buffers["input"][0].first,
      buffers["output"][0].first, NUM);

  printf("Starting execution.\n");
  auto start = chrono::system_clock::now();
  CommandComponents cmdComponents = getCommandComponents(device, computeQueueFamilyIndex, descriptorObjects.second, pComponents);
  execute(cmdComponents, device);
  auto end = chrono::system_clock::now();
  auto elapsed_time = chrono::duration_cast<chrono::microseconds>(end-start);
  cout<<"Elapsed time in microseconds: " << elapsed_time.count() <<endl;

  showResult(buffers["input"][0].second, buffers["output"][0].second, device);

  // Cleanup all the bunch of stuffs.
  cleanup(device, currentInstance, buffers, pComponents, shaderModule,
          descriptorObjects, cmdComponents);
  return 0;
}
