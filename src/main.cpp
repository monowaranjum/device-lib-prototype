#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <list>
#include <map>
#include <random>
#include <vector>

#include <vulkan/vulkan.hpp>

#define epsilon 1e-3f
#define LOCAL_X 64
#define LOCAL_Y 64

#define NAIVE 0
#define MEM_COALESCED 1
#define SMEM_CACHE 2
#define TILE_1D 3
#define TIlE_2D 4
#define LATENCY_HIDING 5
#define WARP_TILED 6
#define DOUBLE_BUFFER 7

using namespace std;

bool isPowerOfTwo(uint32_t n) { return n && !(n & (n - 1)); }

void matMulGen(float *a, float *b, float *c, int M, int K, int N,
               bool compute = false) {
  random_device rd;
  mt19937 e2(rd());
  uniform_real_distribution<> dist(-10.0, 10.0);

  // Populate

  for (int i = 0; i < M * K; i++) {
    a[i] = dist(e2);
  }

  for (int i = 0; i < K * N; i++) {
    b[i] = dist(e2);
  }

  // Do the actual Matmul

  if (compute) {
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        float temp = 0.0;
        for (int k = 0; k < K; k++) {
          temp += a[i * K + k] * b[k * N + j];
        }
        c[i * N + j] = temp;
      }
    }
  }
}

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
      "VulkanComputeGEMM", // Application Name
      1,                   // Application Version
      nullptr,             // Engine Name or nullptr
      0,                   // Engine Version
      VK_API_VERSION_1_1   // Vulkan API version
  };

  const std::vector<const char *> Layers = {};
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
  std::cout << "Max Compute Shader Local Size (Max Allowable thread count): "
            << deviceLimits.maxComputeWorkGroupInvocations << std::endl;
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
getMatmulBuffers(vk::Device &device, vk::PhysicalDevice &physicalDevice,
                 uint32_t computeQueueFamilyIndex, float *a, float *b,
                 uint32_t *dim, int M, int K, int N) {

  // Matrix A is M X K
  // Matrix B is K X N
  // Matrix C is M X N

  const uint32_t BufferSize_A = M * K * sizeof(float); // All buffers are fp32
  const uint32_t BufferSize_B = K * N * sizeof(float);
  const uint32_t BufferSize_C = M * N * sizeof(float);
  const uint32_t BufferSize_dim =
      3 *
      sizeof(uint32_t); // The dimensions of the input matrices are given here.

  vk::BufferCreateInfo bufferCreateInfo_A{
      vk::BufferCreateFlags(),                 // Flags
      BufferSize_A,                            // Size
      vk::BufferUsageFlagBits::eStorageBuffer, // Usage
      vk::SharingMode::eExclusive,             // Sharing mode
      1,                                       // Number of queue family indices
      &computeQueueFamilyIndex                 // List of queue family indices
  };

  vk::BufferCreateInfo bufferCreateInfo_B{
      vk::BufferCreateFlags(),                 // Flags
      BufferSize_B,                            // Size
      vk::BufferUsageFlagBits::eStorageBuffer, // Usage
      vk::SharingMode::eExclusive,             // Sharing mode
      1,                                       // Number of queue family indices
      &computeQueueFamilyIndex                 // List of queue family indices
  };

  vk::BufferCreateInfo bufferCreateInfo_C{
      vk::BufferCreateFlags(),                 // Flags
      BufferSize_C,                            // Size
      vk::BufferUsageFlagBits::eStorageBuffer, // Usage
      vk::SharingMode::eExclusive,             // Sharing mode
      1,                                       // Number of queue family indices
      &computeQueueFamilyIndex                 // List of queue family indices
  };

  vk::BufferCreateInfo bufferCreateInfo_dim{
      vk::BufferCreateFlags(),                 // Flags
      BufferSize_dim,                          // Size
      vk::BufferUsageFlagBits::eStorageBuffer, // Usage
      vk::SharingMode::eExclusive,             // Sharing mode
      1,                                       // Number of queue family indices
      &computeQueueFamilyIndex                 // List of queue family indices
  };

  vk::Buffer inBuffer_A = device.createBuffer(bufferCreateInfo_A);
  vk::Buffer inBuffer_B = device.createBuffer(bufferCreateInfo_B);
  vk::Buffer inBuffer_dim = device.createBuffer(bufferCreateInfo_dim);
  vk::Buffer outBuffer_C = device.createBuffer(bufferCreateInfo_C);

  // Memory req
  vk::MemoryRequirements inBuffer_A_MemoryRequirements =
      device.getBufferMemoryRequirements(inBuffer_A);

  vk::MemoryRequirements inBuffer_B_MemoryRequirements =
      device.getBufferMemoryRequirements(inBuffer_B);

  vk::MemoryRequirements inBuffer_dim_MemoryRequirements =
      device.getBufferMemoryRequirements(inBuffer_dim);

  vk::MemoryRequirements outBuffer_C_MemoryRequirements =
      device.getBufferMemoryRequirements(outBuffer_C);

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
  vk::MemoryAllocateInfo inBuffer_A_MemoryAllocateInfo(
      inBuffer_A_MemoryRequirements.size, memoryTypeIndex);
  vk::MemoryAllocateInfo inBuffer_B_MemoryAllocateInfo(
      inBuffer_B_MemoryRequirements.size, memoryTypeIndex);
  vk::MemoryAllocateInfo inBuffer_dim_MemoryAllocateInfo(
      inBuffer_dim_MemoryRequirements.size, memoryTypeIndex);
  vk::MemoryAllocateInfo outBuffer_C_MemoryAllocateInfo(
      outBuffer_C_MemoryRequirements.size, memoryTypeIndex);

  vk::DeviceMemory inBuffer_A_Memory =
      device.allocateMemory(inBuffer_A_MemoryAllocateInfo);

  vk::DeviceMemory inBuffer_B_Memory =
      device.allocateMemory(inBuffer_B_MemoryAllocateInfo);

  vk::DeviceMemory inBuffer_dim_Memory =
      device.allocateMemory(inBuffer_dim_MemoryAllocateInfo);

  vk::DeviceMemory outBuffer_C_Memory =
      device.allocateMemory(outBuffer_C_MemoryAllocateInfo);

  // Map memory and write
  float *inBuffer_A_Ptr = static_cast<float *>(
      device.mapMemory(inBuffer_A_Memory, 0, BufferSize_A));
  float *inBuffer_B_Ptr = static_cast<float *>(
      device.mapMemory(inBuffer_B_Memory, 0, BufferSize_B));
  uint32_t *inBuffer_dim_Ptr = static_cast<uint32_t *>(
      device.mapMemory(inBuffer_dim_Memory, 0, BufferSize_dim));

  for (int k = 0; k < M * K; ++k) {
    inBuffer_A_Ptr[k] = a[k];
  }
  for (int k = 0; k < K * N; k++) {
    inBuffer_B_Ptr[k] = b[k];
  }
  inBuffer_dim_Ptr[0] = M;
  inBuffer_dim_Ptr[1] = K;
  inBuffer_dim_Ptr[2] = N;

  device.unmapMemory(inBuffer_A_Memory);
  device.unmapMemory(inBuffer_B_Memory);
  device.unmapMemory(inBuffer_dim_Memory);

  // Bind buffers to memory
  device.bindBufferMemory(inBuffer_A, inBuffer_A_Memory,
                          0); // Buffer Object to DeviceMemoryObject mapping
  device.bindBufferMemory(inBuffer_B, inBuffer_B_Memory, 0);
  device.bindBufferMemory(inBuffer_dim, inBuffer_dim_Memory, 0);
  device.bindBufferMemory(outBuffer_C, outBuffer_C_Memory, 0);

  vector<pair<vk::Buffer, vk::DeviceMemory>> inputVectors = {
      make_pair(inBuffer_A, inBuffer_A_Memory),
      make_pair(inBuffer_B, inBuffer_B_Memory),
      make_pair(inBuffer_dim, inBuffer_dim_Memory)};
  vector<pair<vk::Buffer, vk::DeviceMemory>> outputVectors = {
      make_pair(outBuffer_C, outBuffer_C_Memory)};
  map<string, vector<pair<vk::Buffer, vk::DeviceMemory>>> ret;
  ret["input"] = inputVectors;
  ret["output"] = outputVectors;

  return ret;
}

void cleanup(vk::Device &device, vk::Instance &instance,
             map<string, vector<pair<vk::Buffer, vk::DeviceMemory>>> &buffers,
             PipelineComponents &pComponents, vk::ShaderModule &shaderModule,
             pair<vk::DescriptorPool, vk::DescriptorSet> &descriptorObjects,
             CommandComponents &cmdComponents) {
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
                             PipelineComponents &pipelineComponents,
                             string shader_name) {
  std::vector<char> shaderContents;
  string file_name =
      "/home/rashik/Documents/compute_shader_exp/build/shaders/" + shader_name;
  if (std::ifstream shaderFile{file_name.c_str(),
                               std::ios::binary | std::ios::ate}) {
    const size_t fileSize =
        shaderFile.tellg(); // The file opened at the end. Getting the position
                            // here is the filesize
    shaderFile.seekg(0);    // Resetting file pointer to the front.
    shaderContents.resize(fileSize, '\0');
    shaderFile.read(shaderContents.data(), fileSize);
  } else {
    printf("Shader file not found or unreadable.\n");
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
        vk::ShaderStageFlagBits::eCompute},
       {2, vk::DescriptorType::eStorageBuffer, 1,
        vk::ShaderStageFlagBits::eCompute},
       {3, vk::DescriptorType::eStorageBuffer, 1,
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
               vector<pair<vk::Buffer, vk::DeviceMemory>> &inBuffers,
               vector<pair<vk::Buffer, vk::DeviceMemory>> &outBuffers, int m,
               int k, int n) {
  // Descriptor sets must be allocated in a vk::DescriptorPool, so we need to
  // create one first
  vk::DescriptorPoolSize descriptorPoolSize(vk::DescriptorType::eStorageBuffer,
                                            4); // 4 storage buffers for matmul
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

  vk::DescriptorBufferInfo inBuffer_A_Info(inBuffers[0].first, 0,
                                           m * k * sizeof(float));
  vk::DescriptorBufferInfo inBuffer_B_Info(inBuffers[1].first, 0,
                                           k * n * sizeof(float));
  vk::DescriptorBufferInfo inBuffer_dim_Info(inBuffers[2].first, 0,
                                             3 * sizeof(uint32_t));
  vk::DescriptorBufferInfo outBuffer_C_Info(outBuffers[0].first, 0,
                                            m * n * sizeof(float));

  const std::vector<vk::WriteDescriptorSet> writeDescriptorSets = {
      {descriptorSet, 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr,
       &inBuffer_A_Info},
      {descriptorSet, 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr,
       &inBuffer_B_Info},
      {descriptorSet, 2, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr,
       &inBuffer_dim_Info},
      {descriptorSet, 3, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr,
       &outBuffer_C_Info},
  };
  device.updateDescriptorSets(writeDescriptorSets, {});
  return make_pair(descriptorPool, descriptorSet);
}

void conditionalDispatch(vk::CommandBuffer &cmdBuffer, int M, int N,
                         int kernelCode) {
  int m = isPowerOfTwo(M) ? (M / 64) : (M / 64) + 1;
  int n = isPowerOfTwo(N) ? (N / 8) : (N / 8) + 1;
  switch (kernelCode) {
  case NAIVE:
    cmdBuffer.dispatch(m, n, 1);
    printf("NAIVE: Dispatch size (%d, %d, 1)\nTotal WG count %d\n", m, n, m*n);
    break;
  case MEM_COALESCED:
    cmdBuffer.dispatch(m, n, 1);
    printf("MEM_COALESCED: Dispatch size (%d, %d , 1)\nTotal WG count: %d\n", m, n, m*n );
    break;
  case SMEM_CACHE:
    m = isPowerOfTwo(M) ? (M / 16) : (M / 16) + 1;
    n = isPowerOfTwo(N) ? (N / 16) : (N / 16) + 1;
    cmdBuffer.dispatch(m, n, 1);
    printf("SMEM_CACHE: Dispatch size (%d, %d, 1)\nTotal WG count %d\n", m, n, m*n);
    break;
  case TILE_1D:
    m = isPowerOfTwo(M) ? (M / 64) : (M / 64) + 1;
    n = isPowerOfTwo(N) ? (N / 64) : (N / 64) + 1;
    cmdBuffer.dispatch(m, n, 1);
    printf("TILE_1D: Dispatch size (%d, %d, 1)\nTotal WG count %d\n", m, n, m*n);
    break;
  default:
    printf("Kernel code %d not recognized. Implementation maybe missing.\n", kernelCode);
    break;
  }
}

CommandComponents getCommandComponents(vk::Device &device,
                                       uint32_t computeQueueFamilyIndex,
                                       vk::DescriptorSet &descriptorset,
                                       PipelineComponents &pipelineComponents,
                                       int M, int N, int kernelId) {
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

  //   if (isPowerOfTwo(M) && isPowerOfTwo(N)) {
  //     cmdBuffer.dispatch((M / LOCAL_X), (N / LOCAL_Y), 1);
  //   } else if (isPowerOfTwo(M) && !isPowerOfTwo(N)) {
  //     cmdBuffer.dispatch((M / LOCAL_X), (N / LOCAL_Y) + 1,
  //                        1); // This one is error prone. # TODO: Fix it
  //   } else if (!isPowerOfTwo(M) && isPowerOfTwo(N)) {
  //     cmdBuffer.dispatch((M / LOCAL_X) + 1, (N / LOCAL_Y), 1);
  //   } else {
  //     cmdBuffer.dispatch((M / LOCAL_X) + 1, (N / LOCAL_Y) + 1, 1);
  //   }
  conditionalDispatch(cmdBuffer, M, N, kernelId);
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

void showResult(vk::DeviceMemory &outBufferMemory, float *c, vk::Device &device,
                int size, int debugFlag) {

  float *outBufferPtr = static_cast<float *>(
      device.mapMemory(outBufferMemory, 0, size * sizeof(float)));
  int count = 0;
  int k = 0;
  for (k = 0; k < size; k++) {
    if (fabs(outBufferPtr[k] - c[k]) > epsilon) {
      count++;
    }
    else if (debugFlag == 0) {
      printf("At position %d Expected: %f , Got %f\n",k,  c[k], outBufferPtr[k]);
    }
  }

  printf("Mismatch check complete. Count: %d , Total %d , error: %lf\n", count,
         k, (float)((float)count / (float)k));
  device.unmapMemory(outBufferMemory);
}

int main(int argc, char *argv[]) {

  string shader_name(argv[1]);

  map<string, int> kernelCodeMap;
  kernelCodeMap["naive_matmul.spv"] = NAIVE;
  kernelCodeMap["coalesced_matmul.spv"] = MEM_COALESCED;
  kernelCodeMap["smem_cached.spv"] = SMEM_CACHE;
  kernelCodeMap["1d-blocktiled.spv"] = TILE_1D;

  uint32_t m = static_cast<uint32_t>(stoi(argv[2]));
  uint32_t k = static_cast<uint32_t>(stoi(argv[3]));
  uint32_t n = static_cast<uint32_t>(stoi(argv[4]));

  int checkResultFlag = 1;
  if (argc >= 6) {
    checkResultFlag = strcmp(argv[5], "-rcheck");
    printf("Check result flag is ON (%d).\n", checkResultFlag);
  }
  int printLog = 1;
  if(argc >= 7){
    printLog = strcmp(argv[6], "-printLog");
    printf("Printing Log flag is ON (%d).\n", printLog);
  }

  float *a = (float *)malloc(m * k * sizeof(float));
  float *b = (float *)malloc(k * n * sizeof(float));
  uint32_t dim[3] = {m, k, n};
  float *c = (float *)malloc(m * n * sizeof(float));

  // Ground truth generation.
  if (!checkResultFlag) {
    matMulGen(a, b, c, m, k, n, true);
    printf("Ground truth generation complete.\n");
  } else {
    matMulGen(a, b, c, m, k, n);
    printf("Data population complete.\n");
  }

  // Create a bunch of stuff that are not related to buffer.
  vk::Instance currentInstance = getVulkanInstance();
  vk::PhysicalDevice physicalDevice = getPhysicalDevice(currentInstance);
  uint32_t computeQueueFamilyIndex = getComputeQueueFamilyIndex(physicalDevice);
  vk::Device device = getDevice(computeQueueFamilyIndex, physicalDevice);

  // Start doing stuff that are related to buffers.
  map<string, vector<pair<vk::Buffer, vk::DeviceMemory>>> buffers =
      getMatmulBuffers(device, physicalDevice, computeQueueFamilyIndex, a, b,
                       dim, m, k, n);
  // TODO: Stop hardcoding stuff and implement the logic.

  printf("Number of input buffers: %lu\n", buffers["input"].size());
  printf("Number of output buffers: %lu\n", buffers["output"].size());

  PipelineComponents pComponents;
  vk::ShaderModule shaderModule = getPipeline(device, pComponents, shader_name);
  auto descriptorObjects =
      getDescriptors(device, pComponents.descriptorSetLayout, buffers["input"],
                     buffers["output"], m, k, n);

  printf("Starting execution.\n");
  auto start = chrono::system_clock::now();
  CommandComponents cmdComponents = getCommandComponents(
      device, computeQueueFamilyIndex, descriptorObjects.second, pComponents, m,
      n, kernelCodeMap[shader_name]);
  execute(cmdComponents, device);
  auto end = chrono::system_clock::now();
  auto elapsed_time = chrono::duration_cast<chrono::microseconds>(end - start);
  cout << "Elapsed time in microseconds: " << elapsed_time.count() << endl;

  if (!checkResultFlag) {
    showResult(buffers["output"][0].second, c, device, m * n, printLog);
  }

  // Cleanup all the bunch of stuffs.
  cleanup(device, currentInstance, buffers, pComponents, shaderModule,
          descriptorObjects, cmdComponents);

  free(a);
  free(b);
  free(c);
  return 0;
}
