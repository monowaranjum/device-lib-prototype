#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>
#include <list>
#include <vector>
#include <vulkan/vulkan.hpp>

using namespace std;

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

vk::Device getDevice(uint32_t computeQueueFamilyIndex, vk::PhysicalDevice& physicalDevice){
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

void cleanup(vk::Device& device, vk::Instance& instance){
  printf("Starting cleanup.\n");
  device.destroy();
  instance.destroy();
}



int main(int argc, char const *argv[]) {
  vk::Instance currentInstance = getVulkanInstance();
  vk::PhysicalDevice physicalDevice = getPhysicalDevice(currentInstance);
  uint32_t computeQueueFamilyIndex = getComputeQueueFamilyIndex(physicalDevice);
  vk::Device device = getDevice(computeQueueFamilyIndex, physicalDevice);

  cleanup(device, currentInstance);
  return 0;
}
