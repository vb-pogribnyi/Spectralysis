#include "VulkanCommon.h"


static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
	VkDebugUtilsMessageSeverityFlagBitsEXT           messageSeverity,
	VkDebugUtilsMessageTypeFlagsEXT                  messageTypes,
	const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
	void* pUserData
) {
	std::cout << "VALIDATION:" << std::endl << pCallbackData->pMessage << std::endl;
	return VK_FALSE;
}


VulkanContext setupContext(std::vector<const char*>& extensions)
{
	VulkanContext context;
	// Check validation layers presence
	const char* validationLayer = "VK_LAYER_KHRONOS_validation";
	uint32_t layerCount = 0;
	if (vkEnumerateInstanceLayerProperties(&layerCount, 0) != VK_SUCCESS)
		throw std::runtime_error("Cannot count layer properties");
	std::vector<VkLayerProperties> availableLayers(layerCount);
	if (vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data()) != VK_SUCCESS)
		throw std::runtime_error("Cannot enumerate layer properties");
	bool layerFound = false;
	for (const auto& layer : availableLayers) {
		if (strcmp(validationLayer, layer.layerName) == 0) {
			layerFound = true;
			break;
		}
	}
	if (!layerFound) throw std::runtime_error("Validation layer is not found");
	extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

	std::cout << "Extensions required:" << std::endl;
	for (auto ext : extensions) std::cout << ext << std::endl;
	std::cout << std::endl;

	// Create the instance
	VkApplicationInfo appInfo = {
		.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
		.pNext = 0,
		.pApplicationName = "Spectralysis",
		.applicationVersion = VK_MAKE_VERSION(0, 0, 1),
		.pEngineName = "No Engine",
		.engineVersion = VK_MAKE_VERSION(0, 0, 1),
		.apiVersion = VK_API_VERSION_1_0,
	};

	VkInstanceCreateInfo instanceCI = {
		.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
		.pNext = 0,
		.flags = 0,
		.pApplicationInfo = &appInfo,
		.enabledLayerCount = 1,
		.ppEnabledLayerNames = &validationLayer,
		.enabledExtensionCount = (uint32_t)extensions.size(),
		.ppEnabledExtensionNames = extensions.data()
	};

	if (vkCreateInstance(&instanceCI, 0, &context.instance) != VK_SUCCESS)
		throw std::runtime_error("Cannot create instance");

	// Setup debug
	VkDebugUtilsMessengerCreateInfoEXT messencerCI = {
		.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
		.pNext = 0,
		.flags = 0,
		.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT |
			VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
			VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT,
		.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
			VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
			VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
		.pfnUserCallback = debugCallback,
		.pUserData = 0
	};

	auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
		context.instance,
		"vkCreateDebugUtilsMessengerEXT"
	);
	if (!func || func(context.instance, &messencerCI, 0, &context.messenger) != VK_SUCCESS)
		throw std::runtime_error("Cannot create debug messenger");

	uint32_t cnt = 0;
	vkEnumeratePhysicalDevices(context.instance, &cnt, 0);
	std::vector<VkPhysicalDevice> physicalDevices(cnt);
	vkEnumeratePhysicalDevices(context.instance, &cnt, physicalDevices.data());
	int idx = 0;
	int bestComputeFamilies = 0;
	for (VkPhysicalDevice physicalDevice : physicalDevices) {
		VkPhysicalDeviceProperties deviceProperties;
		vkGetPhysicalDeviceProperties(physicalDevice, &deviceProperties);
		vkEnumerateDeviceExtensionProperties(physicalDevice, 0, &cnt, 0);
		std::vector<VkExtensionProperties> extensions(cnt);
		vkEnumerateDeviceExtensionProperties(physicalDevice, 0, &cnt, extensions.data());

		// Select the queue family for graphics/presentation and compute
		uint32_t qFamilyCount = 0;
		int computeFamilies = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &qFamilyCount, 0);
		std::vector<VkQueueFamilyProperties> qFamilyProperties(qFamilyCount);
		vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &qFamilyCount, qFamilyProperties.data());
		for (uint32_t i = 0; i < qFamilyCount; i++) {
			bool supportsCompute = qFamilyProperties[i].queueFlags & VK_QUEUE_COMPUTE_BIT;

			if (supportsCompute && qFamilyProperties[i].queueCount) {
				computeFamilies++;
				std::cout << "Queue family " << i << " is ok, " << qFamilyProperties[i].queueCount << " queues." << std::endl;
			}
		}

		std::cout << "Device: " << deviceProperties.deviceName << std::endl;
		if (computeFamilies > bestComputeFamilies) {
			context.physicalDevice = physicalDevice;
			//context.minUniformBufferOffset = deviceProperties.limits.minUniformBufferOffsetAlignment;
			std::cout << "Compute queues: " << computeFamilies << ". Selecting" << std::endl << std::endl;
			bestComputeFamilies = computeFamilies;

			if (bestComputeFamilies >= 3) break;
		}
	}

	// Select queue families for the device
	uint32_t qFamilyCount = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(context.physicalDevice, &qFamilyCount, 0);
	std::vector<VkQueueFamilyProperties> qFamilyProperties(qFamilyCount);
	vkGetPhysicalDeviceQueueFamilyProperties(context.physicalDevice, &qFamilyCount, qFamilyProperties.data());
	std::list<int> queueIdxs;
	for (uint32_t i = 0; i < qFamilyCount; i++) {
		bool supportsCompute = qFamilyProperties[i].queueFlags & VK_QUEUE_COMPUTE_BIT;
		bool supportsGraphics = qFamilyProperties[i].queueFlags & VK_QUEUE_GRAPHICS_BIT;

		if (supportsCompute && qFamilyProperties[i].queueCount) {
			// Dedicated compute queues are appended to the front
			if (supportsGraphics) queueIdxs.push_back(i);
			else queueIdxs.push_front(i);
		}
	}
	context.graphicsFamilyIdx = queueIdxs.back();
	context.sdftFamilyIdx = queueIdxs.front();
	int transferQueuesCnt = 0;
	for (uint32_t i = 0; i < qFamilyCount; i++) {
		bool supportsTransfer = qFamilyProperties[i].queueFlags & VK_QUEUE_TRANSFER_BIT;

		if (supportsTransfer && qFamilyProperties[i].queueCount) {
			context.transferFamilyIdx = i;
			transferQueuesCnt = qFamilyProperties[i].queueCount;
			if (std::find(queueIdxs.begin(), queueIdxs.end(), i) == queueIdxs.end()) break;
		}
	}
	queueIdxs.push_back(context.transferFamilyIdx);

	// Create logical device
	std::vector<float> priorities = { 0.5, 0.5, 0.5, 0.5 };
	std::vector<VkDeviceQueueCreateInfo> queueFamilyCIs;
	for (int queueIdx : queueIdxs) {
		VkDeviceQueueCreateInfo queueCI = {
			.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
			.pNext = 0,
			.flags = 0,
			.queueFamilyIndex = (uint32_t)queueIdx,
			.queueCount = 4,
			.pQueuePriorities = priorities.data()
		};
		queueFamilyCIs.push_back(queueCI);
	}
	queueFamilyCIs.back().queueCount = 1;
	VkPhysicalDeviceFeatures deviceFeatures = {};
	std::vector<const char*> deviceExtensions = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };
	VkDeviceCreateInfo deviceCI = {
		.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
		.pNext = 0,
		.flags = 0,
		.queueCreateInfoCount = (uint32_t)queueFamilyCIs.size(),
		.pQueueCreateInfos = queueFamilyCIs.data(),
		.enabledLayerCount = 0,
		.ppEnabledLayerNames = 0,
		.enabledExtensionCount = (uint32_t)deviceExtensions.size(),
		.ppEnabledExtensionNames = deviceExtensions.data(),
		.pEnabledFeatures = &deviceFeatures
	};

	if (vkCreateDevice(context.physicalDevice, &deviceCI, 0, &context.device) != VK_SUCCESS)
		throw std::runtime_error("Cannot create logical device");

	int specWidth = 1024 * 16;
	int specHeight = 1024;
	int maskWidth = 1024 * 16;
	int maskHeight = 1024;
	context.specWidth = specWidth;
	context.specHeight = specHeight;
	context.maskWidth = maskWidth;
	context.maskHeight = maskHeight;
	context.specgramRaw = createStorageImage(context, specWidth, specHeight);
	context.specgramFilt = createStorageImage(context, specWidth, specHeight);
	//context.mask = createStorageImage(context, maskWidth, maskHeight);

	return context;
}

void destroyContext(VulkanContext& context)
{
	vkDestroyImageView(context.device, context.specgramFilt.image.view, 0);
	vkDestroyImage(context.device, context.specgramFilt.image.data, 0);
	vkFreeMemory(context.device, context.specgramFilt.memory, 0);
	vkDestroyImageView(context.device, context.specgramRaw.image.view, 0);
	vkDestroyImage(context.device, context.specgramRaw.image.data, 0);
	vkFreeMemory(context.device, context.specgramRaw.memory, 0);


	auto destroyDebug = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
		context.instance,
		"vkDestroyDebugUtilsMessengerEXT"
	);
	destroyDebug(context.instance, context.messenger, 0);
	vkDestroyDevice(context.device, 0);
	vkDestroyInstance(context.instance, 0);
}

void createImageView(VkDevice device, VkImage image, VkFormat format, VkImageAspectFlags aspectFlags, VkImageView* imageView)
{
	VkImageViewCreateInfo imageViewCI = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		.pNext = 0,
		.flags = 0,
		.image = image,
		.viewType = VK_IMAGE_VIEW_TYPE_2D,
		.format = format,
		.components = {
			.r = VK_COMPONENT_SWIZZLE_IDENTITY,
			.g = VK_COMPONENT_SWIZZLE_IDENTITY,
			.b = VK_COMPONENT_SWIZZLE_IDENTITY,
			.a = VK_COMPONENT_SWIZZLE_IDENTITY
			},
		.subresourceRange = {
			.aspectMask = aspectFlags,
			.baseMipLevel = 0,
			.levelCount = 1,
			.baseArrayLayer = 0,
			.layerCount = 1
		}
	};
	if (vkCreateImageView(device, &imageViewCI, 0, imageView) != VK_SUCCESS)
		throw std::runtime_error("Cannot create image view");
}

Shader getShaderModule(VkDevice device, std::string filename, VkShaderStageFlagBits stage)
{
	// Read the file
	std::ifstream file(filename, std::ios::binary | std::ios::ate);
	if (!file.is_open()) throw std::runtime_error("Cannot find " + filename);
	std::vector<char> buffer(file.tellg());
	file.seekg(0);
	file.read(buffer.data(), buffer.size());
	file.close();
	std::cout << "File: " + filename << " size: " << buffer.size() << std::endl;

	// Create shader module
	VkShaderModule shaderModule;
	VkShaderModuleCreateInfo shaderModuleCI = {
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.pNext = 0,
		.flags = 0,
		.codeSize = buffer.size(),
		.pCode = reinterpret_cast<const uint32_t*>(buffer.data())
	};
	if (vkCreateShaderModule(device, &shaderModuleCI, 0, &shaderModule) != VK_SUCCESS)
		throw std::runtime_error("Cannot create shader module");

	VkPipelineShaderStageCreateInfo shaderStageCI = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
		.pNext = 0,
		.flags = 0,
		.stage = stage,
		.module = shaderModule,
		.pName = "main",
		.pSpecializationInfo = 0
	};
	Shader result = {
		.shaderModule = shaderModule,
		.stageCI = shaderStageCI
	};
	return result;
}

std::pair<VkBuffer, VkDeviceMemory> createBuffer(VkDevice device, VkPhysicalDevice physicalDevice, std::vector<uint32_t> queueFamilyIndices, 
	VkDeviceSize size, VkMemoryPropertyFlags properties, VkBufferUsageFlags usage)
{
	VkBuffer buffer;
	VkDeviceMemory memory;

	VkSharingMode sharingMode = queueFamilyIndices.size() > 1 ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE;
	VkBufferCreateInfo bufferCI = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.pNext = 0,
		.flags = 0,
		.size = size,
		.usage = usage,
		.sharingMode = sharingMode,
		.queueFamilyIndexCount = (uint32_t)queueFamilyIndices.size(),
		.pQueueFamilyIndices = queueFamilyIndices.data()
	};

	if (vkCreateBuffer(device, &bufferCI, 0, &buffer) != VK_SUCCESS)
		throw std::runtime_error("Cannot create buffer");

	VkMemoryRequirements memoryRequirements;
	VkPhysicalDeviceMemoryProperties memoryProperties;
	vkGetBufferMemoryRequirements(device, buffer, &memoryRequirements);
	vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);
	uint32_t memoryTypeIdx = (uint32_t)-1;
	for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; i++) {
		if ((memoryRequirements.memoryTypeBits & (i << i)) &&
			((memoryProperties.memoryTypes[i].propertyFlags & properties) == properties)) {
			memoryTypeIdx = i;
			break;
		}
	}
	std::cout << "Buffer memory " << (memoryTypeIdx == (uint32_t)-1 ? "NOT supported" : "supported") << std::endl;
	if (memoryTypeIdx == (uint32_t)-1)
		throw std::runtime_error("Memory type is not suppotred");

	VkMemoryAllocateInfo memoryAI = {
		.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		.pNext = 0,
		.allocationSize = memoryRequirements.size,
		.memoryTypeIndex = memoryTypeIdx
	};

	if (vkAllocateMemory(device, &memoryAI, 0, &memory) != VK_SUCCESS)
		throw std::runtime_error("Cannot allocate memory for buffer");
	vkBindBufferMemory(device, buffer, memory, 0);

	return { buffer, memory };
}

std::pair<VkImage, VkDeviceMemory> createImage(VkDevice device, VkPhysicalDevice physicalDevice, uint32_t queueFamilyIndex,
	uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, 
	VkImageUsageFlags usage, VkMemoryPropertyFlags properties)
{
	VkImage image;
	VkDeviceMemory memory;
	VkImageCreateInfo imageCI = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		.pNext = 0,
		.flags = 0,
		.imageType = VK_IMAGE_TYPE_2D,
		.format = format,
		.extent = {.width = width, .height = height, .depth = 1},
		.mipLevels = 1,
		.arrayLayers = 1,
		.samples = VK_SAMPLE_COUNT_1_BIT,
		.tiling = tiling,
		.usage = usage,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.queueFamilyIndexCount = 1,
		.pQueueFamilyIndices = &queueFamilyIndex,
		.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED
	};
	if (vkCreateImage(device, &imageCI, 0, &image) != VK_SUCCESS)
		throw std::runtime_error("Cannot create image");
	VkMemoryRequirements requirements;
	vkGetImageMemoryRequirements(device, image, &requirements);

	VkPhysicalDeviceMemoryProperties memoryProperties;
	vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);
	uint32_t memoryTypeIdx = (uint32_t)-1;
	for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; i++) {
		if ((requirements.memoryTypeBits & (i << i)) &&
			((memoryProperties.memoryTypes[i].propertyFlags & properties) == properties)) {
			memoryTypeIdx = i;
			break;
		}
	}
	VkMemoryAllocateInfo memoryAI = {
		.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		.pNext = 0,
		.allocationSize = requirements.size,
		.memoryTypeIndex = memoryTypeIdx
	};
	if (vkAllocateMemory(device, &memoryAI, 0, &memory) != VK_SUCCESS)
		throw std::runtime_error("Cannot allocate image memory");
	vkBindImageMemory(device, image, memory, 0);

	return { image, memory };
}

void createDescriptorSet(VkDevice device, std::vector<Binding> bindingsIn, std::vector<VkDescriptorSet>& descriptorSets, 
	VkDescriptorPool *descriptorPool, VkDescriptorSetLayout* descriptorSetLayout)
{
	uint32_t bindingIdx = 0;
	if (*descriptorSetLayout == 0) {
		// Create descriptor set layout
		std::vector<VkDescriptorSetLayoutBinding> bindings;
		for (Binding b : bindingsIn) {
			bindings.push_back({
				.binding = bindingIdx++,
				.descriptorType = b.type,
				.descriptorCount = 1,
				.stageFlags = b.stageFlags,
				.pImmutableSamplers = 0
				});
		};

		VkDescriptorSetLayoutCreateInfo layoutCI = {
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
			.pNext = 0,
			.flags = 0,
			.bindingCount = (uint32_t)bindings.size(),
			.pBindings = bindings.data()
		};
		if (vkCreateDescriptorSetLayout(device, &layoutCI, 0, descriptorSetLayout) != VK_SUCCESS)
			throw std::runtime_error("Cannot create descriptor set layout");
	}
	// Create descriptor pool
	std::vector<VkDescriptorPoolSize> poolSizes;
	for (Binding b : bindingsIn) {
		poolSizes.push_back({
			.type = b.type,
			.descriptorCount = 1
			});
	};

	VkDescriptorPoolCreateInfo poolCI = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.pNext = 0,
		.flags = 0,
		.maxSets = (uint32_t)descriptorSets.size(),
		.poolSizeCount = (uint32_t)poolSizes.size(),
		.pPoolSizes = poolSizes.data()
	};
	if (vkCreateDescriptorPool(device, &poolCI, 0, descriptorPool) != VK_SUCCESS)
		throw std::runtime_error("Cannot create descriptor pool");

	// Create descriptor set
	VkDescriptorSetAllocateInfo setAI = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.pNext = 0,
		.descriptorPool = *descriptorPool,
		.descriptorSetCount = 1,
		.pSetLayouts = descriptorSetLayout
	};
	int i = 0;
	for (VkDescriptorSet& descriptorSet : descriptorSets) {
		if (vkAllocateDescriptorSets(device, &setAI, &descriptorSet) != VK_SUCCESS)
			throw std::runtime_error("Cannot allocate descriptor set");

		std::vector< VkWriteDescriptorSet> bufferWrites;
		bindingIdx = 0;
		std::vector<VkDescriptorBufferInfo> bufferInfos;
		std::vector<VkDescriptorImageInfo> imageInfos;
		bufferInfos.reserve(4);  // Reserve some values not to mess up the pointers
		imageInfos.reserve(4);
		for (Binding b : bindingsIn) {
			VkWriteDescriptorSet writeDescriptorSet = {
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.pNext = 0,
				.dstSet = descriptorSet,
				.dstBinding = bindingIdx++,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = b.type,
				.pImageInfo = 0,
				.pBufferInfo = 0
			};
			if (b.type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) {
				
				bufferInfos.push_back({
					.buffer = b.buffer.data,
					.offset = 0,
					.range = VK_WHOLE_SIZE
					});
				writeDescriptorSet.pBufferInfo = &(bufferInfos[bufferInfos.size() - 1]);
			}
			else if (b.type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) {
				imageInfos.push_back({
					.sampler = b.image.sampler,
					.imageView = b.image.view,
					.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
					});
				writeDescriptorSet.pImageInfo = &(imageInfos[imageInfos.size() - 1]);
			}
			else if (b.type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE) {
				imageInfos.push_back({
					.sampler = b.image.sampler,
					.imageView = b.image.view,
					.imageLayout = VK_IMAGE_LAYOUT_GENERAL
					});
				writeDescriptorSet.pImageInfo = &(imageInfos[imageInfos.size() - 1]);
			}
			bufferWrites.push_back(writeDescriptorSet);
		};
		vkUpdateDescriptorSets(device, (uint32_t)bufferWrites.size(), bufferWrites.data(), 0, 0);
		i++;
	}
}

void createDescriptorSet(VkDevice device, std::vector<Binding> bindingsIn, VkDescriptorSet& descriptorSet, VkDescriptorPool* descriptorPool, VkDescriptorSetLayout* setLayout)
{
	std::vector<VkDescriptorSet> dset = { descriptorSet };
	createDescriptorSet(device, bindingsIn, dset, descriptorPool, setLayout);
	descriptorSet = dset[0];
}

Binding createStorageImage(VulkanContext context, uint32_t width, uint32_t height)
{
	Binding result;
	std::pair<VkImage, VkDeviceMemory> image = createImage(context.device, context.physicalDevice, context.graphicsFamilyIdx,
		width, height, VK_FORMAT_R8G8B8A8_UNORM,
		VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
		VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	result.image.data = image.first;
	result.memory = image.second;
	createImageView(context.device, result.image.data, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT, &result.image.view);
	result.image.sampler = VK_NULL_HANDLE;
	result.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	result.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

	return result;
}
