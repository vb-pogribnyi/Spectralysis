#pragma once
#include <vector>
#include <list>
#include <string>
#include <iostream>
#include <fstream>
#include <vulkan/vulkan.h>


struct Shader {
	VkShaderModule shaderModule;
	VkPipelineShaderStageCreateInfo stageCI;
};

struct Binding {
	VkDescriptorType type;
	VkShaderStageFlags stageFlags;
	union {
		struct {
			VkBuffer data;
			int step;
			int size;
		} buffer;
		struct {
			VkImage data;
			VkImageView view;
			VkSampler sampler;
		} image;
	};
	VkDeviceMemory memory;
};

struct PipelineInfo {
	VkDescriptorSetLayout descriptorSetLayout;
	VkDescriptorPool descriptorPool;
	VkPipelineLayout pipelineLayout;
	VkPipeline pipeline;
};

struct VulkanContext {
	VkDebugUtilsMessengerEXT messenger;
	VkPhysicalDevice physicalDevice;
	VkDevice device;
	VkInstance instance;
	//int minUniformBufferOffset;
	Binding specgramRaw;
	Binding specgramFilt;
	//Binding mask;
	int graphicsFamilyIdx;
	int sdftFamilyIdx;
	int transferFamilyIdx;

	int specWidth;
	int specHeight;
	int maskWidth;
	int maskHeight;
};

VulkanContext setupContext(std::vector<const char*>& extensions);
void destroyContext(VulkanContext& context);
void createImageView(VkDevice device, VkImage image, VkFormat format, VkImageAspectFlags aspectFlags, VkImageView* imageView);
Shader getShaderModule(VkDevice device, std::string filename, VkShaderStageFlagBits stage);
std::pair<VkBuffer, VkDeviceMemory>
createBuffer(VkDevice device, VkPhysicalDevice physicalDevice, std::vector<uint32_t> queueFamilyIndices, VkDeviceSize size, VkMemoryPropertyFlags properties, VkBufferUsageFlags usage);
std::pair<VkImage, VkDeviceMemory>
createImage(VkDevice device, VkPhysicalDevice physicalDevice, uint32_t queueFamilyIndex, 
	uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, 
	VkImageUsageFlags usage, VkMemoryPropertyFlags properties);
void createDescriptorSet(VkDevice device, std::vector<Binding> bindingsIn, std::vector<VkDescriptorSet>& descriptorSets, VkDescriptorPool *descriptorPool, VkDescriptorSetLayout* setLayout = 0);
void createDescriptorSet(VkDevice device, std::vector<Binding> bindingsIn, VkDescriptorSet& descriptorSet, VkDescriptorPool* descriptorPool, VkDescriptorSetLayout* setLayout = 0);
Shader getShaderModule(VkDevice device, std::string filename, VkShaderStageFlagBits stage);
Binding createStorageImage(VulkanContext context, uint32_t width, uint32_t height);

