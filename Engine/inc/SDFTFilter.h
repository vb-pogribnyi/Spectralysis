#pragma once
#include <iostream>
#include <vulkan/vulkan.h>
#include <vector>
#include <list>
#include <glm.hpp>
#include "VulkanCommon.h"

//struct ShaderImage {
//	VkImage image;
//	VkDeviceMemory memory;
//	VkImageView view;
//};
#define SUMMATION_SIZE 32
#define SUMMATION_WIDTH 32


// Mimics SDFTFilterState
struct SDFTProps {
	int spec_height;
	int segment_width;
	int signal_length;
	int max_signal_size;
	int hop;

	int hostMaskHeight;
	int hostMaskWidth;
};

struct SDFTState {
	int stageStride;
	int hop;
	int isWriteImg;
	int isInverse;
	int isShift;
	int specHeight;
};

struct LinearResize {
	int src_rows;
	int src_cols;
	int dst_rows;
	int dst_cols;
};

struct FIRState {
	int signal_len;
	int hop;
	int spec_height;
};

struct SUMState {
	int stride;
	int out_stride;
	int spec_height;
	int signal_len;
};

// Returned by update(int chunk)
// Returns raw signal spectrogram, filtered signal and its spectrogram
struct ChunkUpdate {
	VkDescriptorSet specRaw;
	VkDescriptorSet mask;
	VkDescriptorSet filter;
	VkDescriptorSet filtered;
	VkDescriptorSet specFilt;
};

struct Chunk {
	// Buffers
	std::pair<VkBuffer, VkDeviceMemory> sdftTemp1Buffer;
	std::pair<VkBuffer, VkDeviceMemory> sdftTemp2Buffer;
	std::pair<VkBuffer, VkDeviceMemory> filterTemp1Buffer;
	std::pair<VkBuffer, VkDeviceMemory> filterTemp2Buffer;
	std::pair<VkBuffer, VkDeviceMemory> specRawBuffer;
	std::pair<VkBuffer, VkDeviceMemory> specFiltBuffer;
	std::pair<VkBuffer, VkDeviceMemory> maskBuffer;
	std::pair<VkBuffer, VkDeviceMemory> maskHostBuffer;
	std::pair<VkBuffer, VkDeviceMemory> signalRawBuffer;
	std::pair<VkBuffer, VkDeviceMemory> signalRawExtBuffer;
	std::pair<VkBuffer, VkDeviceMemory> signalFiltBuffer;
	std::pair<VkBuffer, VkDeviceMemory> filtersBuffer;
	// Transfer
	std::pair<VkBuffer, VkDeviceMemory> uploadBuffer;
	std::pair<VkBuffer, VkDeviceMemory> bufferSignal;
	std::pair<VkBuffer, VkDeviceMemory> bufferSpec;

	// Bindings
	Binding sdftTemp1Binding;
	Binding sdftTemp2Binding;
	Binding filterTemp1Binding;
	Binding filterTemp2Binding;
	Binding filterBinding;
	Binding specRawBinding;
	Binding specFiltBinding;
	Binding maskBinding;
	Binding maskHostBinding;
	Binding signalRawBinding;
	Binding signalRawExtBinding;
	Binding signalFiltBinding;
	Binding filtersBinding;

	// Descriptor sets
	std::pair <VkDescriptorSet, VkDescriptorPool> srcDSet;
	std::pair <VkDescriptorSet, VkDescriptorPool> srcDSetExt;
	std::pair <VkDescriptorSet, VkDescriptorPool> temp1DSet;
	std::pair <VkDescriptorSet, VkDescriptorPool> temp2DSet;
	std::pair <VkDescriptorSet, VkDescriptorPool> filterTemp1DSet;
	std::pair <VkDescriptorSet, VkDescriptorPool> filterTemp2DSet;
	std::pair <VkDescriptorSet, VkDescriptorPool> dstSDFTDSet;
	std::pair <VkDescriptorSet, VkDescriptorPool> dstSDFTFiltDSet;
	std::pair <VkDescriptorSet, VkDescriptorPool> maskDSet;
	std::pair <VkDescriptorSet, VkDescriptorPool> filterDSet;
	std::pair <VkDescriptorSet, VkDescriptorPool> maskHostDSet;
	std::pair <VkDescriptorSet, VkDescriptorPool> filteredDSet;

	// Commands - SDFT
	VkPipelineStageFlags waitStagesCompute;
	VkPipelineStageFlags waitStagesTransfer;
	VkCommandPool cmdPoolTransfer;
	VkCommandPool cmdPoolCompute;
	VkCommandBuffer cmdBuffUploadSDFT;
	VkCommandBuffer cmdBuffDowndloadSDFT;
	VkCommandBuffer cmdBuffProcessSDFT;
	VkSemaphore smphUploadedSDFT;
	VkSemaphore smphProcessedSDFT;
	VkSubmitInfo submitInfoSDFTUpload;
	VkSubmitInfo submitInfoSDFTDownload;
	VkSubmitInfo submitInfoSDFTProcess;
	VkFence fenceSDFT;

	// Commands - Filtering
	VkCommandBuffer cmdBuffMaskRead;
	VkCommandBuffer cmdBuffMaskSDFT;
	VkCommandBuffer cmdBuffFilter;
	VkSemaphore smphMaskRead;
	VkSemaphore smphFiltersRdy;
	VkSubmitInfo submitInfoMaskRead;
	VkSubmitInfo submitInfoFilter;
	VkSubmitInfo submitInfoMaskSDFT;
	VkFence fenceFilter;
};

class SDFTFilter
{
public:
	SDFTFilter(SDFTProps props);
	SDFTFilter(VulkanContext context, SDFTProps props);
	~SDFTFilter();
	Chunk chunk;
	void initChunk(Chunk& chunk);
	void recordChunk(Chunk& chunk);
	void destroyChunk(Chunk& chunk);


	/// <summary>
	/// Reads mask from the data array, calculates filters, applies them to the signal,
	/// calculates the spectrogram of the filtered signal
	/// </summary>
	/// <param name="chunk">Index of a chunk to update</param>
	/// <param name="data">Pointer to the mask data, stored in ARGB format. 4-bytes int for every pixel</param>
	/// <returns></returns>
	void update(Chunk& chunk, int* data);
	void update(int* mask, const std::vector<float>& signalIn, std::vector<float>& signalOut);
	void calcSDFT(const std::vector<float>& signalIn, std::vector<float>& specOut);
	int getSpecWidth();

private:
	SDFTProps props;
	VulkanContext context;

	// TRANSFER
	VkBufferMemoryBarrier toSDFTBufferBarrier;
	VkCommandPool transferCommandPool;
	VkCommandBuffer transferCommandBuffer;


	// DEVICE RELATED (move to context?)
	VkQueue transferQueue;
	VkQueue readMaskQueue;
	VkQueue filterQueue;
	// sdft performing queues
	VkQueue rawSDFTQueue;
	VkQueue filteredSDFTQueue;
	VkQueue createFilterQueue;


	// Buffers
	VkDescriptorSetLayout sdftDescriptorSetLayout;
	VkPipelineLayout sdftPipelineLayout;
	VkPipelineLayout readPipelineLayout;
	VkPipelineLayout filterPipelineLayout;
	VkPipelineLayout sumPipelineLayout;

	// Pipelines
	VkPipeline sdftPipeline;
	VkPipeline sdftImgPipeline;
	VkPipeline filterPipeline;
	VkPipeline sumPipeline;
	VkPipeline readMaskPipeline;
	void createDescriptorSets();
	void createStorageBuffer(VkDeviceSize size, std::pair<VkBuffer, VkDeviceMemory>& buffer, Binding& binding, bool is_host_visible=false);
	void recordSDFT(VkCommandBuffer commandBuffer, VkDescriptorSet src, VkDescriptorSet dst, Chunk chunk, VkBuffer inBuffer, VkBuffer outBuffer, bool isInverse, bool isShift);
};

