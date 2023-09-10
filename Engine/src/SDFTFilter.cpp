#include "SDFTFilter.h"
#include <chrono>

// #define PROFILING

SDFTFilter::SDFTFilter(SDFTProps props) : props(props)
{
	std::vector<const char*> extensions = {};
	context = setupContext(extensions);
	this->SDFTFilter::SDFTFilter(context, props);
}

SDFTFilter::SDFTFilter(VulkanContext context, SDFTProps props) : context(context), props(props)
{
	vkGetDeviceQueue(context.device, context.sdftFamilyIdx, 0, &readMaskQueue);
	vkGetDeviceQueue(context.device, context.sdftFamilyIdx, 1, &rawSDFTQueue);
	vkGetDeviceQueue(context.device, context.sdftFamilyIdx, 2, &filteredSDFTQueue);
	vkGetDeviceQueue(context.device, context.sdftFamilyIdx, 3, &createFilterQueue);
	vkGetDeviceQueue(context.device, context.graphicsFamilyIdx, 2, &filterQueue);
	vkGetDeviceQueue(context.device, context.transferFamilyIdx, 0, &transferQueue);

	sdftDescriptorSetLayout = 0;
	initChunk(chunk);
	createDescriptorSets();
	recordChunk(chunk);

}

SDFTFilter::~SDFTFilter()
{
	vkDestroyPipeline(context.device, filterPipeline, 0);
	vkDestroyPipeline(context.device, readMaskPipeline, 0);
	vkDestroyPipeline(context.device, sdftPipeline, 0);
	vkDestroyPipeline(context.device, sumPipeline, 0);
	vkDestroyPipelineLayout(context.device, filterPipelineLayout, 0);
	vkDestroyPipelineLayout(context.device, readPipelineLayout, 0);
	vkDestroyPipelineLayout(context.device, sdftPipelineLayout, 0); 
	vkDestroyPipelineLayout(context.device, sumPipelineLayout, 0);

	destroyChunk(chunk);

	vkDestroyDescriptorSetLayout(context.device, sdftDescriptorSetLayout, 0);
	vkDestroyCommandPool(context.device, transferCommandPool, 0);
}

void SDFTFilter::initChunk(Chunk& chunk)
{
	// BUFFERS
	VkDeviceSize tempSize = sizeof(glm::vec2) * props.spec_height * props.segment_width;
	createStorageBuffer(tempSize, chunk.sdftTemp1Buffer, chunk.sdftTemp1Binding);
	createStorageBuffer(tempSize, chunk.sdftTemp2Buffer, chunk.sdftTemp2Binding);

	VkDeviceSize size = sizeof(glm::vec2) * (props.spec_height + props.hop * props.segment_width);					// Chunk signal size
	createStorageBuffer(size, chunk.signalRawBuffer, chunk.signalRawBinding);
	createStorageBuffer(size + sizeof(glm::vec2) * props.spec_height, chunk.signalRawExtBuffer, chunk.signalRawExtBinding);
	createStorageBuffer(size, chunk.signalFiltBuffer, chunk.signalFiltBinding);
	chunk.uploadBuffer = createBuffer(context.device, context.physicalDevice, { (uint32_t)context.transferFamilyIdx }, 
		size + sizeof(glm::vec2) * props.spec_height,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
	chunk.bufferSignal = createBuffer(context.device, context.physicalDevice, { (uint32_t)context.transferFamilyIdx }, size,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		VK_BUFFER_USAGE_TRANSFER_DST_BIT);

	createStorageBuffer(size * props.spec_height / SUMMATION_SIZE, chunk.filterTemp1Buffer, chunk.filterTemp1Binding);
	createStorageBuffer(size * props.spec_height / SUMMATION_SIZE, chunk.filterTemp2Buffer, chunk.filterTemp2Binding);

	VkDeviceSize specSize = sizeof(glm::vec2) * props.segment_width * props.spec_height;
	createStorageBuffer(specSize, chunk.specRawBuffer, chunk.specRawBinding);
	createStorageBuffer(specSize, chunk.specFiltBuffer, chunk.specFiltBinding);
	createStorageBuffer(specSize, chunk.maskBuffer, chunk.maskBinding);
	createStorageBuffer(specSize, chunk.filtersBuffer, chunk.filtersBinding);
	chunk.bufferSpec = createBuffer(context.device, context.physicalDevice, { (uint32_t)context.transferFamilyIdx }, specSize,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		VK_BUFFER_USAGE_TRANSFER_DST_BIT);

	VkDeviceSize hostSpecSize = sizeof(glm::vec2) * props.hostMaskWidth * props.hostMaskHeight;
	createStorageBuffer(hostSpecSize, chunk.maskHostBuffer, chunk.maskHostBinding, true);

	// DESCRIPTOR SETS
	int pieceWidth = props.hop * props.segment_width + props.spec_height;
	createDescriptorSet(context.device, { chunk.signalRawBinding },
		chunk.srcDSet.first, & chunk.srcDSet.second, &sdftDescriptorSetLayout);
	createDescriptorSet(context.device, { chunk.signalRawExtBinding },
		chunk.srcDSetExt.first, & chunk.srcDSetExt.second, &sdftDescriptorSetLayout);
	createDescriptorSet(context.device, { chunk.signalFiltBinding },
		chunk.filteredDSet.first, & chunk.filteredDSet.second, &sdftDescriptorSetLayout);

	createDescriptorSet(context.device, { chunk.sdftTemp1Binding },
		chunk.temp1DSet.first, &chunk.temp1DSet.second, &sdftDescriptorSetLayout);
	createDescriptorSet(context.device, { chunk.sdftTemp2Binding },
		chunk.temp2DSet.first, &chunk.temp2DSet.second, &sdftDescriptorSetLayout);
	createDescriptorSet(context.device, { chunk.filterTemp1Binding },
		chunk.filterTemp1DSet.first, &chunk.filterTemp1DSet.second, &sdftDescriptorSetLayout);
	createDescriptorSet(context.device, { chunk.filterTemp2Binding },
		chunk.filterTemp2DSet.first, &chunk.filterTemp2DSet.second, &sdftDescriptorSetLayout);

	createDescriptorSet(context.device, { chunk.specRawBinding },
		chunk.dstSDFTDSet.first, & chunk.dstSDFTDSet.second, &sdftDescriptorSetLayout);
	createDescriptorSet(context.device, { chunk.specFiltBinding },
		chunk.dstSDFTFiltDSet.first, & chunk.dstSDFTFiltDSet.second, &sdftDescriptorSetLayout);
	createDescriptorSet(context.device, { chunk.maskBinding },
		chunk.maskDSet.first, & chunk.maskDSet.second, &sdftDescriptorSetLayout);
	createDescriptorSet(context.device, { chunk.filtersBinding },
		chunk.filterDSet.first, & chunk.filterDSet.second, &sdftDescriptorSetLayout);


	createDescriptorSet(context.device, { chunk.maskHostBinding },
		chunk.maskHostDSet.first, & chunk.maskHostDSet.second, &sdftDescriptorSetLayout);

	// Commands - SDFT
	VkCommandPoolCreateInfo transferCommandPoolCI = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
		.pNext = 0,
		.flags = 0,
		.queueFamilyIndex = (uint32_t)context.transferFamilyIdx
	};
	VkCommandPoolCreateInfo computeCommandPoolCI = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
		.pNext = 0,
		.flags = 0,
		.queueFamilyIndex = (uint32_t)context.sdftFamilyIdx
	};
	if (vkCreateCommandPool(context.device, &transferCommandPoolCI, 0, &chunk.cmdPoolTransfer) != VK_SUCCESS)
		throw std::runtime_error("Cannot create chunk transfer command pool");
	if (vkCreateCommandPool(context.device, &computeCommandPoolCI, 0, &chunk.cmdPoolCompute) != VK_SUCCESS)
		throw std::runtime_error("Cannot create chunk compute command pool");
	VkCommandBufferAllocateInfo transferCommandBufferAI = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.pNext = 0,
		.commandPool = chunk.cmdPoolTransfer,
		.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		.commandBufferCount = 2
	};
	VkCommandBufferAllocateInfo computeCommandBufferAI = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.pNext = 0,
		.commandPool = chunk.cmdPoolCompute,
		.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		.commandBufferCount = 4
	};
	if (vkAllocateCommandBuffers(context.device, &transferCommandBufferAI, &chunk.cmdBuffUploadSDFT) != VK_SUCCESS)
		throw std::runtime_error("Cannot create chunk SDFT upload command buffer");
	if (vkAllocateCommandBuffers(context.device, &transferCommandBufferAI, &chunk.cmdBuffDowndloadSDFT) != VK_SUCCESS)
		throw std::runtime_error("Cannot create chunk SDFT download command buffer");
	if (vkAllocateCommandBuffers(context.device, &computeCommandBufferAI, &chunk.cmdBuffProcessSDFT) != VK_SUCCESS)
		throw std::runtime_error("Cannot create chunk SDFT process command buffer");
	if (vkAllocateCommandBuffers(context.device, &computeCommandBufferAI, &chunk.cmdBuffMaskRead) != VK_SUCCESS)
		throw std::runtime_error("Cannot create chunk Mask Read command buffer");
	if (vkAllocateCommandBuffers(context.device, &computeCommandBufferAI, &chunk.cmdBuffMaskSDFT) != VK_SUCCESS)
		throw std::runtime_error("Cannot create chunk Filters creation command buffer");
	if (vkAllocateCommandBuffers(context.device, &computeCommandBufferAI, &chunk.cmdBuffFilter) != VK_SUCCESS)
		throw std::runtime_error("Cannot create chunk Filters process command buffer");

	VkSemaphoreCreateInfo semaphoreCI = {
		.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
		.pNext = 0,
		.flags = 0
	};
	if (vkCreateSemaphore(context.device, &semaphoreCI, 0, &chunk.smphUploadedSDFT) != VK_SUCCESS)
		throw std::runtime_error("Cannot create chunk SDFT Uploaded semaphore");
	if (vkCreateSemaphore(context.device, &semaphoreCI, 0, &chunk.smphProcessedSDFT) != VK_SUCCESS)
		throw std::runtime_error("Cannot create chunk SDFT Processed semaphore");
	if (vkCreateSemaphore(context.device, &semaphoreCI, 0, &chunk.smphMaskRead) != VK_SUCCESS)
		throw std::runtime_error("Cannot create chunk Mask Read semaphore");
	if (vkCreateSemaphore(context.device, &semaphoreCI, 0, &chunk.smphFiltersRdy) != VK_SUCCESS)
		throw std::runtime_error("Cannot create chunk Filters ready semaphore");

	VkFenceCreateInfo fenceCI = {
		.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
		.pNext = 0,
		.flags = 0 //VK_FENCE_CREATE_SIGNALED_BIT
	};
	if (vkCreateFence(context.device, &fenceCI, 0, &chunk.fenceSDFT) != VK_SUCCESS)
		throw std::runtime_error("Cannot create chunk SDFT Processed fence");
	if (vkCreateFence(context.device, &fenceCI, 0, &chunk.fenceFilter) != VK_SUCCESS)
		throw std::runtime_error("Cannot create chunk SDFT Processed fence");
}

void SDFTFilter::recordChunk(Chunk& chunk)
{
	VkDeviceSize size = sizeof(glm::vec2) * (props.spec_height + props.hop * props.segment_width);
	VkDeviceSize specSize = sizeof(glm::vec2) * props.segment_width * props.spec_height;
	// TODO: Make one BufferBI?
	VkCommandBufferBeginInfo transferBufferBI = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.pNext = 0,
		.flags = 0,
		.pInheritanceInfo = 0
	};
	VkCommandBufferBeginInfo sdftBufferBI = {
	.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
	.pNext = 0,
	.flags = 0,
	.pInheritanceInfo = 0
	};

	vkBeginCommandBuffer(chunk.cmdBuffUploadSDFT, &transferBufferBI);
	VkBufferCopy bufferCopyRegion = {
		.srcOffset = 0,
		.dstOffset = 0,
		.size = size
	};
	vkCmdCopyBuffer(chunk.cmdBuffUploadSDFT, chunk.uploadBuffer.first, chunk.signalRawExtBuffer.first, 1, &bufferCopyRegion);
	vkEndCommandBuffer(chunk.cmdBuffUploadSDFT);

	vkBeginCommandBuffer(chunk.cmdBuffDowndloadSDFT, &transferBufferBI);
	VkBufferCopy specBufferCopyRegion = {
		.srcOffset = 0,
		.dstOffset = 0,
		.size = specSize
	};
	vkCmdCopyBuffer(chunk.cmdBuffDowndloadSDFT, chunk.specFiltBuffer.first, chunk.bufferSpec.first, 1, &specBufferCopyRegion);
	vkEndCommandBuffer(chunk.cmdBuffDowndloadSDFT);

	if (vkBeginCommandBuffer(chunk.cmdBuffProcessSDFT, &sdftBufferBI) != VK_SUCCESS)
		throw std::runtime_error("Cannot begin SDFT Process buffer");
	vkCmdBindPipeline(chunk.cmdBuffProcessSDFT, VK_PIPELINE_BIND_POINT_COMPUTE, sdftPipeline);

	recordSDFT(chunk.cmdBuffProcessSDFT, chunk.srcDSetExt.first, chunk.dstSDFTFiltDSet.first, chunk, chunk.signalRawExtBuffer.first, chunk.specFiltBuffer.first, false, true);

	if (vkEndCommandBuffer(chunk.cmdBuffProcessSDFT) != VK_SUCCESS)
		throw std::runtime_error("Cannot begin command buffer");

	chunk.waitStagesCompute = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
	chunk.waitStagesTransfer = VK_PIPELINE_STAGE_TRANSFER_BIT;
	chunk.submitInfoSDFTUpload = {
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.pNext = 0,
		.waitSemaphoreCount = 0,
		.pWaitSemaphores = 0,
		.pWaitDstStageMask = 0,
		.commandBufferCount = 1,
		.pCommandBuffers = &chunk.cmdBuffUploadSDFT,
		.signalSemaphoreCount = 1,
		.pSignalSemaphores = &chunk.smphUploadedSDFT
	};

	chunk.submitInfoSDFTDownload = {
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.pNext = 0,
		.waitSemaphoreCount = 1,
		.pWaitSemaphores = &chunk.smphProcessedSDFT,
		.pWaitDstStageMask = &chunk.waitStagesTransfer,
		.commandBufferCount = 1,
		.pCommandBuffers = &chunk.cmdBuffDowndloadSDFT,
		.signalSemaphoreCount = 0,
		.pSignalSemaphores = 0
	};

	chunk.submitInfoSDFTProcess = {
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.pNext = 0,
		.waitSemaphoreCount = 1,
		.pWaitSemaphores = &chunk.smphUploadedSDFT,
		.pWaitDstStageMask = &chunk.waitStagesCompute,
		.commandBufferCount = 1,
		.pCommandBuffers = &chunk.cmdBuffProcessSDFT,
		.signalSemaphoreCount = 1,
		.pSignalSemaphores = &chunk.smphProcessedSDFT
	};

	// FILTERING
	FIRState state = {
		.signal_len = props.hop * props.segment_width + props.spec_height,
		.hop = props.hop,
		.spec_height = props.spec_height
	};
	SUMState sumState = {
		.stride = props.spec_height / SUMMATION_SIZE,
		.out_stride = props.spec_height / SUMMATION_SIZE,
		.spec_height = props.spec_height / SUMMATION_SIZE,
		.signal_len = props.spec_height + props.hop * props.segment_width
	};
	VkBufferMemoryBarrier filterBarrier = {
		   .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
		   .pNext = 0,
		   .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
		   .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT,
		   .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		   .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		   .offset = 0,
		   .size = VK_WHOLE_SIZE
	};
	// chunk.filterDSet.first, chunk.srcDSetExt.first, chunk.filteredDSet.first
	int nStages = (int)log2(props.spec_height / SUMMATION_SIZE);
	std::vector< VkDescriptorSet> descriptorsSrcTemp1 = { chunk.filterDSet.first, chunk.srcDSetExt.first, chunk.filterTemp1DSet.first };
	std::vector< VkDescriptorSet> descriptorsTemp1Temp2 = { chunk.filterTemp1DSet.first, chunk.filterTemp2DSet.first };
	std::vector< VkDescriptorSet> descriptorsTemp2Temp1 = { chunk.filterTemp2DSet.first, chunk.filterTemp1DSet.first };
	std::vector< VkDescriptorSet> descriptorsTemp1Dst = { chunk.filterTemp1DSet.first, chunk.filteredDSet.first };
	std::vector< VkDescriptorSet> descriptorsTemp2Dst = { chunk.filterTemp2DSet.first, chunk.filteredDSet.first };
	//std::vector< VkDescriptorSet> descriptors = { chunk.filterDSet.first, src, dst };
	if (vkBeginCommandBuffer(chunk.cmdBuffFilter, &sdftBufferBI) != VK_SUCCESS)
		throw std::runtime_error("Cannot begin command buffer");
	vkCmdBindPipeline(chunk.cmdBuffFilter, VK_PIPELINE_BIND_POINT_COMPUTE, filterPipeline);
	vkCmdBindDescriptorSets(chunk.cmdBuffFilter, VK_PIPELINE_BIND_POINT_COMPUTE, filterPipelineLayout, 0, (uint32_t)descriptorsSrcTemp1.size(), descriptorsSrcTemp1.data(), 0, 0);
	vkCmdPushConstants(chunk.cmdBuffFilter, filterPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(FIRState), &state);
	vkCmdDispatch(chunk.cmdBuffFilter, std::max(props.spec_height / SUMMATION_SIZE, 1), std::max(state.signal_len / SUMMATION_WIDTH, 1), 1);
	filterBarrier.buffer = chunk.filterTemp1Buffer.first;
	vkCmdPipelineBarrier(chunk.cmdBuffFilter, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
		VK_DEPENDENCY_BY_REGION_BIT, 0, nullptr, 1, &filterBarrier, 0, nullptr);

	// Sum the multiplications
	vkCmdBindPipeline(chunk.cmdBuffFilter, VK_PIPELINE_BIND_POINT_COMPUTE, sumPipeline);
	for (int stage = 0; stage < nStages - 1; stage++) {
		sumState.stride = sumState.stride / 2;
		if (stage % 2 == 0) {
			vkCmdBindDescriptorSets(chunk.cmdBuffFilter, VK_PIPELINE_BIND_POINT_COMPUTE, sumPipelineLayout, 0, (uint32_t)descriptorsTemp1Temp2.size(), descriptorsTemp1Temp2.data(), 0, 0);
		}
		else {
			vkCmdBindDescriptorSets(chunk.cmdBuffFilter, VK_PIPELINE_BIND_POINT_COMPUTE, sumPipelineLayout, 0, (uint32_t)descriptorsTemp2Temp1.size(), descriptorsTemp2Temp1.data(), 0, 0);
		}
		vkCmdPushConstants(chunk.cmdBuffFilter, sumPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(SUMState), &sumState);
		vkCmdDispatch(chunk.cmdBuffFilter, std::max(sumState.stride / SUMMATION_SIZE, 1), std::max(state.signal_len / SUMMATION_WIDTH, 1), 1);
		vkCmdPipelineBarrier(chunk.cmdBuffFilter, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
			VK_DEPENDENCY_BY_REGION_BIT, 0, nullptr, 1, &filterBarrier, 0, nullptr);
		if (stage % 2 == 0) {
			filterBarrier.buffer = chunk.filterTemp2Buffer.first;
		}
		else {
			filterBarrier.buffer = chunk.filterTemp1Buffer.first;
		}
		vkCmdPipelineBarrier(chunk.cmdBuffFilter, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
			VK_DEPENDENCY_BY_REGION_BIT, 0, nullptr, 1, &filterBarrier, 0, nullptr);
	}
	if (nStages % 2 == 0) {
		vkCmdBindDescriptorSets(chunk.cmdBuffFilter, VK_PIPELINE_BIND_POINT_COMPUTE, sumPipelineLayout, 0, (uint32_t)descriptorsTemp2Dst.size(), descriptorsTemp2Dst.data(), 0, 0);
	}
	else {
		vkCmdBindDescriptorSets(chunk.cmdBuffFilter, VK_PIPELINE_BIND_POINT_COMPUTE, sumPipelineLayout, 0, (uint32_t)descriptorsTemp1Dst.size(), descriptorsTemp1Dst.data(), 0, 0);
	}
	sumState.stride = sumState.stride / 2; // Stride should be 1 here
	sumState.out_stride = 1;
	vkCmdPushConstants(chunk.cmdBuffFilter, sumPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(SUMState), &sumState);
	vkCmdDispatch(chunk.cmdBuffFilter, 1, std::max(state.signal_len / SUMMATION_WIDTH, 1), 1);
	if (vkEndCommandBuffer(chunk.cmdBuffFilter) != VK_SUCCESS)
		throw std::runtime_error("Cannot begin command buffer");


	if (vkBeginCommandBuffer(chunk.cmdBuffMaskSDFT, &sdftBufferBI) != VK_SUCCESS)
		throw std::runtime_error("Cannot begin SDFT Process buffer");
	vkCmdBindPipeline(chunk.cmdBuffMaskSDFT, VK_PIPELINE_BIND_POINT_COMPUTE, sdftPipeline);

	recordSDFT(chunk.cmdBuffMaskSDFT, chunk.maskDSet.first, chunk.filterDSet.first, chunk, chunk.maskBuffer.first, chunk.specRawBuffer.first, true, true);

	if (vkEndCommandBuffer(chunk.cmdBuffMaskSDFT) != VK_SUCCESS)
		throw std::runtime_error("Cannot begin command buffer");

	// Run read-resize queue
	vkBeginCommandBuffer(chunk.cmdBuffMaskRead, &transferBufferBI);

	LinearResize resize = {
		.src_rows = props.hostMaskHeight,
		.src_cols = props.hostMaskWidth,
		.dst_rows = props.spec_height,
		.dst_cols = props.segment_width
	};
	std::vector< VkDescriptorSet> pipeInput = { chunk.maskHostDSet.first, chunk.maskDSet.first };
	vkCmdBindPipeline(chunk.cmdBuffMaskRead, VK_PIPELINE_BIND_POINT_COMPUTE, readMaskPipeline);

	vkCmdPushConstants(chunk.cmdBuffMaskRead, readPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(LinearResize), &resize);
	vkCmdBindDescriptorSets(chunk.cmdBuffMaskRead, VK_PIPELINE_BIND_POINT_COMPUTE, readPipelineLayout, 0, (uint32_t)pipeInput.size(), pipeInput.data(), 0, 0);
	vkCmdDispatch(chunk.cmdBuffMaskRead, std::max(props.spec_height / 1024, 1), props.segment_width, 1);

	vkEndCommandBuffer(chunk.cmdBuffMaskRead);

	chunk.submitInfoMaskRead = {
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.pNext = 0,
		.waitSemaphoreCount = 0,
		.pWaitSemaphores = 0,
		.pWaitDstStageMask = 0,
		.commandBufferCount = 1,
		.pCommandBuffers = &chunk.cmdBuffMaskRead,
		.signalSemaphoreCount = 1,
		.pSignalSemaphores = &chunk.smphMaskRead
	};

	chunk.submitInfoFilter = {
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.pNext = 0,
		.waitSemaphoreCount = 1,
		.pWaitSemaphores = &chunk.smphFiltersRdy,
		.pWaitDstStageMask = &chunk.waitStagesCompute,
		.commandBufferCount = 1,
		.pCommandBuffers = &chunk.cmdBuffFilter,
		.signalSemaphoreCount = 0,
		.pSignalSemaphores = 0
	};

	chunk.submitInfoMaskSDFT = {
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.pNext = 0,
		.waitSemaphoreCount = 1,
		.pWaitSemaphores = &chunk.smphMaskRead,
		.pWaitDstStageMask = &chunk.waitStagesCompute,
		.commandBufferCount = 1,
		.pCommandBuffers = &chunk.cmdBuffMaskSDFT,
		.signalSemaphoreCount = 1,
		.pSignalSemaphores = &chunk.smphFiltersRdy
	};
}

void SDFTFilter::destroyChunk(Chunk& chunk)
{
	vkDestroyFence(context.device, chunk.fenceFilter, 0);
	vkDestroySemaphore(context.device, chunk.smphMaskRead, 0);
	vkDestroySemaphore(context.device, chunk.smphFiltersRdy, 0);

	vkDestroyFence(context.device, chunk.fenceSDFT, 0);
	vkDestroySemaphore(context.device, chunk.smphProcessedSDFT, 0);
	vkDestroySemaphore(context.device, chunk.smphUploadedSDFT, 0);
	vkDestroyCommandPool(context.device, chunk.cmdPoolCompute, 0);
	vkDestroyCommandPool(context.device, chunk.cmdPoolTransfer, 0);

	vkDestroyDescriptorPool(context.device, chunk.maskHostDSet.second, 0);
	vkDestroyDescriptorPool(context.device, chunk.filterDSet.second, 0);
	vkDestroyDescriptorPool(context.device, chunk.maskDSet.second, 0);
	vkDestroyDescriptorPool(context.device, chunk.dstSDFTFiltDSet.second, 0);
	vkDestroyDescriptorPool(context.device, chunk.dstSDFTDSet.second, 0);
	vkDestroyDescriptorPool(context.device, chunk.temp2DSet.second, 0);
	vkDestroyDescriptorPool(context.device, chunk.temp1DSet.second, 0);
	vkDestroyDescriptorPool(context.device, chunk.filteredDSet.second, 0);
	vkDestroyDescriptorPool(context.device, chunk.srcDSetExt.second, 0);
	vkDestroyDescriptorPool(context.device, chunk.srcDSet.second, 0);
	vkDestroyDescriptorPool(context.device, chunk.filterTemp1DSet.second, 0);
	vkDestroyDescriptorPool(context.device, chunk.filterTemp2DSet.second, 0);

	vkFreeMemory(context.device, chunk.maskHostBuffer.second, 0);
	vkDestroyBuffer(context.device, chunk.maskHostBuffer.first, 0);
	vkFreeMemory(context.device, chunk.filtersBuffer.second, 0);
	vkDestroyBuffer(context.device, chunk.filtersBuffer.first, 0);
	vkFreeMemory(context.device, chunk.maskBuffer.second, 0);
	vkDestroyBuffer(context.device, chunk.maskBuffer.first, 0);
	vkFreeMemory(context.device, chunk.specFiltBuffer.second, 0);
	vkDestroyBuffer(context.device, chunk.specFiltBuffer.first, 0);
	vkFreeMemory(context.device, chunk.specRawBuffer.second, 0);
	vkDestroyBuffer(context.device, chunk.specRawBuffer.first, 0);
	vkFreeMemory(context.device, chunk.signalFiltBuffer.second, 0);
	vkDestroyBuffer(context.device, chunk.signalFiltBuffer.first, 0);
	vkFreeMemory(context.device, chunk.signalRawBuffer.second, 0);
	vkDestroyBuffer(context.device, chunk.signalRawBuffer.first, 0);
	vkFreeMemory(context.device, chunk.signalRawExtBuffer.second, 0);
	vkDestroyBuffer(context.device, chunk.signalRawExtBuffer.first, 0);
	vkFreeMemory(context.device, chunk.sdftTemp2Buffer.second, 0);
	vkDestroyBuffer(context.device, chunk.sdftTemp2Buffer.first, 0);
	vkFreeMemory(context.device, chunk.sdftTemp1Buffer.second, 0);
	vkDestroyBuffer(context.device, chunk.sdftTemp1Buffer.first, 0);
	vkFreeMemory(context.device, chunk.filterTemp2Buffer.second, 0);
	vkDestroyBuffer(context.device, chunk.filterTemp2Buffer.first, 0);
	vkFreeMemory(context.device, chunk.filterTemp1Buffer.second, 0);
	vkDestroyBuffer(context.device, chunk.filterTemp1Buffer.first, 0);

	vkDestroyBuffer(context.device, chunk.uploadBuffer.first, 0);
	vkFreeMemory(context.device, chunk.uploadBuffer.second, 0);
	vkDestroyBuffer(context.device, chunk.bufferSignal.first, 0);
	vkFreeMemory(context.device, chunk.bufferSignal.second, 0);
	vkDestroyBuffer(context.device, chunk.bufferSpec.first, 0);
	vkFreeMemory(context.device, chunk.bufferSpec.second, 0);
}

void SDFTFilter::update(Chunk& chunk, int* data)
{
#ifdef PROFILING
	auto start = std::chrono::high_resolution_clock::now();
	std::cout << std::endl << "Processing. ";
#endif
	// READ AND RESIZE THE MASK
	void* memptr;
	uint32_t size = props.hostMaskHeight * props.segment_width * sizeof(int);
	vkMapMemory(context.device, chunk.maskHostBuffer.second, 0, size, 0, &memptr);
	memcpy(memptr, data, size);
	vkUnmapMemory(context.device, chunk.maskHostBuffer.second);
#ifdef PROFILING
	auto end = std::chrono::high_resolution_clock::now();
	float time_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	std::cout << "to mapped: " << time_ms << ' ';
	start = std::chrono::high_resolution_clock::now();
#endif

	if (vkQueueSubmit(rawSDFTQueue, 1, &chunk.submitInfoMaskRead, VK_NULL_HANDLE) != VK_SUCCESS)
		throw std::runtime_error("Cannot submit to filtering queue");
	if (vkQueueSubmit(rawSDFTQueue, 1, &chunk.submitInfoMaskSDFT, VK_NULL_HANDLE) != VK_SUCCESS)
		throw std::runtime_error("Cannot submit to filtering queue");

	if (vkQueueSubmit(rawSDFTQueue, 1, &chunk.submitInfoFilter, chunk.fenceFilter) != VK_SUCCESS)
		throw std::runtime_error("Cannot submit to filtering queue");
	vkWaitForFences(context.device, 1, &chunk.fenceFilter, VK_TRUE, (uint64_t)-1);
	vkResetFences(context.device, 1, &chunk.fenceFilter);

#ifdef PROFILING
	end = std::chrono::high_resolution_clock::now();
	time_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	std::cout << "filtering: " << time_ms << ' ';
	start = std::chrono::high_resolution_clock::now();
#endif
}

void SDFTFilter::update(int* mask, const std::vector<float>& signalIn, std::vector<float>& signalOut)
{
	// Upload the signal onto GPU
#ifdef PROFILING
	auto start = std::chrono::high_resolution_clock::now();
#endif
	VkDeviceSize size = sizeof(glm::vec2) * signalIn.size();
	std::vector<glm::vec2> input(signalIn.size());
	// The signalIn must include spectrogram_height / 2 items from both sides
	for (uint32_t i = 0; i < (uint32_t)input.size(); i++) {
		input[i].x = signalIn[i];
		input[i].y = 0;
	}
#ifdef PROFILING
	auto end = std::chrono::high_resolution_clock::now();
	float time_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	std::cout << "Form signal: " << time_ms << ' ';
	start = std::chrono::high_resolution_clock::now();
#endif
	void* memptr;
	vkMapMemory(context.device, chunk.uploadBuffer.second, 0, size, 0, &memptr);
	memcpy(memptr, input.data(), (size_t)size);
	vkUnmapMemory(context.device, chunk.uploadBuffer.second);
#ifdef PROFILING
	end = std::chrono::high_resolution_clock::now();
	time_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	std::cout << "copy mapped: " << time_ms << ' ';
	start = std::chrono::high_resolution_clock::now();
#endif

	// Transfer into device local buffer
	VkCommandBufferBeginInfo transferBufferBI = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.pNext = 0,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
		.pInheritanceInfo = 0
	};
	vkBeginCommandBuffer(transferCommandBuffer, &transferBufferBI);
	VkBufferCopy bufferCopyRegion = {
		.srcOffset = (VkDeviceSize)(props.spec_height / 2 * sizeof(glm::vec2)),
		.dstOffset = 0,
		.size = size - (props.spec_height * sizeof(glm::vec2))
	};
	vkCmdCopyBuffer(transferCommandBuffer, chunk.uploadBuffer.first, chunk.signalRawBuffer.first, 1, &bufferCopyRegion);
	bufferCopyRegion.srcOffset = 0;
	bufferCopyRegion.size = size;
	vkCmdCopyBuffer(transferCommandBuffer, chunk.uploadBuffer.first, chunk.signalRawExtBuffer.first, 1, &bufferCopyRegion);
	vkEndCommandBuffer(transferCommandBuffer);

	VkSubmitInfo submitInfo = {
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.pNext = 0,
		.waitSemaphoreCount = 0,
		.pWaitSemaphores = 0,
		.pWaitDstStageMask = 0,
		.commandBufferCount = 1,
		.pCommandBuffers = &transferCommandBuffer,
		.signalSemaphoreCount = 0,
		.pSignalSemaphores = 0
	};
	vkQueueSubmit(transferQueue, 1, &submitInfo, VK_NULL_HANDLE);
	vkQueueWaitIdle(transferQueue);
#ifdef PROFILING
	end = std::chrono::high_resolution_clock::now();
	time_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	std::cout << "to device_local: " << time_ms << ' ';
	start = std::chrono::high_resolution_clock::now();
#endif

	// Process the signal
	update(chunk, mask);

#ifdef PROFILING
	end = std::chrono::high_resolution_clock::now();
	time_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	std::cout << std::endl << "Overall processing: " << time_ms << std::endl;
	start = std::chrono::high_resolution_clock::now();
#endif
	// Output the results
	signalOut.resize(props.spec_height + props.hop * props.segment_width);
	VkDeviceSize signalSize = sizeof(glm::vec2) * signalOut.size();

	vkBeginCommandBuffer(transferCommandBuffer, &transferBufferBI);
	VkBufferCopy signalBufferCopyRegion = {
		.srcOffset = 0,
		.dstOffset = 0,
		.size = signalSize
	};
	vkCmdCopyBuffer(transferCommandBuffer, chunk.signalFiltBuffer.first, chunk.bufferSignal.first, 1, &signalBufferCopyRegion);
	vkEndCommandBuffer(transferCommandBuffer);

	vkQueueSubmit(transferQueue, 1, &submitInfo, VK_NULL_HANDLE);
	vkQueueWaitIdle(transferQueue);

#ifdef PROFILING
	end = std::chrono::high_resolution_clock::now();
	time_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	std::cout << "to host_local: " << time_ms << ' ';
	start = std::chrono::high_resolution_clock::now();
#endif
	//void* memptr;
	vkMapMemory(context.device, chunk.bufferSignal.second, 0, signalSize, 0, &memptr);
	// memcpy(signalOut.data(), memptr, signalSize);
	float* outpt = (float* )memptr;
	for (uint32_t i = 0; i < (uint32_t)signalOut.size(); i++) {
		signalOut[i] = outpt[i * 2];
	}
	vkUnmapMemory(context.device, chunk.bufferSignal.second);
#ifdef PROFILING
	end = std::chrono::high_resolution_clock::now();
	time_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	std::cout << "from mapped: " << time_ms << std::endl << std::endl;
#endif
}

void SDFTFilter::calcSDFT(const std::vector<float>& signalIn, std::vector<float>& specOut)
{
#ifdef PROFILING
	auto start = std::chrono::high_resolution_clock::now();
#endif
	// Upload the signal onto GPU
	VkDeviceSize size = sizeof(glm::vec2) * signalIn.size();
	std::vector<glm::vec2> input(signalIn.size());
	// The signalIn must include spectrogram_height / 2 items from both sides
	for (uint32_t i = 0; i < (uint32_t)input.size(); i++) {
		input[i].x = signalIn[i];
		input[i].y = 0;
	}

#ifdef PROFILING
	auto end = std::chrono::high_resolution_clock::now();
	float time_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	std::cout << "From signal: " << time_ms << ' ';
	start = std::chrono::high_resolution_clock::now();
#endif
	void* memptr;
	vkMapMemory(context.device, chunk.uploadBuffer.second, 0, size, 0, &memptr);
	memcpy(memptr, input.data(), (uint32_t)size);
	vkUnmapMemory(context.device, chunk.uploadBuffer.second);

#ifdef PROFILING
	end = std::chrono::high_resolution_clock::now();
	time_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	std::cout << "to mapped: " << time_ms << ' ';
	start = std::chrono::high_resolution_clock::now();
#endif
	if (vkQueueSubmit(transferQueue, 1, &chunk.submitInfoSDFTUpload, VK_NULL_HANDLE) != VK_SUCCESS)
		throw std::runtime_error("Cannot submit to SDFT Upload queue");
	if (vkQueueSubmit(rawSDFTQueue, 1, &chunk.submitInfoSDFTProcess, VK_NULL_HANDLE) != VK_SUCCESS)
		throw std::runtime_error("Cannot submit to SDFT Process queue");
	if (vkQueueSubmit(transferQueue, 1, &chunk.submitInfoSDFTDownload, chunk.fenceSDFT) != VK_SUCCESS)
		throw std::runtime_error("Cannot submit to SDFT Download queue");

	// Output the results
	specOut.resize(props.spec_height * props.segment_width);
	VkDeviceSize specSize = sizeof(glm::vec2) * specOut.size();
	vkWaitForFences(context.device, 1, &chunk.fenceSDFT, VK_TRUE, (uint64_t)-1);
	vkResetFences(context.device, 1, &chunk.fenceSDFT);

#ifdef PROFILING
	end = std::chrono::high_resolution_clock::now();
	time_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	std::cout << std::endl << "SDFT overall: " << time_ms << ' ';
	start = std::chrono::high_resolution_clock::now();
#endif
	vkMapMemory(context.device, chunk.bufferSpec.second, 0, specSize, 0, &memptr);
	for (uint32_t i = 0; i < (uint32_t)specOut.size(); i++) {
		specOut[i] = sqrt(((float*)memptr)[i*2] * ((float*)memptr)[i*2] + ((float*)memptr)[i*2 + 1] * ((float*)memptr)[i*2 + 1]);
	}
	vkUnmapMemory(context.device, chunk.bufferSpec.second);

#ifdef PROFILING
	end = std::chrono::high_resolution_clock::now();
	time_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	std::cout << "from mapped: " << time_ms << std::endl << std::endl;
#endif
}

int SDFTFilter::getSpecWidth()
{
	return (int)(props.spec_height * (ceil((double)props.max_signal_size / props.hop) + 1));
}

void SDFTFilter::createDescriptorSets()
{
	// Pipelines
	Shader sdft = getShaderModule(context.device, "Shaders/sdft.spv", VK_SHADER_STAGE_COMPUTE_BIT);
	Shader read = getShaderModule(context.device, "Shaders/read.spv", VK_SHADER_STAGE_COMPUTE_BIT);
	Shader filter = getShaderModule(context.device, "Shaders/filter.spv", VK_SHADER_STAGE_COMPUTE_BIT);
	Shader sum = getShaderModule(context.device, "Shaders/sum.spv", VK_SHADER_STAGE_COMPUTE_BIT);
	std::vector<VkDescriptorSetLayout> descriptorLayouts = { sdftDescriptorSetLayout , sdftDescriptorSetLayout };
	VkPushConstantRange sdftConstantRange = {
		.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
		.offset = 0,
		.size = sizeof(SDFTState)
	};
	VkPushConstantRange readConstantRange = {
		.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
		.offset = 0,
		.size = sizeof(LinearResize)
	};
	VkPushConstantRange filterConstantRange = {
		.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
		.offset = 0,
		.size = sizeof(FIRState)
	};
	VkPushConstantRange sumConstantRange = {
		.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
		.offset = 0,
		.size = sizeof(SUMState)
	};
	VkPipelineLayoutCreateInfo computeLayoutCI = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.pNext = 0,
		.flags = 0,
		.setLayoutCount = (uint32_t)descriptorLayouts.size(),
		.pSetLayouts = (VkDescriptorSetLayout*)descriptorLayouts.data(),
		.pushConstantRangeCount = 1,
		.pPushConstantRanges = &sdftConstantRange
	};
	if (vkCreatePipelineLayout(context.device, &computeLayoutCI, 0, &sdftPipelineLayout) != VK_SUCCESS)
		throw std::runtime_error("Cannot create compute pipeline layout");

	computeLayoutCI.pPushConstantRanges = &readConstantRange;
	if (vkCreatePipelineLayout(context.device, &computeLayoutCI, 0, &readPipelineLayout) != VK_SUCCESS)
		throw std::runtime_error("Cannot create compute pipeline layout");

	computeLayoutCI.pPushConstantRanges = &sumConstantRange;
	if (vkCreatePipelineLayout(context.device, &computeLayoutCI, 0, &sumPipelineLayout) != VK_SUCCESS)
		throw std::runtime_error("Cannot create compute pipeline layout");


	// LAYOUT CHANGED HERE
	// Filter layout: <Filters, Input, Output>
	computeLayoutCI.pPushConstantRanges = &filterConstantRange;
	descriptorLayouts.push_back(sdftDescriptorSetLayout);
	computeLayoutCI.setLayoutCount = (uint32_t)descriptorLayouts.size();
	computeLayoutCI.pSetLayouts = (VkDescriptorSetLayout*)descriptorLayouts.data();
	if (vkCreatePipelineLayout(context.device, &computeLayoutCI, 0, &filterPipelineLayout) != VK_SUCCESS)
		throw std::runtime_error("Cannot create compute pipeline layout");

	VkComputePipelineCreateInfo computePipelineCI = {
		.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
		.pNext = 0,
		.flags = 0,
		.stage = sdft.stageCI,
		.layout = sdftPipelineLayout,
		.basePipelineHandle = VK_NULL_HANDLE,
		.basePipelineIndex = 0
	};
	if (vkCreateComputePipelines(context.device, VK_NULL_HANDLE, 1, &computePipelineCI, 0, &sdftPipeline) != VK_SUCCESS)
		throw std::runtime_error("Cannot create compute pipeline");

	computePipelineCI.layout = readPipelineLayout;
	computePipelineCI.stage = read.stageCI;
	if (vkCreateComputePipelines(context.device, VK_NULL_HANDLE, 1, &computePipelineCI, 0, &readMaskPipeline) != VK_SUCCESS)
		throw std::runtime_error("Cannot create compute pipeline");

	computePipelineCI.layout = filterPipelineLayout;
	computePipelineCI.stage = filter.stageCI;
	if (vkCreateComputePipelines(context.device, VK_NULL_HANDLE, 1, &computePipelineCI, 0, &filterPipeline) != VK_SUCCESS)
		throw std::runtime_error("Cannot create compute pipeline");

	computePipelineCI.layout = sumPipelineLayout;
	computePipelineCI.stage = sum.stageCI;
	if (vkCreateComputePipelines(context.device, VK_NULL_HANDLE, 1, &computePipelineCI, 0, &sumPipeline) != VK_SUCCESS)
		throw std::runtime_error("Cannot create compute pipeline");

	// Transfer command buffer
	VkCommandPoolCreateInfo transferCommandPoolCI = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
		.pNext = 0,
		.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
		.queueFamilyIndex = (uint32_t)context.transferFamilyIdx
	};
	if (vkCreateCommandPool(context.device, &transferCommandPoolCI, 0, &transferCommandPool) != VK_SUCCESS)
		throw std::runtime_error("Cannot create filter command pool");
	VkCommandBufferAllocateInfo transferCommandBufferAI = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.pNext = 0,
		.commandPool = transferCommandPool,
		.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		.commandBufferCount = 1
	};
	if (vkAllocateCommandBuffers(context.device, &transferCommandBufferAI, &transferCommandBuffer) != VK_SUCCESS)
		throw std::runtime_error("Cannot create transfer command buffer");
	vkDestroyShaderModule(context.device, filter.shaderModule, 0);
	vkDestroyShaderModule(context.device, read.shaderModule, 0);
	vkDestroyShaderModule(context.device, sdft.shaderModule, 0);
	vkDestroyShaderModule(context.device, sum.shaderModule, 0);
}

void SDFTFilter::createStorageBuffer(VkDeviceSize size, std::pair<VkBuffer, VkDeviceMemory>& buffer, Binding& binding, bool is_host_visible)
{
	std::vector<uint32_t> queueFamilies = { (uint32_t)context.sdftFamilyIdx };
	// if (is_drawing) queueFamilies.push_back((uint32_t)context.graphicsFamilyIdx);
	VkMemoryPropertyFlags memProperties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
	if (is_host_visible) memProperties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
	buffer = createBuffer(context.device, context.physicalDevice, queueFamilies,
		size, memProperties,
		VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
	binding.buffer.data = buffer.first;
	binding.buffer.step = 0;
	binding.memory = buffer.second;
	binding.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
}


// In and out buffers should the ones bound to the src and dst descriptor sets
void SDFTFilter::recordSDFT(VkCommandBuffer commandBuffer, VkDescriptorSet src, VkDescriptorSet dst, Chunk chunk, 
	VkBuffer inBuffer, VkBuffer outBuffer, bool isInverse, bool isShift) 
{
	SDFTState state = {
		.stageStride = 4,
		.hop = 10,
		.isWriteImg = 0,
		.isInverse = 0,
		.isShift = 0,
		.specHeight = props.spec_height
	};
	int nStages = (int)log2(props.spec_height);
	std::vector< VkDescriptorSet> pipeInput = { src, chunk.temp1DSet.first };
	std::vector< VkDescriptorSet> pipeOutput = { chunk.temp1DSet.first, dst };
	if (nStages % 2 == 1) {
		pipeOutput = { chunk.temp2DSet.first, dst };
	}
	std::vector< VkDescriptorSet> temp1temp2 = { chunk.temp1DSet.first, chunk.temp2DSet.first };
	std::vector< VkDescriptorSet> temp2temp1 = { chunk.temp2DSet.first, chunk.temp1DSet.first };
	VkBufferMemoryBarrier sdftBarrier = {
			   .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
			   .pNext = 0,
			   .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
			   .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT,
			   .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			   .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			   .offset = 0,
			   .size = VK_WHOLE_SIZE
	};
	for (int stage = 0; stage < nStages; stage++) {
		int stride = (int)pow(2, nStages - stage - 1);
		state.stageStride = stride;
		if (stage == 0) {
			if (!isInverse) state.hop = props.hop;
			else state.hop = props.spec_height;
			if (isInverse) state.isInverse = 1;
			sdftBarrier.buffer = inBuffer;
			vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, sdftPipelineLayout, 0, (uint32_t)pipeInput.size(), pipeInput.data(), 0, 0);
			vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
				VK_DEPENDENCY_BY_REGION_BIT, 0, nullptr, 1, &sdftBarrier, 0, nullptr);
		}
		else if (stage == nStages - 1) {
			state.isWriteImg = 1;
			state.hop = props.spec_height;
			if (isInverse) state.isInverse = 1;
			if (isShift) state.isShift = 1;
			sdftBarrier.buffer = outBuffer;
			vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, sdftPipelineLayout, 0, (uint32_t)pipeOutput.size(), pipeOutput.data(), 0, 0);
			vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
				VK_DEPENDENCY_BY_REGION_BIT, 0, nullptr, 1, &sdftBarrier, 0, nullptr);
		}
		else {
			state.isShift = 0;
			state.isInverse = 0;
			state.hop = props.spec_height;
			if (stage % 2 == 0) {
				sdftBarrier.buffer = chunk.sdftTemp2Buffer.first;
				vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, sdftPipelineLayout, 0, (uint32_t)temp2temp1.size(), temp2temp1.data(), 0, 0);
				vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
					VK_DEPENDENCY_BY_REGION_BIT, 0, nullptr, 1, &sdftBarrier, 0, nullptr);
			}
			else {
				sdftBarrier.buffer = chunk.sdftTemp1Buffer.first;
				vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, sdftPipelineLayout, 0, (uint32_t)temp1temp2.size(), temp1temp2.data(), 0, 0);
				vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
					VK_DEPENDENCY_BY_REGION_BIT, 0, nullptr, 1, &sdftBarrier, 0, nullptr);
			}
		}
		vkCmdPushConstants(commandBuffer, sdftPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(SDFTState), &state);
		vkCmdDispatch(commandBuffer, std::max(props.spec_height / 1024, 1), props.segment_width, 1);
	}
}
