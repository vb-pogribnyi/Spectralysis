#include "engine_wrapper.h"
#include <SDFTFilter.h>
#include <VulkanCommon.h>
#include <iostream>

static VulkanContext context;
static SDFTFilter *filter;

void SDFTFilterInit(int hostMaskHeight, int hostMaskWidth, int hop, int specHeight) {
	std::cout << "Initializing SDFTFilter" << std::endl;
	SDFTProps filterProps = {
		.spec_height = specHeight,
		.segment_width = SEGMENT_WIDTH,
		.signal_length = 1024,
		// .max_signal_size = 44 * 1024 * 60 * 1,  // 1 minute @ 44kbps
		// .max_signal_size = 16 * 1024 * 40,  // 40 sec @ 16kbps
		.hop = hop,
		.hostMaskHeight = hostMaskHeight,
		.hostMaskWidth = hostMaskWidth
	};
	std::vector<const char*> extensions = {"VK_KHR_surface", "VK_KHR_win32_surface"};
	context = setupContext(extensions);
	filter = new SDFTFilter(context, filterProps);
}

void SDFTFilterUpdate(int* mask, const std::vector<float>& signalIn, std::vector<float>& signalOut) {
	filter->update(mask, signalIn, signalOut);
}

void calcSDFT(const std::vector<float>& signalIn, std::vector<float>& specOut) {
	filter->calcSDFT(signalIn, specOut);
}