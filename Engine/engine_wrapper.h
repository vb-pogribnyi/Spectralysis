#pragma once
#include <vector>
#include "dlib_export.h"

#define SPEC_HEIGHT 1024
#define SEGMENT_WIDTH 32

DLIB_EXPORT void SDFTFilterInit(int hostMaskHeight, int hostMaskWidth, int hop, int specHeight);
DLIB_EXPORT void SDFTFilterUpdate(int* mask, const std::vector<float>& signalIn, std::vector<float>& signalOut);
DLIB_EXPORT void calcSDFT(const std::vector<float>& signalIn, std::vector<float>& specOut);