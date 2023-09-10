#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <chrono>
#include "engine_wrapper.h"

namespace py = pybind11;

class Spectralysis {
private:
	std::vector<int> mask;
	std::vector<float> specFilt;
	std::vector<float> signalIn;
	std::vector<float> signalFilt;
	int hop = 128;
	int specHeight = 1024;
public:
	Spectralysis(int hop, int specHeight) : hop(hop), specHeight(specHeight) {
		SDFTFilterInit(specHeight, SEGMENT_WIDTH, hop, specHeight);
		
		signalIn.resize(hop * SEGMENT_WIDTH + 2 * specHeight);
		mask.resize(SEGMENT_WIDTH * specHeight);
		signalFilt.resize(hop * SEGMENT_WIDTH + specHeight);
		specFilt.resize(SEGMENT_WIDTH * specHeight);
		
		std::fill(signalIn.begin(), signalIn.end(), 1.0f);
		std::fill(mask.begin(), mask.end(), 0);
		std::fill(signalFilt.begin(), signalFilt.end(), 0.0f);
		std::fill(specFilt.begin(), specFilt.end(), 0.0f);
		
		for (int i = 0; i < mask.size(); i++) {
			if (i / specHeight < 2) {
				mask[i] = 0x0000FF;
			}
		}
		
		/*
		for (int i = 0; i < 1024*4; i++) {
			std::cout << std::hex << mask[i] << ' ';
		}
		*/

		
		// THIS LINE ALLOCATES THE RIGHT AMOUNT OF MEMORY FOR THE UPLOAD BUFFER
		SDFTFilterUpdate(mask.data(), signalIn, signalFilt);
		/*
		for (int i = 0; i < 128; i++) {
			std::cout << signalFilt[i] << std::endl;
		}*/
		
	}
	
	py::array process(
		const py::array_t<float, py::array::c_style | py::array::forcecast>& in, 
		const py::array_t<int, py::array::c_style | py::array::forcecast>& in_mask
	) {
		py::buffer_info info_in = in.request();
		py::buffer_info info_mask = in_mask.request();
		float* ptr_in = static_cast<float*> (info_in.ptr);
		int* ptr_mask = static_cast<int*> (info_mask.ptr);
		// std::cout << ptr << ' ' << ptr[0] << ' ' << ptr[1] << ' ' << ptr[2] << std::endl;
		/*
		std::cout << out.size() << std::endl;
		for (int i = 0; i < out.size(); i++) {
			std::cout << ptr_out[i] << ' ';
		}
		*/
		
		memcpy(mask.data(), ptr_mask, mask.size() * sizeof(int));
		memcpy(signalIn.data(), ptr_in, signalIn.size() * sizeof(float));
		/*
		for (int i = 0; i < mask.size(); i++) {
			std::cout << mask[i] << ' ';
			if (i % 10 == 0) std::cout << std::endl;
		}
		std::cout << std::endl << mask.size() << std::endl;
		std::cout << std::endl;
		for (int i = 0; i < signalIn.size(); i++) {
			std::cout << signalIn[i] << ' ';
		}
		std::cout << std::endl << signalIn.size() << std::endl;
		*/
		auto start = std::chrono::high_resolution_clock::now();
		SDFTFilterUpdate(mask.data(), signalIn, signalFilt);
		auto end = std::chrono::high_resolution_clock::now();
		std::cout << "Processing executed: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << std::endl;
		py::array output = py::cast(signalFilt);
		/*
		for (int i = 0; i < signalFilt.size(); i++) {
			std::cout << signalFilt[i] << ' ';
		}
		std::cout << std::endl;
		std::cout << signalFilt.size() << std::endl;
		std::cout << output.size() << std::endl;
		std::cout << std::endl;
		*/
		
		return output;
	}
	
	py::array sdft(
		const py::array_t<float, py::array::c_style | py::array::forcecast>& in
	) {
		std::cout << "inside: sdft" << std::endl;
		py::buffer_info info_in = in.request();
		std::cout << "reqested: sdft" << std::endl;
		float* ptr_in = static_cast<float*> (info_in.ptr);
		std::cout << "got pointer: sdft" << std::endl;

		memcpy(signalFilt.data(), ptr_in, signalFilt.size() * sizeof(float));
		std::cout << "copied memory: sdft" << std::endl;
		std::cout << "calling: sdft" << std::endl;
		auto start = std::chrono::high_resolution_clock::now();
		calcSDFT(signalFilt, specFilt);
		auto end = std::chrono::high_resolution_clock::now();
		std::cout << "SDFT executed: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << std::endl;
		/*
		for (int i = 0; i < signalFilt.size(); i++) {
			std::cout << i << ' ' << signalFilt[i] << ' ';
		}
		std::cout << std::endl;
		std::cout << signalFilt.size() << std::endl;
		std::cout << std::endl;
		for (int i = 0; i < specFilt.size(); i++) {
			if (i % 10 == 0) std::cout << std::endl;
			std::cout << i << ' ' << specFilt[i] << ' ';
		}
		std::cout << std::endl;
		std::cout << std::endl;
		std::cout << std::endl;
		*/
		py::array output = py::cast(specFilt);
		return output;
	}
	
	std::vector<int> getsize() {
		std::vector<int> result = {
			hop * SEGMENT_WIDTH + 2 * specHeight,
			SEGMENT_WIDTH * specHeight,
			hop * SEGMENT_WIDTH + specHeight,
			SEGMENT_WIDTH * specHeight
		};
		return result;
	}
};

void test_func(py::module &m) {
    
    py::class_<Spectralysis>(m, "Spectralysis")
    .def(py::init<int, int>(), py::arg("hop"), py::arg("spec_height"))
    .def("process", &Spectralysis::process)
    .def("sdft", &Spectralysis::sdft)
    .def("getsize", &Spectralysis::getsize);
}

PYBIND11_MODULE(PySpectralysis, m) {
    // Optional docstring
    m.doc() = "The test library";
    
    test_func(m);
}
