/*
 * Empty C++ Application
 */

#include <iostream>

//#include "hls_math.h"

#include "boost/shared_ptr.hpp"	// shared_ptr
#include "caffe/caffe.hpp"
#include "caffe/net.hpp"

int main()
{
	std::cout << "------------ Caffe Test -------------" << std::endl;

	std::string path_prototext = "lenet.prototxt";
	std::string path_caffemodel = "lenet_iter_10000.caffemodel";

	// Use CPU mode
	caffe::Caffe::set_mode(caffe::Caffe::CPU);

	// Load caffemodel
	caffe::Net<float> net(path_prototext, caffe::Phase::TEST);
	net.CopyTrainedLayersFrom(path_caffemodel);

	//
	std::vector<std::string> layer_names = net.blob_names();
	for (unsigned int i = 0; i < layer_names.size(); i++) {
		std::cout << "Layer " << i << ": " << layer_names.at(i) << std::endl;

		// -----
		auto b = net.layer_by_name(layer_names.at(i)).get()->blobs();
		std::cout << "\t Size:" << b.size() << std::endl;

		if (b.size() == 0) {
			std::cout << "\t continuing..." << std::endl;
			continue;
		}

		for (unsigned int j = 0; j < b.size(); j++) {
			std::cout << "\t shape string: " << b.at(j).get()->shape_string() << std::endl;
			std::cout << "\t ";
			for (int c = 0; c < b.at(j).get()->num_axes(); c++) {
				std::cout <<
					b.at(j).get()->shape(c);
				if (c < b.at(j).get()->num_axes() - 1) {
					std::cout << " x ";
				}
			}
			std::cout << std::endl;

			float* ptr = (float*)(b.at(j).get()->data().get()->cpu_data());

			std::cout << "\t " <<
				ptr[0] << " " << ptr[1] << " " <<
				ptr[2] << " " << ptr[3] << std::endl;
		}
	}

	// Show weight
	auto b = net.layer_by_name("ip1").get()->blobs();
	float* ptr = (float*)(b.at(1).get()->data().get()->cpu_data());
	for (int p = 0; p < b.at(1).get()->shape(0); p++)
		std::cout << "\t " << ptr[p] << std::endl;

	//	std::cout << "\t size: " << w.size() << std::endl;
	//	for(unsigned int i = 0; i < w.size(); i++){
	//		std::cout << w.at(i) << std::endl;;
	//	}

	// Save caffemodel
	//net.ToHDF5("aaa.cafemodel");


	std::cout << "-------------------------------------" << std::endl;

	return 0;
}
