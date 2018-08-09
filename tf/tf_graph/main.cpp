/*
 * https://medium.com/jim-fleming/loading-a-tensorflow-graph-with-the-c-api-4caaff88463f
 */

#include <iostream>

#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"

using namespace tensorflow;

int main(int argc, char* argv[])
{
	std::cout << "---------- TensorFlow in C/C++ -----------" << std::endl;

	// Initialize a tensorflow session
	Session* session;
	Status status = NewSession(SessionOptions(), &session);
	if (!status.ok()) {
		std::cout << status.ToString() << std::endl;
		return 1;
	}

	// Read in the protobuf graph we exported
	GraphDef graph_def;
	status = ReadBinaryProto(Env::Default(), "graph.pb", &graph_def);
	if (!status.ok()) {
		std::cout << status.ToString() << std::endl;
		return 1;
	}

	// Add the graph to the session
	status = session->Create(graph_def);
	if (!status.ok()) {
		std::cout << status.ToString() << std::endl;
		return 1;
	}

	// Setup inputs and outputs:

	// Our graph doesn't require any inputs, since it specifies default values,
	// but we'll change an input to demonstrate.
	Tensor a(DT_FLOAT, TensorShape());
	a.scalar<float>()() = 3.0;

	Tensor b(DT_FLOAT, TensorShape());
	b.scalar<float>()() = 2.0;

	std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
			{ "a", a },
			{ "b", b },
	};

	// The session will initialize the outputs
	std::vector<tensorflow::Tensor> outputs;

	// Run the session, evaluating our "c" operation from the graph
	status = session->Run(inputs, {"c"}, {}, &outputs);
	if (!status.ok()) {
		std::cout << status.ToString() << std::endl;
		return 1;
	}

	// Grab the first output (we only evaluated one graph node: "c")
	// and convert the node to a scalar representation.
	auto output_c = outputs[0].scalar<float>();

	// (There are similar methods for vectors and matrices here:
	// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/public/tensor.h)

	// Print the results
	//std::cout << outputs[0].DebugString() << std::endl; // Tensor<type: float shape: [] values: 6>
	std::cout << output_c() << std::endl; // 6

	// Free any resources used by the session
	session->Close();

	std::cout << "------------------------------------------" << std::endl;

	return 0;
}
