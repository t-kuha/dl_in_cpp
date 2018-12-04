// Run TensorFlow Lite model
//

#include <iostream>

#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/model.h"
#include "tensorflow/contrib/lite/optional_debug_tools.h"
#include "tensorflow/contrib/lite/string_util.h"

#define LOG(x) std::cerr


int main(int argc, const char * argv[]) 
{
    std::cout << "..... tflite ....." << std::endl;
    
    // Load tflite model
    std::string model_name = "linear.tflite";
    
    std::unique_ptr<tflite::FlatBufferModel> model;
    std::unique_ptr<tflite::Interpreter> interpreter;
    model = tflite::FlatBufferModel::BuildFromFile(model_name.c_str());
    if (!model) {
        LOG(FATAL) << "Failed to mmap model " << model_name << std::endl;
        exit(-1);
    }

    LOG(INFO) << "Loaded model " << model_name << std::endl;
    model->error_reporter();
    LOG(INFO) << "resolved reporter" << std::endl;
    
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    if (!interpreter) {
        LOG(FATAL) << "Failed to construct interpreter" << std::endl;
        exit(-1);
    }
    
    int input = interpreter->inputs()[0];
    
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        LOG(FATAL) << "Failed to allocate tensors!";
    }
    
//    PrintInterpreterState(interpreter.get());

    // Set input value
    float input_val = 10;
    std::cout << "Input value = " << input_val << std::endl;
    interpreter->typed_tensor<float>(input)[0] = input_val;
    
    // Run inference
    if (interpreter->Invoke() != kTfLiteOk) {
        LOG(FATAL) << "Failed to invoke tflite!" << std::endl;
    }

    // Show result
    std::cout << "Output value = " << 
        interpreter->typed_output_tensor<float>(0)[0] << std::endl;

    std::cout << ".................." << std::endl;
    
    return 0;
}
