// Use custom operator in tflite
// 
// https://www.tensorflow.org/lite/apis
// https://www.tensorflow.org/lite/custom_operators
// 


#include <iostream>

#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/kernels/kernel_util.h"

#define LOG(x) std::cerr

TfLiteStatus ASinPrepare(TfLiteContext* context, TfLiteNode* node);
TfLiteStatus ASinEval(TfLiteContext* context, TfLiteNode* node);
TfLiteRegistration* Register_ASIN();


int main(int argc, const char * argv[]) 
{
    std::cout << "..... tflite ....." << std::endl;
    
    // Load tflite model ( y = asin(x + offset)  where offset = 0.2 )
    std::string model_name = "tflite_op_asin.tflite";

    std::unique_ptr<tflite::FlatBufferModel> model;
    std::unique_ptr<tflite::Interpreter> interpreter;
    model = tflite::FlatBufferModel::BuildFromFile(model_name.c_str());
    if (!model) {
        LOG(FATAL) << "Failed to mmap model " << model_name << std::endl;
        return -1;
    }

    model->error_reporter();

    // Add custom operation (arcsine)
    tflite::ops::builtin::BuiltinOpResolver resolver;
    resolver.AddCustom("Asin", Register_ASIN());

    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    if (!interpreter) {
        LOG(FATAL) << "Failed to construct interpreter" << std::endl;
        return -1;
    }

    
    const std::vector<int> inputs = interpreter->inputs();
    const std::vector<int> outputs = interpreter->outputs();

    // Tensor to show
    int input = inputs[0];
    int output = outputs[0];

    // Show some info
    LOG(INFO) << "----- Input -----" << std::endl;
    std::cout << "Num. of input: " << inputs.size() << std::endl;
    for(int i = 0; i < inputs.size(); i++){
        std::cout << "\t " << i << ": " << interpreter->GetInputName(i) << std::endl;
    }

    TfLiteIntArray* input_dims = interpreter->tensor(input)->dims;
    if(input_dims->size > 0){
        std::cout << "dims: " << input_dims->size << ", size = [ ";
        for(int i = 0; i < input_dims->size; i++){
            std::cout << input_dims->data[i] << " ";
        }
        std::cout << "]" << std::endl;
    }


    // Set input value
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        LOG(FATAL) << "Failed to allocate tensors!";
        return -1;
    }

    interpreter->typed_tensor<float>(input)[0] = 0.05;
    interpreter->typed_tensor<float>(input)[1] = 0.13;
    interpreter->typed_tensor<float>(input)[2] = -0.34;

    // PrintInterpreterState(interpreter.get());
    
    // Run inference
    if (interpreter->Invoke() != kTfLiteOk) {
        LOG(FATAL) << "Failed to invoke tflite!" << std::endl;
    }


    // Show info on output tensor; only valid after inference
    LOG(INFO) << "----- Output -----" << std::endl;
    std::cout << "Num. of output: " << outputs.size() << std::endl;
    for(int i = 0; i < outputs.size(); i++){
        std::cout << "\t " << i << ": " << interpreter->GetOutputName(i) << std::endl;
    }

    TfLiteIntArray* output_dims = interpreter->tensor(output)->dims;
    if(output_dims->size > 0){
        std::cout << "dims: " << output_dims->size << ", size = [ ";
        for(int i = 0; i < output_dims->size; i++){
            std::cout << output_dims->data[i] << " ";
        }
        std::cout << "]" << std::endl;
    }

    // Show result
    auto output_size = output_dims->data[output_dims->size - 1];

    std::cout << "----- result -----" << std::endl;
    for(int i = 0; i < output_size; i++){
        if (interpreter->tensor(output)->type == kTfLiteFloat32) {
            std::cout << "[" << i << "]:  " << 
                interpreter->typed_tensor<float>(input)[i] << " -> " <<
                interpreter->typed_output_tensor<float>(0)[i] << std::endl;
        }
    }

    std::cout << ".................." << std::endl;

    return 0;
}



TfLiteStatus ASinPrepare(TfLiteContext* context, TfLiteNode* node) {
  using namespace tflite;
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* input = GetInput(context, node, 0);
  TfLiteTensor* output = GetOutput(context, node, 0);

  int num_dims = NumDimensions(input);

  TfLiteIntArray* output_size = TfLiteIntArrayCreate(num_dims);
  for (int i=0; i<num_dims; ++i) {
    output_size->data[i] = input->dims->data[i];
  }

  return context->ResizeTensor(context, output, output_size);
}

TfLiteStatus ASinEval(TfLiteContext* context, TfLiteNode* node) {
  using namespace tflite;
  const TfLiteTensor* input = GetInput(context, node,0);
  TfLiteTensor* output = GetOutput(context, node,0);

  float* input_data = input->data.f;
  float* output_data = output->data.f;

  size_t count = 1;
  int num_dims = NumDimensions(input);
  for (int i = 0; i < num_dims; ++i) {
    count *= input->dims->data[i];
  }

  for (size_t i=0; i<count; ++i) {
    output_data[i] = asin(input_data[i]);
  }
  return kTfLiteOk;
}

TfLiteRegistration* Register_ASIN() {
  static TfLiteRegistration r = {nullptr, nullptr, ASinPrepare, ASinEval};
  return &r;
}