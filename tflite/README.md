### How to build

- ${CC} is g++ or aarch-linux-gnu-g++ etc.

```bash
$ export TF_ROOT=<path to root of tensorflow source>

$ ${CC} main.cpp \
-std=c++11 \
-I${TF_ROOT} \
-I${TF_ROOT}/tensorflow \
-I${TF_ROOT}/tensorflow/contrib/lite/tools/make/downloads \
-I${TF_ROOT}/tensorflow/contrib/lite/tools/make/downloads/eigen \
-I${TF_ROOT}/tensorflow/contrib/lite/tools/make/downloads/absl \
-I${TF_ROOT}/tensorflow/contrib/lite/tools/make/downloads/gemmlowp \
-I${TF_ROOT}/tensorflow/contrib/lite/tools/make/downloads/neon_2_sse \
-I${TF_ROOT}/tensorflow/contrib/lite/tools/make/downloads/farmhash/src \
-I${TF_ROOT}/tensorflow/contrib/lite/tools/make/downloads/flatbuffers/include \
libtensorflow-lite.a -lpthread -ldl \
-o linear.out
```


***
### Tips

- For the list of default registered ops, see _tensorflow/contrib/lite/kernels/register.cc_

- Pretrained model can be found at: https://www.tensorflow.org/lite/models
