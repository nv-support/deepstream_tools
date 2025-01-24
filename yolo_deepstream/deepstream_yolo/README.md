# Deploy YOLO Models With DeepStream #

**This sample shows how to integrate YOLO models with customized output layer parsing for detected objects with DeepStreamSDK.**

## 1. Sample contents: ##
- `deepstream_app_config_yolo.txt`: DeepStream reference app configuration file for using YOLO models as the primary detector.
- `config_infer_primary_yoloV4.txt`: Configuration file for the GStreamer nvinfer plugin for the YoloV4 detector model.
- `config_infer_primary_yoloV7.txt`: Configuration file for the GStreamer nvinfer plugin for the YoloV7 detector model.
- `config_infer_primary_yoloV8.txt`: Configuration file for the GStreamer nvinfer plugin for the YoloV8 detector model.
- `config_infer_primary_yoloV8_dla.txt`: Configuration file for the GStreamer nvinfer plugin for the YoloV8 detector model running with DLA on Jetson.
- `config_infer_primary_yoloV9.txt`: Configuration file for the GStreamer nvinfer plugin for the YoloV9 detector model.
- `nvdsinfer_custom_impl_Yolo/nvdsparsebbox_Yolo.cpp`: Output layer parsing function for detected objects for the Yolo models.
- `nvdsinfer_custom_impl_Yolo/nvdsparsebbox_Yolo_cuda.cu`: Output layer parsing function for detected objects for the Yolo models by CUDA.

## 2. Download and Build ##

```sh
  $ cd ~/
  $ git clone https://github.com/NVIDIA-AI-IOT/yolo_deepstream.git
  $ cd ~/yolo_deepstream/deepstream_yolo/nvdsinfer_custom_impl_Yolo
  $ make
  $ cd ..
```

## 3. Prepare model and Run with deepstream-app ##

### TIPS: DeepStream 6.1.1+ is required ###

#### Yolov4 

- Go to this pytorch repository <https://github.com/Tianxiaomo/pytorch-YOLOv4> where you can convert YOLOv4 Pytorch model into **ONNX**
- Other famous YOLOv4 pytorch repositories as references:
  - <https://github.com/WongKinYiu/PyTorch_YOLOv4>
  - <https://github.com/bubbliiiing/yolov4-pytorch>
  - <https://github.com/maudzung/Complex-YOLOv4-Pytorch>
  - <https://github.com/AllanYiin/YoloV4>
- Or you can download reference ONNX model directly from here ([link](https://nvidia.box.com/s/achcifjwl1ac99tdvfwtfmxgec5d0pro)).  

- Run
  ```bash
  $ deepstream-app -c deepstream_app_config_yolo.txt
  ```
  The output result will output to `yolo.mp4`

#### YOLOv7
- Follow the guide https://github.com/WongKinYiu/yolov7#export, export a dynamic-batch-1-output onnx-model

```bash
$ python export.py --weights ./yolov7.pt --grid --simplify --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640 --dynamic-batch
```

- Or use the qat model exported from [yolov7_qat](../yolov7_qat/README.md)

- Generate the TensorRT engine with the following command:

```bash
$ /usr/src/tensorrt/bin/trtexec --onnx=yolov7_qat_640.onnx --int8 --fp16 --minShapes=images:1x3x640x640 --optShapes=images:8x3x640x640 --maxShapes=images:16x3x640x640 --saveEngine=yolov7_qat_640_gpu_b16.engine
```

- Run
```bash
$ deepstream-app -c deepstream_app_config_yolo.txt -t
```
The output result will output to `yolo.mp4`

#### YOLOv8
##### 1. Run on GPU with int8 precision with calibrated model(GPU Only)
- Convert model with trtexec

  The model can be found in [model-list](../README.md#model-list)

  ```bash
  # batchsize = 16
  $ /usr/src/tensorrt/bin/trtexec --onnx=yolov8s_640_dynamic.onnx --fp16 --int8 --verbose --calib=yolov8s_gpu_precision_config_calib.cache --saveEngine=yolov8s_ptq_640_gpu_b16.engine --minShapes=x.1:16x3x640x640 --optShapes=x.1:16x3x640x640 --maxShapes=x.1:16x3x640x640 --precisionConstraints=obey --layerPrecisions=Split_36:fp16,Reshape_37:fp16,Transpose_38:fp16,Softmax_39:fp16,Conv_41:fp16,Sub_64:fp16,Concat_65:fp16,Mul_67:fp16,Sigmoid_68:fp16,Concat_69:fp16
  ```

- Enable the configure on config files, edit [deepstream_app_config_yolo.txt](./deepstream_app_config_yolo.txt)
  ```ini
  # batch-size = 16
  config-file=config_infer_primary_yoloV8.txt
  ```

##### 2. Run on DLA with our DLA-spec-finetuned model(DLA Only)
- Download sample ONNX models and convert the ONNX model to TensorRT engine 

  Download model and coresponding calibration file from [model-list](../README.md#model-list)

  Choose one of the following methods

  * Convert model with trtexec

      ```bash
      $ /usr/src/tensorrt/bin/trtexec --onnx=yolov8s_DAT_640_noqdq.onnx --fp16 --int8 --verbose --calib=yolov8s_DAT_precision_config_calib.cache --precisionConstraints=obey --layerPrecisions=Split_36:fp16,Reshape_37:fp16,Transpose_38:fp16,Softmax_39:fp16,Conv_41:fp16,Sub_64:fp16,Concat_65:fp16,Mul_67:fp16,Sigmoid_68:fp16,Concat_69:fp16 --saveEngine=yolov8s_DAT_640_noqdq_DLA.engine --useDLACore=0 --allowGPUFallback
      ```

  * Run the download and build script

    ```bash
    ./build_DLA_engine.sh
    ```

- Enable the configuration for yolov8 in config files, edit [deepstream_app_config_yolo.txt](./deepstream_app_config_yolo.txt)

  ```ini
  ...
  num-sources=1
  ...
  config-file=config_infer_primary_yoloV8_dla.txt
  [streammux]
  ...
  batch-size=1
  ...

  [primary-gie]
  ...
  batch-size=1
  ...
  ```

  Edit [config_infer_primary_yoloV8_dla.txt](./config_infer_primary_yoloV8_dla.txt)
  ```
  model-engine-file=yolov8s_DAT_640_noqdq_DLA.engine
  ...
  enable-dla=1
  use-dla-core=1 # or 0
  ```

##### 3. Run
  ```bash
  $ deepstream-app -c deepstream_app_config_yolo.txt
  ```
  The output result will output to `yolo.mp4`

#### YOLOv9
- Get ONNX model by one of the following methods
  * Follow the guide https://github.com/WongKinYiu/yolov9/issues/2#issuecomment-1960519506, export a dynamic-batch-1-output onnx-model
    ```bash
    $ python export.py --weights yolov9-s-converted.pt --dynamic --include onnx
    ```
  * Download the ONNX model directly from [yolov9-s onnx model](https://nvidia.box.com/s/dzch7bx0xlap4hoc5nk9huy72w33wbc9)
- Put the ONNX model file under yolo_deepstream/deepstream_yolo directory
- Modify the [deepstream_app_config_yolo.txt](./deepstream_app_config_yolo.txt) file
  ```ini
  config-file=config_infer_primary_yoloV9.txt
  ```
- Run
  ```bash
  deepstream-app -c deepstream_app_config_yolo.txt -t
  ```
  The output result will output to `yolo.mp4`

#### YOLOv8/YOLOv9 TIPS

 - How to export yolov8 models with official repo
 Follow the guide https://github.com/ultralytics/ultralytics/tree/v8.2.103, export a dynamic-batch-1-output onnx-model

    ```python
    from ultralytics import YOLO
    # Load a model
    model = YOLO("yolov8s.pt")  # load a pretrained model (recommended for training)
    path = model.export(format="onnx")  # export the model to ONNX format
    ```

 - For yolov8 and yolov9 models user must add a transpose node at the end of the network through [onnx-graphsurgeon](https://github.com/NVIDIA/TensorRT/tree/release/9.2/tools/onnx-graphsurgeon) :
    ```bash
    $ python append_transpose_yolov8_v9.py
    ```

- Run
```bash
deepstream-app -c deepstream_app_config_yolo.txt -t
```
The output result will output to `yolo.mp4`

## 4. CUDA Post Processing

this sample provide two ways of yolov7/yolov8/yolov9/yolov11 post-processing(decode yolo result, not include NMS), CPU version and GPU version
- CPU implement can be found in: [nvdsparsebbox_Yolo.cpp](nvdsinfer_custom_impl_Yolo/nvdsparsebbox_Yolo.cpp)
- CUDA implement can be found in: [nvdsparsebbox_Yolo_cuda.cu](nvdsinfer_custom_impl_Yolo/nvdsparsebbox_Yolo_cuda.cu)

Default will use CUDA-post processing. To enable CPU post-processing:
in [config_infer_primary_yoloV7.txt](config_infer_primary_yoloV7.txt), [config_infer_primary_yoloV8.txt](config_infer_primary_yoloV8.txt) or  [config_infer_primary_yoloV9.txt](config_infer_primary_yoloV9.txt)

- `parse-bbox-func-name=NvDsInferParseCustomYoloV7_cuda` -> `parse-bbox-func-name=NvDsInferParseCustomYoloV7`
- `disable-output-host-copy=1` -> `disable-output-host-copy=0`

The performance of the CPU-post-processing and CUDA-post-processing result can be found in [Performance](../#Performance)

