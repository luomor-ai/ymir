Yolo v4, v3 and v2 for Windows and Linux
(neural networks for object detection)
Paper YOLO v4: https://arxiv.org/abs/2004.10934

Paper Scaled YOLO v4: https://arxiv.org/abs/2011.08036 use to reproduce results: ScaledYOLOv4

More details in articles on medium:

Scaled_YOLOv4
YOLOv4
Manual: https://github.com/AlexeyAB/darknet/wiki

Discussion:

Reddit
Google-groups
Discord
About Darknet framework: http://pjreddie.com/darknet/

Darknet Continuous Integration CircleCI TravisCI Contributors License: Unlicense DOI arxiv.org arxiv.org colab colab

YOLOv4 model zoo

Requirements (and how to install dependencies)

Pre-trained models

FAQ - frequently asked questions

Explanations in issues

Yolo v4 in other frameworks (TensorRT, TensorFlow, PyTorch, OpenVINO, OpenCV-dnn, TVM,...)

Datasets

Yolo v4, v3 and v2 for Windows and Linux

(neural networks for object detection)
GeForce RTX 2080 Ti
Youtube video of results
How to evaluate AP of YOLOv4 on the MS COCO evaluation server
How to evaluate FPS of YOLOv4 on GPU
Pre-trained models
Requirements for Windows, Linux and macOS
Yolo v4 in other frameworks
Datasets
Improvements in this repository
How to use on the command line
For using network video-camera mjpeg-stream with any Android smartphone
How to compile on Linux/macOS (using CMake)
Using also PowerShell
How to compile on Linux (using make)
How to compile on Windows (using CMake)
How to compile on Windows (using vcpkg)
How to train with multi-GPU
How to train (to detect your custom objects)
How to train tiny-yolo (to detect your custom objects)
When should I stop training
Custom object detection
How to improve object detection
How to mark bounded boxes of objects and create annotation files
How to use Yolo as DLL and SO libraries
Citation

Darknet Logo

scaled_yolov4 AP50:95 - FPS (Tesla V100) Paper: https://arxiv.org/abs/2011.08036

modern_gpus AP50:95 / AP50 - FPS (Tesla V100) Paper: https://arxiv.org/abs/2004.10934

tkDNN-TensorRT accelerates YOLOv4 ~2x times for batch=1 and 3x-4x times for batch=4.

tkDNN: https://github.com/ceccocats/tkDNN
OpenCV: https://gist.github.com/YashasSamaga/48bdb167303e10f4d07b754888ddbdcf
GeForce RTX 2080 Ti
Network Size	Darknet, FPS (avg)	tkDNN TensorRT FP32, FPS	tkDNN TensorRT FP16, FPS	OpenCV FP16, FPS	tkDNN TensorRT FP16 batch=4, FPS	OpenCV FP16 batch=4, FPS	tkDNN Speedup
320	100	116	202	183	423	430	4.3x
416	82	103	162	159	284	294	3.6x
512	69	91	134	138	206	216	3.1x
608	53	62	103	115	150	150	2.8x
Tiny 416	443	609	790	773	1774	1353	3.5x
Tiny 416 CPU Core i7 7700HQ	3.4	-	-	42	-	39	12x
Yolo v4 Full comparison: map_fps
Yolo v4 tiny comparison: tiny_fps
CSPNet: paper and map_fps comparison: https://github.com/WongKinYiu/CrossStagePartialNetworks
Yolo v3 on MS COCO: Speed / Accuracy (mAP@0.5) chart
Yolo v3 on MS COCO (Yolo v3 vs RetinaNet) - Figure 3: https://arxiv.org/pdf/1804.02767v1.pdf
Yolo v2 on Pascal VOC 2007: https://hsto.org/files/a24/21e/068/a2421e0689fb43f08584de9d44c2215f.jpg
Yolo v2 on Pascal VOC 2012 (comp4): https://hsto.org/files/3a6/fdf/b53/3a6fdfb533f34cee9b52bdd9bb0b19d9.jpg
Youtube video of results
Yolo v4	Scaled Yolo v4
Others: https://www.youtube.com/user/pjreddie/videos

How to evaluate AP of YOLOv4 on the MS COCO evaluation server
Download and unzip test-dev2017 dataset from MS COCO server: http://images.cocodataset.org/zips/test2017.zip
Download list of images for Detection tasks and replace the paths with yours: https://raw.githubusercontent.com/AlexeyAB/darknet/master/scripts/testdev2017.txt
Download yolov4.weights file 245 MB: yolov4.weights (Google-drive mirror yolov4.weights )
Content of the file cfg/coco.data should be
classes= 80
train  = <replace with your path>/trainvalno5k.txt
valid = <replace with your path>/testdev2017.txt
names = data/coco.names
backup = backup
eval=coco
Create /results/ folder near with ./darknet executable file
Run validation: ./darknet detector valid cfg/coco.data cfg/yolov4.cfg yolov4.weights
Rename the file /results/coco_results.json to detections_test-dev2017_yolov4_results.json and compress it to detections_test-dev2017_yolov4_results.zip
Submit file detections_test-dev2017_yolov4_results.zip to the MS COCO evaluation server for the test-dev2019 (bbox)
How to evaluate FPS of YOLOv4 on GPU
Compile Darknet with GPU=1 CUDNN=1 CUDNN_HALF=1 OPENCV=1 in the Makefile
Download yolov4.weights file 245 MB: yolov4.weights (Google-drive mirror yolov4.weights )
Get any .avi/.mp4 video file (preferably not more than 1920x1080 to avoid bottlenecks in CPU performance)
Run one of two commands and look at the AVG FPS:
include video_capturing + NMS + drawing_bboxes: ./darknet detector demo cfg/coco.data cfg/yolov4.cfg yolov4.weights test.mp4 -dont_show -ext_output
exclude video_capturing + NMS + drawing_bboxes: ./darknet detector demo cfg/coco.data cfg/yolov4.cfg yolov4.weights test.mp4 -benchmark
Pre-trained models
There are weights-file for different cfg-files (trained for MS COCO dataset):

FPS on RTX 2070 (R) and Tesla V100 (V):

yolov4-p6.cfg - 1280x1280 - 72.1% mAP@0.5 (54.0% AP@0.5:0.95) - 32(V) FPS - xxx BFlops (xxx FMA) - 487 MB: yolov4-p6.weights

pre-trained weights for training: https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-p6.conv.289
yolov4-p5.cfg - 896x896 - 70.0% mAP@0.5 (51.6% AP@0.5:0.95) - 43(V) FPS - xxx BFlops (xxx FMA) - 271 MB: yolov4-p5.weights

pre-trained weights for training: https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-p5.conv.232
yolov4-csp-x-swish.cfg - 640x640 - 69.9% mAP@0.5 (51.5% AP@0.5:0.95) - 23(R) FPS / 50(V) FPS - 221 BFlops (110 FMA) - 381 MB: yolov4-csp-x-swish.weights

pre-trained weights for training: https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-csp-x-swish.conv.192
yolov4-csp-swish.cfg - 640x640 - 68.7% mAP@0.5 (50.0% AP@0.5:0.95) - 70(V) FPS - 120 (60 FMA) - 202 MB: yolov4-csp-swish.weights

pre-trained weights for training: https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-csp-swish.conv.164
yolov4x-mish.cfg - 640x640 - 68.5% mAP@0.5 (50.1% AP@0.5:0.95) - 23(R) FPS / 50(V) FPS - 221 BFlops (110 FMA) - 381 MB: yolov4x-mish.weights

pre-trained weights for training: https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4x-mish.conv.166
yolov4-csp.cfg - 202 MB: yolov4-csp.weights paper Scaled Yolo v4

just change width= and height= parameters in yolov4-csp.cfg file and use the same yolov4-csp.weights file for all cases:

width=640 height=640 in cfg: 67.4% mAP@0.5 (48.7% AP@0.5:0.95) - 70(V) FPS - 120 (60 FMA) BFlops
width=512 height=512 in cfg: 64.8% mAP@0.5 (46.2% AP@0.5:0.95) - 93(V) FPS - 77 (39 FMA) BFlops
pre-trained weights for training: https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-csp.conv.142
yolov4.cfg - 245 MB: yolov4.weights (Google-drive mirror yolov4.weights ) paper Yolo v4 just change width= and height= parameters in yolov4.cfg file and use the same yolov4.weights file for all cases:

width=608 height=608 in cfg: 65.7% mAP@0.5 (43.5% AP@0.5:0.95) - 34(R) FPS / 62(V) FPS - 128.5 BFlops
width=512 height=512 in cfg: 64.9% mAP@0.5 (43.0% AP@0.5:0.95) - 45(R) FPS / 83(V) FPS - 91.1 BFlops
width=416 height=416 in cfg: 62.8% mAP@0.5 (41.2% AP@0.5:0.95) - 55(R) FPS / 96(V) FPS - 60.1 BFlops
width=320 height=320 in cfg: 60% mAP@0.5 ( 38% AP@0.5:0.95) - 63(R) FPS / 123(V) FPS - 35.5 BFlops
yolov4-tiny.cfg - 40.2% mAP@0.5 - 371(1080Ti) FPS / 330(RTX2070) FPS - 6.9 BFlops - 23.1 MB: yolov4-tiny.weights

enet-coco.cfg (EfficientNetB0-Yolov3) - 45.5% mAP@0.5 - 55(R) FPS - 3.7 BFlops - 18.3 MB: enetb0-coco_final.weights

yolov3-openimages.cfg - 247 MB - 18(R) FPS - OpenImages dataset: yolov3-openimages.weights

CLICK ME - Yolo v3 models
CLICK ME - Yolo v2 models
Put it near compiled: darknet.exe

You can get cfg-files by path: darknet/cfg/

Requirements for Windows, Linux and macOS
CMake >= 3.18: https://cmake.org/download/
Powershell (already installed on windows): https://docs.microsoft.com/en-us/powershell/scripting/install/installing-powershell
CUDA >= 10.2: https://developer.nvidia.com/cuda-toolkit-archive (on Linux do Post-installation Actions)
OpenCV >= 2.4: use your preferred package manager (brew, apt), build from source using vcpkg or download from OpenCV official site (on Windows set system variable OpenCV_DIR = C:\opencv\build - where are the include and x64 folders image)
cuDNN >= 8.0.2 https://developer.nvidia.com/rdp/cudnn-archive (on Linux copy cudnn.h,libcudnn.so... as described here https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#installlinux-tar , on Windows copy cudnn.h,cudnn64_7.dll, cudnn64_7.lib as described here https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#installwindows )
GPU with CC >= 3.0: https://en.wikipedia.org/wiki/CUDA#GPUs_supported
Yolo v4 in other frameworks
Pytorch - Scaled-YOLOv4: https://github.com/WongKinYiu/ScaledYOLOv4
TensorFlow: pip install yolov4 YOLOv4 on TensorFlow 2.0 / TFlite / Android: https://github.com/hunglc007/tensorflow-yolov4-tflite Official TF models: https://github.com/tensorflow/models/tree/master/official/vision/beta/projects/yolo For YOLOv4 - convert yolov4.weights/cfg files to yolov4.pb by using TNTWEN project, and to yolov4.tflite TensorFlow-lite
OpenCV the fastest implementation of YOLOv4 for CPU (x86/ARM-Android), OpenCV can be compiled with OpenVINO-backend for running on (Myriad X / USB Neural Compute Stick / Arria FPGA), use yolov4.weights/cfg with: C++ example or Python example
Intel OpenVINO 2021.2: supports YOLOv4 (NPU Myriad X / USB Neural Compute Stick / Arria FPGA): https://devmesh.intel.com/projects/openvino-yolov4-49c756 read this manual (old manual ) (for Scaled-YOLOv4 models use https://github.com/Chen-MingChang/pytorch_YOLO_OpenVINO_demo )
PyTorch > ONNX:
WongKinYiu/PyTorch_YOLOv4
maudzung/3D-YOLOv4
Tianxiaomo/pytorch-YOLOv4
YOLOv5
ONNX on Jetson for YOLOv4: https://developer.nvidia.com/blog/announcing-onnx-runtime-for-jetson/ and https://github.com/ttanzhiqiang/onnx_tensorrt_project
nVidia Transfer Learning Toolkit (TLT>=3.0) Training and Detection https://docs.nvidia.com/metropolis/TLT/tlt-user-guide/text/object_detection/yolo_v4.html
TensorRT+tkDNN: https://github.com/ceccocats/tkDNN#fps-results
Deepstream 5.0 / TensorRT for YOLOv4 https://github.com/NVIDIA-AI-IOT/yolov4_deepstream or https://github.com/marcoslucianops/DeepStream-Yolo read Yolo is natively supported in DeepStream 4.0 and PDF. Additionally jkjung-avt/tensorrt_demos or wang-xinyu/tensorrtx
Triton Inference Server / TensorRT https://github.com/isarsoft/yolov4-triton-tensorrt
DirectML https://github.com/microsoft/DirectML/tree/master/Samples/yolov4
OpenCL (Intel, AMD, Mali GPUs for macOS & GNU/Linux) https://github.com/sowson/darknet
HIP for Training and Detection on AMD GPU https://github.com/os-hackathon/darknet
ROS (Robot Operating System) https://github.com/engcang/ros-yolo-sort
Xilinx Zynq Ultrascale+ Deep Learning Processor (DPU) ZCU102/ZCU104: https://github.com/Xilinx/Vitis-In-Depth-Tutorial/tree/master/Machine_Learning/Design_Tutorials/07-yolov4-tutorial
Amazon Neurochip / Amazon EC2 Inf1 instances 1.85 times higher throughput and 37% lower cost per image for TensorFlow based YOLOv4 model, using Keras URL
TVM - compilation of deep learning models (Keras, MXNet, PyTorch, Tensorflow, CoreML, DarkNet) into minimum deployable modules on diverse hardware backend (CPUs, GPUs, FPGA, and specialized accelerators): https://tvm.ai/about
Tencent/ncnn: the fastest inference of YOLOv4 on mobile phone CPU: https://github.com/Tencent/ncnn
OpenDataCam - It detects, tracks and counts moving objects by using YOLOv4: https://github.com/opendatacam/opendatacam#-hardware-pre-requisite
Netron - Visualizer for neural networks: https://github.com/lutzroeder/netron
Datasets
MS COCO: use ./scripts/get_coco_dataset.sh to get labeled MS COCO detection dataset
OpenImages: use python ./scripts/get_openimages_dataset.py for labeling train detection dataset
Pascal VOC: use python ./scripts/voc_label.py for labeling Train/Test/Val detection datasets
ILSVRC2012 (ImageNet classification): use ./scripts/get_imagenet_train.sh (also imagenet_label.sh for labeling valid set)
German/Belgium/Russian/LISA/MASTIF Traffic Sign Datasets for Detection - use this parsers: https://github.com/angeligareta/Datasets2Darknet#detection-task
List of other datasets: https://github.com/AlexeyAB/darknet/tree/master/scripts#datasets
Improvements in this repository
developed State-of-the-Art object detector YOLOv4
added