# noface

Rust implementation of deep face cam python project.

**disclaimer**: This project is still work in progress. Going into (hopefully) short hiatus to focus on another project. Currently preview is only sort of working. [Deep Live Cam](https://github.com/hacksider/Deep-Live-Cam) project is getting actively worked on if you want to check out what this project was trying to achieve.

## Requirements

Make sure OpenCV, Clang, and Onnx Runtime are properly configured and installed in your system. You will also need to provide models being used.

3 models required are (**det_10g.onnx**, **w600k_r50.onnx**,**inswapper_128.onnx**) from [insightface](https://github.com/deepinsight/insightface)

If you are wanting to use GPU with Cuda, make sure to set that up as well.

This projects core dependencies are

[opencv - rust binding](https://github.com/twistedfall/opencv-rust)

[ort - rust onnx runtime binding](https://github.com/pykeio/ort)

[insightface models](https://github.com/deepinsight/insightface)

They are amazing library with helpful contributors who helped me debug few issues.

## Progress

**Face Detection + Swap Cropped:**

![face_swap](https://github.com/user-attachments/assets/1957ee68-8399-48b9-a10f-1ab8e3a49144)
![face_swap_v2](https://github.com/user-attachments/assets/05842140-6eea-4232-b98f-f497a48ca1f4)


**Swap Action:**

![sample_2](https://github.com/user-attachments/assets/5f45a27f-6d1f-4e3e-abe2-de4f53cc4caf)


## Needs

- transpose inswapper model output to source frame
- fine tune tensor array reshape functions.
- optimize and double check some post-processing functions.
- general UI quality of life improvement
