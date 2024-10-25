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

**Face Detection:**

![face_detect](https://github.com/user-attachments/assets/cb2f9c29-6b29-45b9-bd09-3a733b35c854)

**Swap:**

![face_swap](https://github.com/user-attachments/assets/1957ee68-8399-48b9-a10f-1ab8e3a49144)


## Needs

- transpose inswapper model output to source frame
- fine tune tensor array reshape functions.
- optimize and double check some post-processing functions.
- general UI quality of life improvement
