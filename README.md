# ECE277_FinalProject

This is the Final Project belonging to LY Li and HY Li.

In this project, we use CUDA to realize some classical and useful Digital Image Processing Algorithms. With the GPU, we can process high resolution image fastly (We provide two images to test). We also provide the algorithms implemented using CPU. Comparing with the GPU version, it is much slower!

We implement two algorithms:
- Sharpening algorithm
- DCT and IDCT (using DCT to compress an image)

To compile our project, you should run
```
pip install -e .
```

Hardware Support:
- Intel(R) Core(TM) i5-8300H CPU @ 2.30GHz   2.30 GHz
- NVIDIA GeForce GTX 1050 Ti

Software Support:
- CUDA
- Pycharm Community
- Visual Studio 2019 Community