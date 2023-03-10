# AdaFNIO: Adaptive Fourier Neural Interpolation Operator for video frame interpolation 
### Hrishikesh Viswanath, Md Ashiqur Rahman, Rashmi Bhaskara, Aniket Bera
### [[arXiv](https://arxiv.org/abs/2211.10791)]

We present, AdaFNIO - Adaptive Fourier Neural Interpolation Operator, a neural operator-based architecture to perform video frame interpolation. 
Current deep learning based methods rely on local convolutions for feature learning and suffer from not being scale-invariant, thus requiring training data to 
be augmented through random flipping and re-scaling. On the other hand, AdaFNIO, learns the features in the frames, independent of input resolution, 
through token mixing and global convolution in the Fourier space or the spectral domain by using Fast Fourier Transform (FFT). 
We show that AdaFNIO can produce visually smooth and accurate results. To evaluate the visual quality of our interpolated frames, 
we calculate the structural similarity index (SSIM) and Peak Signal to Noise Ratio (PSNR) between the generated frame and the ground truth frame. 
We provide the quantitative performance of our model on Vimeo-90K dataset, DAVIS, UCF101 and DISFA+ dataset.

## Environment
- GPU: Nvidia A100
- Cuda: 11.6
- Python: 3.8
- pytorch: 1.9.0+cu111
- cupy: 8.6.0
- numpy: 1.23.3
- torchinfo: 1.7.0
- einops: 0.6.0
- matplotlib: 3.6.0
- pillow: 9.2.0
- sklearn: 1.1.2 
- torchvision: 0.10.0+cu111

## Citation
If you find the code helpful in your resarch or work, please cite the following paper.
```
@article{viswanath2022nio,
  title={NIO: Lightweight neural operator-based architecture for video frame interpolation},
  author={Viswanath, Hrishikesh and Rahman, Md Ashiqur and Bhaskara, Rashmi and Bera, Aniket},
  journal={arXiv preprint arXiv:2211.10791},
  year={2022}
}
```
