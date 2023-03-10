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

