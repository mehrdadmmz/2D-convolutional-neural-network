# 2D Convolution and Image Filtering

This repository demonstrates a simple **2D Convolution** implementation in Python, applied to an astronaut image from `skimage`. Various filters (Gaussian Blur, Sobel, Sharpening) are tested under different settings (kernel size, stride, dilation, padding).

  
## 1. Original Images

Below are the **RGB** and **Grayscale** versions of the astronaut image:

<p align="center"> 
<img width="474" alt="image" src="https://github.com/user-attachments/assets/9ca58f9d-448c-4347-a82b-7de52a882a33" />
</p>

  
## 2. Implementation Overview

- **Conv2D Class**: 
  - Accepts the kernel (`W`), along with padding, stride, and dilation parameters.
  - Performs convolution on a single image or a batch of images.
  - Returns the convolved output as a NumPy array.

- **Filters**:
  - **Gaussian Blur**: Smooths the image, reducing noise and high-frequency details.
  - **Sobel**: Highlights edges (vertical/horizontal) in the image.
  - **Sharpening**: Enhances edges and fine details, making the image crisper.

  
## 3. Filtered Outputs

We apply each filter under three different settings:

1. **Setting a**: 
   - Kernel Size = 3  
   - Stride = (1, 1)  
   - Dilation = (1, 1)  
   - Padding = (1, 1)  
   *Preserves spatial dimension; the effect of each filter is clearly visible.*
   <p align="center">
     <img width="236" alt="image" src="https://github.com/user-attachments/assets/37e6b8b9-5a85-41c1-aba8-b027f92261b4" />
     <img width="236" alt="image" src="https://github.com/user-attachments/assets/5c1d7ce1-f660-473a-9570-128137c68c25" />
     <img width="233" alt="image" src="https://github.com/user-attachments/assets/f3eca51f-e09c-4742-adbf-46c15fe50432" />
   </p>


2. **Setting b**: 
   - Kernel Size = 2  
   - Stride = (2, 2)  
   - Dilation = (1, 1)  
   - Padding = (0, 0)  
   *Downsamples the image (due to stride=2), resulting in a smaller output.*
   <p align="center"> 
   <img width="233" alt="image" src="https://github.com/user-attachments/assets/a0a65b9a-5f1c-43fa-92a7-17f2583a4827" />
   <img width="233" alt="image" src="https://github.com/user-attachments/assets/08ee2dc9-29a1-4595-857c-ff16a6a2c898" />
   <img width="234" alt="image" src="https://github.com/user-attachments/assets/c4717115-0dcb-4666-83da-e3a5363b631b" />
   </p>

3. **Setting c**: 
   - Kernel Size = 3  
   - Stride = (1, 1)  
   - Dilation = (2, 2)  
   - Padding = (2, 2)  
   *Uses dilated kernels and larger padding; captures a bigger context around each pixel while preserving size.*
   <p align="center">
   <img width="233" alt="image" src="https://github.com/user-attachments/assets/7d1628ad-d868-4477-a623-104a7eb4f78b" />
   <img width="235" alt="image" src="https://github.com/user-attachments/assets/6c84dfbe-45dd-4bfa-b274-31a8364ef250" />
   <img width="233" alt="image" src="https://github.com/user-attachments/assets/779d83c7-6c44-4fe9-aca3-8483db2687bc" />
   </p>

  
*(Paste the 3×3 grid of filtered outputs here, showing each filter across the three settings)*

  
## 4. Code Snippet

Below is the main code for reference:

<details>
<summary>Click to expand</summary>

```python
import numpy as np 
import matplotlib.pyplot as plt
from skimage import data
from skimage.color import rgb2gray

# --- Conv2D Class ---
class Conv2D:
    def __init__(self, W, padding=(0, 0), stride=(1, 1), dilation=(1, 1)):
        self.W = np.array(W)[:, :, ::-1, ::-1]
        self.padding = padding
        self.stride = stride
        self.dilation = dilation

    def forward(self, X):
        X = np.array(X)
        if X.ndim == 3:
            X = X[np.newaxis, :]
        N, C_in, H, W_in = X.shape
        C_out, C_in_w, kH, kW = self.W.shape
        assert C_in == C_in_w, "Mismatch in input channels."

        kH_eff = (kH - 1) * self.dilation[0] + 1
        kW_eff = (kW - 1) * self.dilation[1] + 1

        H_out = (H + 2 * self.padding[0] - kH_eff) // self.stride[0] + 1
        W_out = (W_in + 2 * self.padding[1] - kW_eff) // self.stride[1] + 1

        X_padded = np.pad(
            X, 
            ((0, 0), (0, 0), (self.padding[0], self.padding[0]), (self.padding[1], self.padding[1])),
            mode="constant"
        )
        output = np.zeros((N, C_out, H_out, W_out))

        for n in range(N):
            for oc in range(C_out):
                for i in range(H_out):
                    for j in range(W_out):
                        conv_sum = 0
                        for ic in range(C_in):
                            for kh in range(kH):
                                for kw in range(kW):
                                    h_idx = i * self.stride[0] + kh * self.dilation[0]
                                    w_idx = j * self.stride[1] + kw * self.dilation[1]
                                    conv_sum += (
                                        X_padded[n, ic, h_idx, w_idx] * 
                                        self.W[oc, ic, kh, kw]
                                    )
                        output[n, oc, i, j] = conv_sum

        if output.shape[0] == 1:
            output = output[0]
        return output
