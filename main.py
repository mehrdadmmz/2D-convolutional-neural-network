import numpy as np 
import matplotlib.pyplot as plt

from skimage import data
from skimage.color import rgb2gray

class Conv2D:
    def __init__(self, W, padding=(0, 0), stride=(1, 1), dilation=(1, 1)):
        """
        inputs: 
        W        --> Kernel weights; shape is (C_out, C_in, kH, kW)
        padding  --> padding as a tuple (pad_height, pad_width)
        stride   --> stride as a tuple (stride_height, stride_width)
        dilation --> dilation as a tuple (dilation_height, dilation_width)
        """

        # Convert kernel to numpy array and flip it along height and width for convolution
        self.W = np.array(W)[:, :, ::-1, ::-1]
        self.padding = padding
        self.stride = stride
        self.dilation = dilation

    def forward(self, X):
        """
        inputs: 
        X --> input array; shape can be (C_in, H, W) for a single image or (N, C_in, H, W)
        for a batch of images
               
        return: 
        Numpy array of shape (C_out, H_out, W_out) for a single image or 
        (N, C_out, H_out, W_out) for a batched input
        
        """
        X = np.array(X)
        
        # If the input is a single image, add a batch dimension
        if X.ndim == 3:
            X = X[np.newaxis, :]
        
        N, C_in, H, W_in = X.shape
        C_out, C_in_w, kH, kW = self.W.shape
        assert C_in == C_in_w, "Number of input channels in X and W must match"
        
        # Effective kernel size considering dilation
        kH_eff = (kH - 1) * self.dilation[0] + 1
        kW_eff = (kW - 1) * self.dilation[1] + 1
        
        # Calculate output spatial dimensions --> page 20 of Lec7 on CNNs
        H_out = (H + 2 * self.padding[0] - kH_eff) // self.stride[0] + 1
        W_out = (W_in + 2 * self.padding[1] - kW_eff) // self.stride[1] + 1
        
        # Pad the input along height and width, no padding for N and C_in
        X_padded = np.pad(X, ((0, 0), (0, 0), (self.padding[0], self.padding[0]), (self.padding[1], self.padding[1])), mode="constant")
        
        # Initialize the output tensor
        output = np.zeros((N, C_out, H_out, W_out))
        
        # Loop over the batch, output channels, and spatial dimensions
        for n in range(N):
            for oc in range(C_out):
                for i in range(H_out):
                    for j in range(W_out):
                        conv_sum = 0
                        for ic in range(C_in):
                            for kh in range(kH):
                                for kw in range(kW):
                                    # Compute the indices considering stride and dilation
                                    # output(i, j) = sum over kH=m and kW=n of input(i * s + m * d, j * s + n * d) . kernel(m, n)
                                    h_idx = i * self.stride[0] + kh * self.dilation[0]
                                    w_idx = j * self.stride[1] + kw * self.dilation[1]
                                    conv_sum += X_padded[n, ic, h_idx, w_idx] * self.W[oc, ic, kh, kw]
                        output[n, oc, i, j] = conv_sum
                        
        # Remove batch dimension if input was a single image
        if output.shape[0] == 1:
            output = output[0]
            
        return output


if __name__ == "__main__": 
    # Gaussian kernels of size 3 and 2
    gaus_3 = np.array([[1, 2, 1], 
                       [2, 4, 2], 
                       [1, 2, 1]], dtype=np.float32)
    gaus_3 = gaus_3 / gaus_3.sum()
    gaus_2 = np.ones((2, 2), dtype=np.float32) / 4
    
    # Sobel kernels of size 3 and 2
    sobel_3 = np.array([[-1, -2, -1], 
                        [0, 0, 0], 
                        [1, 2, 1]], dtype=np.float32)
    sobel_2 = np.array([[-1, -1], 
                        [1, 1]], dtype=np.float32)
    
    # Sharpening kernels of size 3 and 2
    sharpen_3 = np.array([[0, -1, 0], 
                          [-1, 5, -1], 
                          [0, -1, 0]], dtype=np.float32)
    sharpen_2 = np.array([[2, -1], 
                          [-1, 2]], dtype=np.float32)
    
    # Dictionary of dictionaries for different kernels
    filters = {
        "Gaussian Blur": {3: gaus_3, 2: gaus_2}, 
        "Sobel":         {3: sobel_3, 2: sobel_2}, 
        "Sharpening":    {3: sharpen_3, 2: sharpen_2},
    }
    
    # Settings for testing (a, b, and c)
    settings = {
        "a": {"kernel_size": 3, "stride": (1, 1), "dilation": (1, 1), "padding": (1, 1)}, 
        "b": {"kernel_size": 2, "stride": (2, 2), "dilation": (1, 1), "padding": (0, 0)},
        "c": {"kernel_size": 3, "stride": (1, 1), "dilation": (2, 2), "padding": (2, 2)},
    }
    
    # Load and normalize the image
    rgb_image = data.astronaut()  # shape (H, W, 3)
    rgb_image = rgb_image.astype(np.float32) / 255.0
    
    gray_image = rgb2gray(rgb_image)  # shape (H, W)
    gray_image = gray_image[np.newaxis, :, :]  # shape (1, H, W)
    
    # Plot original images both in RGB and grayscale
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 24))
    
    ax[0].imshow(rgb_image)
    ax[0].set_title("RGB Original Image")
    
    # Squeeze the gray image to remove the extra channel dimension
    ax[1].imshow(np.squeeze(gray_image), cmap='gray')
    ax[1].set_title("Gray Original Image")
    
    plt.show()
    
    # Plotting filtered outputs
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 12))
    fig.suptitle("Filtered Outputs (Row: Filter Type, Columns: Settings)", fontsize=16)
    
    filter_names = list(filters.keys())
    setting_names = list(settings.keys())
    
    for i, filter_name in enumerate(filter_names):
        for j, setting_name in enumerate(setting_names): 
            params = settings[setting_name]
            k_size = params["kernel_size"]
            stride = params["stride"]
            dilation = params["dilation"]
            padding = params["padding"]
            
            kernel = filters[filter_name][k_size]
            W = kernel.reshape(1, 1, k_size, k_size)
            
            CNN_2D = Conv2D(W=W, padding=padding, stride=stride, dilation=dilation)
            
            output = CNN_2D.forward(gray_image)
            output = np.squeeze(output, axis=0)  # Remove the extra channel dimension
            
            # Get the appropriate subplot axis
            ax = axes[i, j]
            ax.imshow(output, cmap='gray')
            ax.set_title(f"{filter_name}\n{setting_name}\nShape: {output.shape}", fontsize=8)
            ax.axis("off")
