# Implementation-of-filter
## Aim:
To implement filters for smoothing and sharpening the images in the spatial domain.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
### Step1
Import necessary libraries: OpenCV, NumPy, and Matplotlib.Read an image, convert it to RGB format, define an 11x11 averaging kernel, and apply 2D convolution filtering.Display the original and filtered images side by side using Matplotlib
</br>
</br> 

### Step2
Decide on the type of filter you want to apply based on your desired outcome. Some common filters include:

a. Averaging filter

b. Gaussian filter

c. Median filter

d. Laplacian filter
</br>
</br> 

### Step3
A filter kernel is a small matrix that is applied to each pixel in the image to produce the filtered result. The size and values of the kernel determine the filter's behavior. For example, an averaging filter kernel has all elements equal to 1/N, where N is the kernel size.
</br>
</br> 

### Step4
Use the library's functions to apply the filter to the image. The filtering process typically involves convolving the image with the filter kernel.
</br>
</br> 

### Step5
Visualize the filtered image using a suitable method (e.g., OpenCV's imshow, Matplotlib). Save the filtered image to a file if needed.
</br>
</br> 

## Program:
```
Developed By   : ROSELIN MARY JOVITA S
Register Number : 212222230122
```
</br>

### 1. Smoothing Filters

#### i) Using Averaging Filter
```Python

import cv2
import numpy as np
import matplotlib.pyplot as plt
image1 = cv2.imread('husky.jpg')
image2 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

kernel = np.ones((11,11), np. float32)/121
image3 = cv2.filter2D(image2, -1, kernel)

plt.figure(figsize=(9,9))
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title('Orignal')
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(image3)
plt.title('Filtered')
plt.axis('off')

```
### Output
![Screenshot 2024-10-03 114544](https://github.com/user-attachments/assets/489002c8-2489-41d1-9476-8b42ad52fba1)

#### ii) Using Weighted Averaging Filter
```Python


kernel2 = np.array([[1,2,1],[2,4,2],[1,2,1]])/16
image4 = cv2.filter2D(image2, -1, kernel2)
plt.imshow(image4)
plt.title('Weighted Averaging Filtered')

```
### Output
![Screenshot 2024-10-03 114557](https://github.com/user-attachments/assets/43b750a6-9b4b-4476-9fa5-5b6ddbfc89a7)


#### iii) Using Gaussian Filter
```Python

gaussian_blur = cv2.GaussianBlur(src=image2, ksize=(11,11), sigmaX=0, sigmaY=0)
plt.imshow(gaussian_blur)
plt.title(' Gaussian Blurring Filtered')

```
### Output
![Screenshot 2024-10-03 114605](https://github.com/user-attachments/assets/c4167947-c86e-44d9-bd8e-5e23335c93bc)


#### iv)Using Median Filter
```Python

median=cv2.medianBlur (src=image2, ksize=11)
plt.imshow(median)
plt.title(' Median Blurring Filtered')

```
### Output
![Screenshot 2024-10-03 114615](https://github.com/user-attachments/assets/e37a3f0c-f073-44c5-ad44-e5a5ac9d08c7)


### 2. Sharpening Filters
#### i) Using Laplacian Linear Kernal
```Python

kernel3 = np.array([[0,1,0], [1, -4,1],[0,1,0]])
image5 =cv2.filter2D(image2, -1, kernel3)
plt.imshow(image5)
plt.title('Laplacian Kernel')
```
### Output

![Screenshot 2024-10-03 114629](https://github.com/user-attachments/assets/7c7c4fe7-2b8f-4686-8e97-d6f59e404fea)

#### ii) Using Laplacian Operator
```Python

new_image = cv2.Laplacian (image2, cv2.CV_64F)
plt.imshow(new_image)
plt.title('Laplacian Operator')

```
### Output

![Screenshot 2024-10-03 114640](https://github.com/user-attachments/assets/d79938a8-9c9a-487a-944e-2a3a6389fe00)



## Result:
Thus the filters are designed for smoothing and sharpening the images in the spatial domain.
