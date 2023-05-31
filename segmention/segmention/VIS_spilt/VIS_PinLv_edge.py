'''
Descripttion: 
version: 
Author: lyh
Date: 2023-03-16 10:10:35
LastEditors: smile
LastEditTime: 2023-04-17 14:31:02
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread(r'spilt_img\spilt_fusion\VIS_spilt\00000.png', 0)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))
rows, cols = img.shape
crow, ccol = rows // 2, cols // 2
fshift[crow-10:crow+10, ccol-10:ccol+10] = 0  # remove low-frequency components
epsilon = 1e-10
magnitude_spectrum_hp = 20*np.log(np.abs(fshift) + epsilon)
f_ishift = np.fft.ifftshift(fshift)
img_filtered = np.fft.ifft2(f_ishift)
img_filtered = np.abs(img_filtered)
threshold = 5
img_contours = cv2.threshold(img_filtered, threshold, 255, cv2.THRESH_TOZERO)[1]
fig, axs = plt.subplots(1, 1, figsize=(10, 5))
axs.imshow(img_contours, cmap='gray')
plt.imshow(img_contours, cmap='gray')
cv2.imwrite('spilt_img/spilt_fusion/VIS_spilt/00000VIS_pinyu.png', img_contours*255)
axs.set_title('Edge Contours')
plt.show()
