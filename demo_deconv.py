import torch
import numpy as np
import cv2
from numpy.linalg import norm
import matplotlib.pyplot as plt
from PIL import Image
from utils.iterative_scheme import iterative_scheme
from utils.utils_test import shift_inv_metrics
from models.network_p4ip import P4IP_Net
from models.network_p4ip_denoiser import P4IP_Denoiser
import os

K_IDX = 3; IM_IDX = 3;


ALPHA = 10.0 # Photon level for noise


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
np.random.seed(20)



# # Load Non-Blind Solver
MODEL_FILE = 'model_zoo/p4ip_100epoch.pth'
p4ip = P4IP_Net(n_iters = 8); p4ip.load_state_dict(torch.load(MODEL_FILE))
p4ip.to(device); p4ip.eval()
# # Load Poisson Denoiser
MODEL_FILE = 'model_zoo/denoiser_p4ip_100epoch.pth'
denoiser = P4IP_Denoiser(n_iters = 8); denoiser.load_state_dict(torch.load(MODEL_FILE))
denoiser.to(device); denoiser.eval()

out_path = "./out/"
in_path = "./mydata/"
max_len = 500
# 获取路径下所有图片文件
imglist = [file for file in os.listdir(in_path) if file.endswith(".jpg") or file.endswith(".png")]

for imgName in imglist:
	img = cv2.imread(in_path + imgName)
	a = max(img.shape[0], img.shape[1])
	# 降采样
	if(a > max_len):
		scale = max_len / a
		img = cv2.resize(img, dsize=None, fx=scale, fy=scale)

	# 将RGB通道转换到第0维
	img_ = img.swapaxes(0, 2).swapaxes(1, 2)

	res = []
	for i in range(3):
		yn = np.asarray(img_[i], dtype=np.float32)
		yn = yn/255.0*ALPHA
		# y = np.random.poisson(np.maximum(ALPHA*yn,0)).astype(np.float32)
		x_list, k_list = iterative_scheme(yn, ALPHA, p4ip, denoiser, {'VERBOSE': True})
		x_blind = x_list[-1]
		res.append((x_blind * 255.0).clip(0, 255))
	res = np.array(res)
	res = res.swapaxes(0, 2).swapaxes(0, 1)
	cv2.imwrite(out_path + imgName, res)

# x = np.asarray(Image.open('data/im'+str(IM_IDX)+'.png'), dtype=np.float32)
# x = x/255.0
# kernel = np.asarray(Image.open('data/kernel'+str(K_IDX)+'.png'), dtype=np.float32)
# kernel = kernel/np.sum(kernel.ravel())
# if yn.ndim > 2:
# 	yn = np.mean(yn,axis=2)
# 	x = np.mean(x,axis=2)
# Noisy + Blurred Image here
# y = np.random.poisson(np.maximum(ALPHA*yn,0)).astype(np.float32)

# psnr_val, ssim_val = shift_inv_metrics(x_blind, x)
# print('Blind Deconv. PSNR / SSIM: %0.2f / %0.3f'%(psnr_val, ssim_val))
# plt.figure(figsize = (18,6))

# plt.subplot(1,3,1); plt.imshow(y/ALPHA, cmap='gray'); plt.axis('off')
# plt.title('Noisy, Blurred')

# plt.subplot(1,3,2); plt.imshow(x_blind, cmap='gray'); plt.axis('off')
# plt.title('Reconstruction')

# plt.subplot(1,3,3); plt.imshow(x, cmap='gray'); plt.axis('off')
# plt.title('Ground Truth')
# plt.show()
