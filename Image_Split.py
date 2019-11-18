'''将高光谱图像分裂为每个像素为M*N*N的子块,M为像素个数'''
import numpy as np
import os
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import time


def Load_Data(datapath,nR,nC):
	data=np.loadtxt(datapath,skiprows=2)[:,2:]
	# print(data[0:5])
	data[:,0:-1]=StandardScaler().fit_transform(data[:,0:-1])
	# print(data[0:5])
	dr=data.shape[0]
	dc=data.shape[1]
	# print(dr,dc)
	all_img_list=[]
	#每一列波段变换为图像
	for c in range(dc-1):
		onecol=data[:,c]
		oneimg=np.reshape(onecol,(nR,nC))
		all_img_list.append(oneimg)
	label_img=np.reshape(data[:,-1],(nR,nC))
	all_img=np.array(all_img_list)
	return all_img,label_img

def Pad_Img(img,pad_size):
	'''对图像进行pad填充'''
	# img=img.astype(np.int)
	img=torch.Tensor(img)

	img=torch.unsqueeze(img,0)
	# print(img.shape)
	tpad=nn.ReflectionPad2d(int(pad_size))  #要4维才行
	img_pad=tpad(img)
	# dim=(pad_size,pad_size,pad_size,pad_size)
	# img_pad=F.pad(img,dim,"reflect")
	# print(img_pad.shape)
	return np.squeeze(img_pad)

# def Split_Img(attri_img,label_img,Sub_block_size):
# 	'''对图像切块，切成Sub_block_size的大小'''
# 	nb=attri_img.shape[0]
# 	nr=label_img.shape[0]
# 	nc=label_img.shape[1]
# 	sub_block_list=[]
# 	label_list=[]
# 	for i in range(nr):
# 		for j in range(nc):
# 			# print(j)
# 			if label_img[i, j] != 0:
# 				label_list.append(label_img[i,j])
# 				band_list = []
# 				for b in range(nb):
# 					band_list.append(attri_img[b,i:i+Sub_block_size,j:j+Sub_block_size])
# 				onepixel_sun_block=np.array(band_list)
# 				# print(onepixel_sun_block.shape)
# 				sub_block_list.append(onepixel_sun_block)
#
#
# 	attri_img_last=np.array(sub_block_list)
# 	return attri_img_last,label_list

def Split_Img(attri_img,label_img,train_rc,Sub_block_size):
	'''对图像切块，切成Sub_block_size的大小'''
	train_rc=train_rc.tolist()
	nb=attri_img.shape[0]
	nr=label_img.shape[0]
	nc=label_img.shape[1]
	sub_block_list=[]
	label_list=[]
	train_list=[]
	train_label_list=[]
	for i in range(nr):
		x_index = np.arange(i, i + Sub_block_size, 1)
		for j in range(nc):
			y_index=np.arange(j,j+Sub_block_size,1)
			indices=np.ix_(x_index,y_index) #笛卡尔内积
			# print(j)
			if label_img[i, j] != 0:
				label_list.append(label_img[i,j]-1)
				band_list = []
				for b in range(nb):
					band_list.append(attri_img[b][indices])
					# band_list.append(attri_img[b,i:i+Sub_block_size,j:j+Sub_block_size])
				onepixel_sun_block=np.array(band_list)
				if [i,j] in train_rc:
					train_list.append(onepixel_sun_block)
					train_label_list.append(label_img[i, j]-1)
				# print(onepixel_sun_block.shape)
				sub_block_list.append(onepixel_sun_block)

	train_img=np.array(train_list)
	attri_img_last=np.array(sub_block_list)
	return train_img,train_label_list,attri_img_last,label_list

class TrainSet(Dataset):
	def __init__(self, data,label):
		# 定义好 image 的路径
		self.data, self.label = data, label

	def __getitem__(self, index):
		return self.data[index], self.label[index]

	def __len__(self):
		return len(self.data)

def Pytorch_Format(attri_img,label_list,train=False):
	dataset=TrainSet(attri_img,label_list)
	loader=0
	if train is True:
		loader=DataLoader(dataset=dataset,batch_size=64,shuffle=True,num_workers=0)
	else:
		loader = DataLoader(dataset=dataset, batch_size=64, shuffle=False, num_workers=0)
	return loader
	# for tx,ty in trainloader:
	# 	# plt.imshow(tx[0,0].numpy())
	# 	# plt.show()
	# 	# exit()
	# 	print(tx.shape)

def Get_Dataloader(Sub_Block_size):
	nR = 145
	nC = 145
	# Sub_Block_size = 31
	Pad_size = int(Sub_Block_size - 1) / 2
	ipath = "D:\\ZHR_USE\\torchmodel\\RNNHyperspectral\\"
	train_rc = np.loadtxt(ipath + "Indian_pines_FCwithRC_训练集.txt", skiprows=2, usecols=(0, 1), dtype=np.int)
	train_rc = train_rc.tolist()
	attri_img, label_img = Load_Data(ipath + "Indian_pines_FCwithRC_Select10Band.txt", nR, nC)
	train_rc = np.loadtxt(ipath + "Indian_pines_FCwithRC_训练集.txt", skiprows=2, usecols=(0, 1))
	plt.imshow(attri_img[2])
	plt.show()
	Pad_attri_img = Pad_Img(attri_img, Pad_size).numpy()
	train_img, train_label, test_img, test_label = Split_Img(Pad_attri_img, label_img, train_rc, Sub_Block_size)

	trainloader=Pytorch_Format(train_img, train_label,True)
	testloader=Pytorch_Format(test_img,test_label)
	return trainloader,testloader
def Get_Salinas_Dataloader(Sub_Block_size):
	nR = 512
	nC = 217
	# Sub_Block_size = 51
	Pad_size = int(Sub_Block_size - 1) / 2
	ipath = "D:\\ZHR_USE\\torchmodel\\RNNHyperspectral\\"
	train_rc = np.loadtxt(ipath + "Salinas_400样本训练集.txt", skiprows=2, usecols=(0, 1), dtype=np.int)
	train_rc = train_rc.tolist()
	attri_img, label_img = Load_Data(ipath + "Salinas_FCwithRC_Select10Band.txt", nR, nC)
	train_rc = np.loadtxt(ipath + "Salinas_400样本训练集.txt", skiprows=2, usecols=(0, 1))
	plt.imshow(attri_img[2])
	plt.show()
	Pad_attri_img = Pad_Img(attri_img, Pad_size).numpy()
	train_img, train_label, test_img, test_label = Split_Img(Pad_attri_img, label_img, train_rc, Sub_Block_size)
	trainloader=Pytorch_Format(train_img, train_label,True)
	testloader=Pytorch_Format(test_img,test_label)
	return trainloader,testloader

def Get_Salinas_Dataloader_RNNdata(Sub_Block_size):
	nR = 512
	nC = 217
	Pad_size = int(Sub_Block_size - 1) / 2
	ipath = "D:\\ZHR_USE\\torchmodel\\RNNHyperspectral\\"
	train_rc = np.loadtxt(ipath + "Salinas_400样本训练集.txt", skiprows=2, usecols=(0, 1), dtype=np.int)
	train_rc = train_rc.tolist()
	attri_img, label_img = Load_Data(ipath + "Salinas_FCwithRC_Select10Band.txt", nR, nC)
	train_rc = np.loadtxt(ipath + "Salinas_400样本训练集.txt", skiprows=2, usecols=(0, 1))
	# plt.imshow(attri_img[2])
	# plt.show()
	Pad_attri_img = Pad_Img(attri_img, Pad_size).numpy()
	train_img, train_label, test_img, test_label = Split_Img(Pad_attri_img, label_img, train_rc, Sub_Block_size)
	#将n*n的矩阵转化为（n*n）*1的序列数据
	train_img=train_img.transpose(0,2,3,1)
	train_img=train_img.reshape(train_img.shape[0],Sub_Block_size*Sub_Block_size,10)
	test_img=test_img.transpose(0,2,3,1)
	test_img=test_img.reshape(test_img.shape[0],Sub_Block_size*Sub_Block_size,10)
	trainloader=Pytorch_Format(train_img, train_label,True)
	testloader=Pytorch_Format(test_img,test_label)
	return trainloader,testloader

def Get_IPData_RNNData(Sub_Block_size):
	nR = 145
	nC = 145
	Pad_size = int(Sub_Block_size - 1) / 2
	ipath = "D:\\ZHR_USE\\torchmodel\\RNNHyperspectral\\"
	train_rc = np.loadtxt(ipath + "Indian_pines_FCwithRC_训练集.txt", skiprows=2, usecols=(0, 1), dtype=np.int)
	train_rc = train_rc.tolist()
	attri_img, label_img = Load_Data(ipath + "Indian_pines_FCwithRC_Select10Band.txt", nR, nC)
	train_rc = np.loadtxt(ipath + "Indian_pines_FCwithRC_训练集.txt", skiprows=2, usecols=(0, 1))
	# plt.imshow(attri_img[2])
	# plt.show()
	Pad_attri_img = Pad_Img(attri_img, Pad_size).numpy()
	train_img, train_label, test_img, test_label = Split_Img(Pad_attri_img, label_img, train_rc, Sub_Block_size)
	#将n*n的矩阵转化为（n*n）*1的序列数据
	train_img=train_img.transpose(0,2,3,1)
	train_img=train_img.reshape(train_img.shape[0],Sub_Block_size*Sub_Block_size,10)
	test_img=test_img.transpose(0,2,3,1)
	test_img=test_img.reshape(test_img.shape[0],Sub_Block_size*Sub_Block_size,10)
	trainloader=Pytorch_Format(train_img, train_label,True)
	testloader=Pytorch_Format(test_img,test_label)
	return trainloader,testloader

if __name__=="__main__":

	# IPselect_col_10=[-2,-1,9, 27, 35, 39, 65, 125, 128, 149, 176, 178,-3]

	t=time.time()
	nR=145
	nC=145
	Sub_Block_size=7
	Pad_size=int(Sub_Block_size-1)/2
	ipath="D:\Data\Hyperspectral\\"
	salpath="D:\Data\Salinas_Variogram&Auto_Correlation3\每个类型_所有变量_相关图_卷积特征结果\\"
	ippath="D:\Data\IndianPine_Variogram&Auto_Correlation3\每个类型_所有变量_相关图_卷积特征结果\\"
	train_rc = np.loadtxt(ipath + "Indian_pines_FCwithRC_训练集.txt", skiprows=2, usecols=(0, 1),dtype=np.int)
	train_rc=train_rc.tolist()
	attri_img,label_img=Load_Data(ipath+"Indian_pines_FCwithRC_Select10Band.txt",nR,nC)
	train_rc=np.loadtxt(ipath+"Indian_pines_FCwithRC_训练集.txt",skiprows=2,usecols=(0,1))
	plt.imshow(attri_img[2])
	plt.show()
	Pad_attri_img=Pad_Img(attri_img,Pad_size).numpy()
	train_img,train_label,test_img,test_label=Split_Img(Pad_attri_img,label_img,train_rc,Sub_Block_size)
	test_img=test_img.transpose(0,2,3,1)
	print(test_img.shape)
	test_img=test_img.reshape(test_img.shape[0],7*7,10)
	print(test_img[0][0],'\n',test_img[0][1],'\n',test_img[0][2])
	print(test_img.shape)
	print(attri_img.shape,test_label)
	print("time",time.time()-t)
	Pytorch_Format(train_img,train_label)
	plt.imshow(Pad_attri_img[0][0].numpy())
	plt.show()