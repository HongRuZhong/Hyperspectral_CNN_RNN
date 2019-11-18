import torch
import torch.nn as nn
import torch.nn.functional as F

# 编写卷积+bn+relu模块
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channals, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channals, **kwargs)
        self.bn = nn.BatchNorm2d(out_channals)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x)

# 编写Inception模块
class Inception(nn.Module): #这是一层的Inception结构
	def Nmm_conv(self,in_planes,out_c,k):
		conv_list=[]
		l=int(k/2.0)
		for x in range(l): #全部采用3x3卷积核去卷，5x5就是2个3x3
			conv_list.append(BasicConv2d(out_c,out_c,kernel_size=3,padding=1))

		block=nn.Sequential(
			BasicConv2d(in_planes, out_c, kernel_size=1), #先弄到要的维度
			*conv_list
		)
		return block

	def __init__(self, in_planes,
				 n1x1, n3x3,n5x5, n7x7,pool_planes):
		super(Inception, self).__init__()
		# 1x1 conv branch
		self.b1 = BasicConv2d(in_planes, n1x1, kernel_size=1)
		self.b3=self.Nmm_conv(in_planes,n3x3,3)
		self.b5 = self.Nmm_conv(in_planes, n5x5, 5)
		self.b7=self.Nmm_conv(in_planes,n7x7,7)
		# self.b11=self.Nmm_conv(in_planes,n11x11,11)
		self.b_pool = nn.MaxPool2d(3, stride=1, padding=1)
		self.bp = BasicConv2d(in_planes, pool_planes,
								  kernel_size=1)



	def forward(self, x):
		y1 = self.b1(x)
		y2 = self.b3(x)
		y3 = self.b5(x)
		y4 = self.b7(x)
		y5=self.bp(self.b_pool(x))
		# y的维度为[batch_size, out_channels, C_out,L_out]
		# 合并不同卷积下的特征图
		return torch.cat([y1, y2, y3, y4,y5], 1)

class Net_HyperS_CNN_Inception(nn.Module):
	def __init__(self,num_classes,nB):
		super(Net_HyperS_CNN_Inception,self).__init__()
		self.pre_layers=BasicConv2d(nB,32,kernel_size=3, padding=1)
		self.maxpool_1=nn.MaxPool2d(2,2)
		self.maxpool_2 = nn.MaxPool2d(2, 2)
		self.incep_1=Inception(nB,8,32,16,16,8)
		self.incep_2=Inception(8+32+16+16+8,16,64,32,32,16)

		self.lv = 80 * 7
		self.fc = self.FC_Layers(self.lv, num_classes)
	def FC_Layers(self, in_c, out_c):
		block = nn.Sequential(
			nn.Linear(in_c, 1024),
			nn.ReLU(),
			nn.Dropout(0.5),
			nn.Linear(1024, 256),
			nn.ReLU(),
			nn.Dropout(0.5),
			nn.Linear(256, out_c),
			# nn.Sigmoid()
		)
		return block
	def forward(self, input):
		x=self.pre_layers(input)
		x=self.incep_1(x)
		x=self.maxpool_1(x)
		x=self.incep_2(x)
		x=self.maxpool_2(x)
		# print("xxxx",x.size())
		x=x.view(x.size(0),self.lv)
		x=self.fc(x)
		return x
class NET(nn.Module):
	def CONV_Layers(self,in_channels,out_channels,kernel_size=3):
		block=nn.Sequential(
		nn.Conv2d(kernel_size=kernel_size,in_channels=in_channels,out_channels=out_channels),
		nn.ReLU(),
		nn.BatchNorm2d(out_channels),
		nn.MaxPool2d(2,2)
		)
		return block
	def CONV_2layers(self,in_channels,out_channels,kernel_size=3):
		block=nn.Sequential(
		nn.Conv2d(kernel_size=kernel_size,in_channels=in_channels,out_channels=in_channels),
		nn.ReLU(),
		nn.BatchNorm2d(in_channels),
		nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels),
		nn.ReLU(),
		nn.BatchNorm2d(out_channels),
		nn.MaxPool2d(2,2)
		)
		return block
	def __init__(self,num_classes,nB):
		super(NET,self).__init__()
		self.conv1=self.CONV_Layers(nB,32)
		self.conv2=self.CONV_2layers(32,64)

		self.lv=64*5*5
		self.fc1=nn.Linear(self.lv,1024)
		self.fc2=nn.Linear(1024,512)
		self.fc3=nn.Linear(512,num_classes)
		self.drop1=nn.Dropout2d()
		self.drop2=nn.Dropout2d()

	def forward(self, input):
		# print(input.shape)
		x=self.conv1(input)
		# print(x.shape)
		x=self.conv2(x)
		print(x.size(0),x.size(1),x.size(2),x.size(3))
		# lv=x.size(1)*x.size(2)*x.size(3)
		x=x.view(x.size(0),self.lv)
		# x=x.flatten(1)  #将4为张量后面的3维拉平
		x=F.relu(self.fc1(x))
		x=self.drop1(x)
		x=F.relu(self.fc2(x))
		x=self.drop2(x)
		x=self.fc3(x)
		return x