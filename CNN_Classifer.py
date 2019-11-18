import Hyperspectral_CNN.Image_Split as Img_S
import numpy as np
import os
import time
import torch
import My_PythonLib as MP
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset,DataLoader
from torchvision import datasets,transforms,utils,models
import sklearn.metrics as metrics
# from tensorboardX import SummaryWriter
import Hyperspectral_CNN.Hyper_Classifer_CNNModel as CNNModel
import PyTorch_Tool_Py.classification_train as train_tool
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
MNIST_path="D:\\Data\\Data_Picture_useML\\Data_MNIST"
CIFAR10_path='D:\\Data\\Data_Picture_useML\\Data_CIFAR\\'
Summary_path="C:\\Users\\仲鸿儒\\PycharmProjects\\untitled\\MNIST_RUN\\Summary\\"

epochs=20

# writer=SummaryWriter()

# class NET(nn.Module):
nB=10
nT=16



def Run():
	#tensorboard 可视化
	input_to_model=torch.rand([64,1,28,28])

	train_loader,test_loader=Img_Split.Get_Salinas_Dataloader()
	print("Load end...")
	net=NET(nT).to(device)
	# writer.add_graph(net, input_to_model)
	# writer.close()
	criterion=nn.CrossEntropyLoss()
	optimizer=optim.SGD(net.parameters(),lr=0.01,momentum=0.9)
	for epoch in range(epochs):
		# for image,label in train_loader:
			# print(image)
		train_tool.train_one_epoch(net,criterion,optimizer,train_loader,device,epoch,print_freq=50)
		if epoch%5==0:
			print("test")
			train_tool.evaluate(net,criterion,test_loader,device)
def qianyixuexi():
	train_loader,test_loader=Img_S.Get_Dataloader()
	model_ft = models.resnet18()
	model_ft.load_state_dict(torch.load('D:\\Learn_Pytorch\\renet_model\\resnet18-5c106cde.pth'))
	monum_ftrs = model_ft.fc.in_features
	model_ft.fc = nn.Linear(monum_ftrs, 10)
	model_ft = model_ft.to(device)
	criterion = nn.CrossEntropyLoss()
	optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
	# decay LR by a factor of 0.1 every 7 epochs
	exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
	for epoch in range(epochs):
		# for image,label in train_loader:
			# print(image)
		train_tool.train_one_epoch(model_ft,criterion,optimizer_ft,train_loader,device,epoch,print_freq=50)
		exp_lr_scheduler.step()
		train_tool.evaluate(model_ft,criterion,test_loader,device)

def CNN_All_Param_Run():
	'''利用CNN对高光谱数据进行分类，所有参数一次输出'''
	opath = "D:\\ZHR_USE\\torchmodel\\RNNHyperspectral\\Results\\IndianPines_AllResults\\"
	epochs=301
	best_acc_dic={} #记录过程中的所有模型的最佳精度
	for s in range(3,31,2):
		train_loader, test_loader = Img_S.Get_Dataloader(s) #读入数据
		# print("train data xinxi:",len(train_loader))
		# for x,y in train_loader:
		# 	print(x.shape,y.shape)
		# exit()
		# train_loader, test_loader = Ima_S.Get_IPData_RNNData(s)

		net=CNNModel.Net_HyperS_CNN_Inception(16,10)
		sub_path = os.path.join(opath,"邻域大小_"+str(s))
		if not os.path.exists(sub_path):
			os.makedirs(sub_path)
		start_time=time.time()
		net.to(device)
		criterion = nn.CrossEntropyLoss()
		optimizer = optim.Adam(net.parameters(), lr=0.01)
		test_iter_num = 0
		# 记录中间结果
		train_loss = []
		train_predict_label = 0
		train_true_label = 0
		test_loss = []
		test_predict_label = 0
		test_true_label = 0
		test_iter_num = 0
		ACC = 0
		for epoch in range(epochs):
			train_loss_one_epoch, train_predict_label, train_true_label = train_tool.train_one_epoch(net,
																									 criterion,
																									 optimizer,
																									 train_loader,
																									 device,
																									 epoch, 50																										 )
			train_loss.append(train_loss_one_epoch)
			if epoch % 10 == 0:
				test_iter_num += 1
				test_loss_one_epoch, test_predict_label, test_true_label = train_tool.evaluate(net, criterion,
																							   test_loader,
																							   device,
																							   test_iter_num)
				test_loss.append(test_loss_one_epoch)
				ACC_one_epoch = metrics.accuracy_score(test_true_label, test_predict_label)
				print("精度ACC：",ACC_one_epoch)
				if ACC_one_epoch > ACC:
					Out_Result(sub_path, net, train_predict_label, train_true_label, test_predict_label,
							   test_true_label,epoch)
					ACC = ACC_one_epoch
		best_acc_dic["邻域大小"+str(s)]=ACC
		#输出花费时间
		fp=open(os.path.join(sub_path,"spent_time.txt"),'w')
		print("spent time: "+str(time.time()-start_time),file=fp)
		fp.close()
		# 输出最终的评价
		# print(train_loss)
		train_loss = np.array(train_loss)
		np.savetxt(os.path.join(sub_path, "train_loss.csv"), train_loss, delimiter=',', fmt="%.04f")
		test_loss = np.array(test_loss)
		np.savetxt(os.path.join(sub_path, "test_loss.csv"), test_loss, delimiter=',', fmt="%.04f")
	best_acc_list=best_acc_dic.items()
	best_acc=np.row_stack(best_acc_list)
	np.savetxt(os.path.join(opath,"不同参数精度对比.txt"),best_acc,fmt='%s',delimiter="\t")

def Out_Result(opath, net, train_predict_label, train_true_label, test_predict_label, test_true_label,iter_No=None):
	# 输出最终的评价

	trainACC, train_kappa, train_report, train_cofM = MP.Result_Evaluate(train_predict_label, train_true_label)
	fp = open(os.path.join(opath, "train_report.txt"), 'w')
	print(("train_ACC:" + str(trainACC) + "\n"), file=fp)
	print(("train_kappa:" + str(train_kappa) + "\n"), file=fp)
	print(train_report, file=fp)
	train_cofM.to_csv(opath + "\\train混淆矩阵.csv")
	testACC, test_kappa, test_report, test_cofM = MP.Result_Evaluate(test_predict_label, test_true_label)
	fp = open(os.path.join(opath, "test_report.txt"), 'w')
	if iter_No!=None:
		print(("第几次迭代：",iter_No),file=fp)
	print(("test_ACC:" + str(testACC) + "\n"), file=fp)
	print(("test_kappa:" + str(test_kappa) + "\n"), file=fp)
	print(test_report, file=fp)
	test_cofM.to_csv(opath + "\\test混淆矩阵.csv")
	#save label 用于画图
	label=np.column_stack((test_true_label,test_predict_label))
	np.savetxt(os.path.join(opath,"测试集预测结果标签.txt"),label,fmt='%d',delimiter='\t')
	# Save model
	torch.save(net, opath + "\\model整体.pkl")
	torch.save(net.state_dict(), opath + "\\model_param.pkl")

if __name__ == '__main__':
	Run()
	# qianyixuexi()