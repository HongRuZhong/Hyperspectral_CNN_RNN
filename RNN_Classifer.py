'''用RNN跑高光譜數據'''
import Hyperspectral_CNN.Image_Split as Ima_S
import Hyperspectral_CNN.Hyper_Classifer_RNNModel as RNNModel
import torch
import torch.nn  as nn
import torch.optim as optim
import numpy as np
import sklearn.metrics as metrics
import os
import time
import warnings
warnings.filterwarnings("ignore")
from torch.utils.tensorboard import SummaryWriter
import My_PythonLib as MP
import PyTorch_Tool_Py.classification_train as train_tool
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

def Check_Data():
	train_loader,test_loader=Ima_S.Get_IPData_RNNData(5)
	for data,label in train_loader:
		print(data.shape)


def RNN_All_Param_Run():
	'''利用RNN对高光谱数据进行分类，所有参数一次输出'''
	opath = "D:\\ZHR_USE\\torchmodel\\RNNHyperspectral\\Results\\IndianPines_AllResults\\"
	epochs=301
	best_acc_dic={} #记录过程中的所有模型的最佳精度
	for s in range(3,31,2):
		train_loader, test_loader = Ima_S.Get_IPData_RNNData(s) #读入数据
		# print("train data xinxi:",len(train_loader))
		# for x,y in train_loader:
		# 	print(x.shape,y.shape)
		# exit()
		# train_loader, test_loader = Ima_S.Get_IPData_RNNData(s)
		for hidden in range(50,51,1): #不同神经元遍历
			models=RNNModel.Get_RNN_Models(10,hidden,2,16,device)
			for x in models.items():#不同模型遍历
				model_name=x[0]
				sub_path = os.path.join(opath,model_name+"\\timestep_"+str(s)+"神经元个数_"+str(hidden))
				if not os.path.exists(sub_path):
					os.makedirs(sub_path)
				start_time=time.time()
				net=x[1]

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
				best_acc_dic[model_name+"\ttimestep:"+str(s)+"\t神经元个数:"+str(hidden)]=ACC
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




def Hyperspectral_RNN():
	'''使用RNN对高光谱数据进行分类'''
	opath="D:\\ZHR_USE\\torchmodel\\RNNHyperspectral\\Results\\双向RNN_Salinas\\"
	writer=SummaryWriter(opath)
	epochs=301
	train_loader,test_loader=Ima_S.Get_Salinas_Dataloader_RNNdata(9)
	#参数
	i_size,hidden_size,num_layers,num_classes=10,50,2,16
	# net=RNNModel.RNN(i_size,hidden_size,num_layers,num_classes,device)
	net = RNNModel.RNN_Bidirectional(i_size, hidden_size, num_layers, num_classes, device)
	# net = RNNModel.LSTM(i_size, hidden_size, num_layers, num_classes, device)
	# net = RNNModel.GRU_Bidirectional(i_size, hidden_size, num_layers, num_classes, device)
	net.to(device)
	dataiter = iter(train_loader)
	hydata, label = dataiter.next()
	writer.add_graph(net,(hydata.float(),))
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(net.parameters(), lr=0.01)
	test_iter_num=0

	#记录中间结果
	train_loss=[]
	train_predict_label=0
	train_true_label=0
	test_loss=[]
	test_predict_label=0
	test_true_label=0
	test_iter_num=0
	ACC=0
	for epoch in range(epochs):
		train_loss_one_epoch,train_predict_label,train_true_label=train_tool.train_one_epoch(net,
																							 criterion,optimizer,train_loader,device,epoch,50,writer)
		train_loss.append(train_loss_one_epoch)
		if epoch%10==0:
			test_iter_num+=1
			test_loss_one_epoch,test_predict_label,test_true_label=train_tool.evaluate(net,criterion,test_loader,device,test_iter_num,writer)
			test_loss.append(test_loss_one_epoch)
			ACC_one_epoch = metrics.accuracy_score(test_true_label, test_predict_label)
			if ACC_one_epoch > ACC:
				Out_Result(opath, net, train_predict_label, train_true_label, test_predict_label, test_true_label)
				ACC = ACC_one_epoch
	#输出最终的评价
	# print(train_loss)
	train_loss = np.array(train_loss)
	np.savetxt(os.path.join(opath, "train_loss.csv"), train_loss, delimiter=',', fmt="%.04f")
	test_loss = np.array(test_loss)
	np.savetxt(os.path.join(opath, "test_loss.csv"), test_loss, delimiter=',', fmt="%.04f")
	writer.close()
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

