import torch
import torch.nn  as nn
class RNN(nn.Module):
	def __init__(self,input_size,hidden_size,num_layers,num_classes,device):
		super(RNN, self).__init__()
		self.device=device
		self.hidden_size=hidden_size
		self.num_layers=num_layers
		self.rnn=nn.RNN(input_size,hidden_size,num_layers,batch_first=True)
		self.fc=nn.Linear(hidden_size,num_classes)
	def forward(self, x):
		#初始化隐状态h和细胞状态c
		x=x.float()
		x=x.to(self.device)
		h0=torch.zeros(self.num_layers,x.size(0),self.hidden_size).to(self.device)
		# c0=torch.zeros(self.num_layers,x.size(0),self.hidden_size).to(self.device)
		# print('h0:',h0.shape)        #h0: torch.Size([2, 32, 12])
		#print('c0:',c0.shape)       #c0: torch.Size([2, 32, 12])
		#反向传播
		out,_ = self.rnn(x,h0)
		# print('h',h)
	   # print('c',c)
		#最后一步全连接输出
		out=self.fc(out[:,-1,:])
		# print(out.shape)    #torch.Size([32, 6])
		return out

	# def __init__(self, input_size, hidden_size, num_layers, num_classes,device):
	# 	super(RNN, self).__init__()
	# 	self.device=device
	# 	self.hidden_size = hidden_size
	# 	self.num_layers = num_layers
    #     self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
	# def forward(self, x):
	# 	x = x.float()
	# 	h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
	# 	#c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
	# 	# 反向传播
	# 	out, _ = self.rnn(x, h0)
	# 	# 最后一步全连接输出
	# 	out = self.fc(out[:, -1, :])
	# 	return out

class RNN_Bidirectional(nn.Module):
	def __init__(self,input_size,hidden_size,num_layers,num_classes,device):
		super(RNN_Bidirectional, self).__init__()
		self.device=device
		self.hidden_size=hidden_size
		self.num_layers=num_layers
		self.rnn=nn.RNN(input_size,hidden_size,num_layers,batch_first=True,bidirectional=True)
		self.fc=nn.Linear(hidden_size*2,num_classes)
	def forward(self, x):
		#初始化隐状态h和细胞状态c
		x=x.float()
		x=x.to(self.device)
		h0=torch.zeros(self.num_layers*2,x.size(0),self.hidden_size).to(self.device)
		# c0=torch.zeros(self.num_layers*2,x.size(0),self.hidden_size).to(self.device)
		# print('h0:',h0.shape)        #h0: torch.Size([2, 32, 12])
		#print('c0:',c0.shape)       #c0: torch.Size([2, 32, 12])
		#反向传播
		out,_ = self.rnn(x,h0)
		# print('h',h)
	   # print('c',c)
		#最后一步全连接输出
		out=self.fc(out[:,-1,:])
		# print(out.shape)    #torch.Size([32, 6])
		return out

class GRU(nn.Module):
	def __init__(self,input_size,hidden_size,num_layers,num_classes,device):
		super(GRU, self).__init__()
		self.device=device
		self.hidden_size=hidden_size
		self.num_layers=num_layers
		self.rnn=nn.GRU(input_size,hidden_size,num_layers,batch_first=True)
		self.fc=nn.Linear(hidden_size,num_classes)
	def forward(self, x):
		#初始化隐状态h和细胞状态c
		x=x.float()
		x=x.to(self.device)
		h0=torch.zeros(self.num_layers,x.size(0),self.hidden_size).to(self.device)
		# c0=torch.zeros(self.num_layers,x.size(0),self.hidden_size).to(self.device)
		# print('h0:',h0.shape)        #h0: torch.Size([2, 32, 12])
		#print('c0:',c0.shape)       #c0: torch.Size([2, 32, 12])
		#反向传播
		out,_ = self.rnn(x,h0)
		# print('h',h)
	   # print('c',c)
		#最后一步全连接输出
		out=self.fc(out[:,-1,:])
		# print(out.shape)    #torch.Size([32, 6])
		return out

class GRU_Bidirectional(nn.Module):
	def __init__(self,input_size,hidden_size,num_layers,num_classes,device):
		super(GRU_Bidirectional, self).__init__()
		self.device=device
		self.hidden_size=hidden_size
		self.num_layers=num_layers
		self.rnn=nn.GRU(input_size,hidden_size,num_layers,batch_first=True,bidirectional=True)
		self.fc=nn.Linear(hidden_size*2,num_classes)
	def forward(self, x):
		#初始化隐状态h和细胞状态c
		x=x.float()
		x=x.to(self.device)
		h0=torch.zeros(self.num_layers*2,x.size(0),self.hidden_size).to(self.device)
		# c0=torch.zeros(self.num_layers*2,x.size(0),self.hidden_size).to(self.device)
		# print('h0:',h0.shape)        #h0: torch.Size([2, 32, 12])
		#print('c0:',c0.shape)       #c0: torch.Size([2, 32, 12])
		#反向传播
		out,_= self.rnn(x,h0)
		# print('h',h)
	   # print('c',c)
		#最后一步全连接输出
		out=self.fc(out[:,-1,:])
		# print(out.shape)    #torch.Size([32, 6])
		return out


#build RNN
class LSTM(nn.Module):
	def __init__(self,input_size,hidden_size,num_layers,num_classes,device):
		super(LSTM, self).__init__()
		self.device=device
		self.hidden_size=hidden_size
		self.num_layers=num_layers
		self.rnn=nn.LSTM(input_size,hidden_size,num_layers,batch_first=True)
		self.fc=nn.Linear(hidden_size,num_classes)
	def forward(self, x):
		#初始化隐状态h和细胞状态c
		x=x.float()
		x=x.to(self.device)
		h0=torch.zeros(self.num_layers,x.size(0),self.hidden_size).to(self.device)
		c0=torch.zeros(self.num_layers,x.size(0),self.hidden_size).to(self.device)
		# print('h0:',h0.shape)        #h0: torch.Size([2, 32, 12])
		#print('c0:',c0.shape)       #c0: torch.Size([2, 32, 12])
		#反向传播
		out,(h,c) = self.rnn(x,(h0,c0))
		# print('h',h)
	   # print('c',c)
		#最后一步全连接输出
		out=self.fc(out[:,-1,:])
		# print(out.shape)    #torch.Size([32, 6])
		return out

class LSTM_Bidirectional(nn.Module):
	def __init__(self,input_size,hidden_size,num_layers,num_classes,device):
		super(LSTM_Bidirectional, self).__init__()
		self.device=device
		self.hidden_size=hidden_size
		self.num_layers=num_layers
		self.rnn=nn.LSTM(input_size,hidden_size,num_layers,batch_first=True,bidirectional=True)
		self.fc=nn.Linear(hidden_size*2,num_classes)
	def forward(self, x):
		#初始化隐状态h和细胞状态c
		x=x.float()
		x=x.to(self.device)
		h0=torch.zeros(self.num_layers*2,x.size(0),self.hidden_size).to(self.device)
		c0=torch.zeros(self.num_layers*2,x.size(0),self.hidden_size).to(self.device)
		# print('h0:',h0.shape)        #h0: torch.Size([2, 32, 12])
		#print('c0:',c0.shape)       #c0: torch.Size([2, 32, 12])
		#反向传播
		out,(h,c) = self.rnn(x,(h0,c0))
		# print('h',h)
	   # print('c',c)
		#最后一步全连接输出
		out=self.fc(out[:,-1,:])
		# print(out.shape)    #torch.Size([32, 6])
		return out

def Get_RNN_Models(input_size,hidden_size,num_layers,num_classes,device):
	models={}
	models["单向RNN"]=RNN(input_size,hidden_size,num_layers,num_classes,device)
	models["双向RNN"] = RNN_Bidirectional(input_size, hidden_size, num_layers, num_classes, device)
	models["单向GRU"] = GRU(input_size, hidden_size, num_layers, num_classes, device)
	models["双向GRU"] = GRU_Bidirectional(input_size, hidden_size, num_layers, num_classes, device)
	models["单向LSTM"] = LSTM(input_size, hidden_size, num_layers, num_classes, device)
	models["双向LSTM"] = LSTM_Bidirectional(input_size, hidden_size, num_layers, num_classes, device)
	return models

for x in Get_RNN_Models(1,1,1,1,1).items():
	print(x[1])