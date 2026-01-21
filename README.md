# HRM Snake

An implementation of Hierarchical Reasoning Model (https://arxiv.org/abs/2506.21734) for the game of Snake.

![output](https://github.com/user-attachments/assets/bc926bb1-504c-46c4-903a-1af1ab96a8b8)

## Usage

```
python3.10 hrm_snake.py
```

## Model Code

```py
class ConvNet(nn.Module):
	def __init__(self):
		super(ConvNet, self).__init__()
		self.conv1 = nn.Conv2d(4, 32, 3, 1)
		self.conv2 = nn.Conv2d(32, 64, 3, 1)
		self.fc = nn.Linear(64*16*16, internal_width)
	def forward(self, x):
		x = self.conv1(x)
		x = F.relu(x)
		x = self.conv2(x)
		x = F.relu(x)
		x = x.flatten(start_dim=1)
		x = self.fc(x)
		return x

class Net(nn.Module):

	def __init__(self):
		super(Net, self).__init__()

		self.n_embd = internal_width
		self.n_head = 1
		self.dropout = 0.5
		self.c_attn = nn.Linear(internal_width, 3 * internal_width, bias=bias)
		self.fc_out2 = nn.Linear(internal_width, internal_width)
		self.resid_dropout = nn.Dropout(0.5)
		self.c_proj = nn.Linear(internal_width, internal_width, bias=bias)
		self.c_fc = nn.Linear(internal_width, 4 * internal_width, bias=bias)
		self.c_fc2  = nn.Linear(4 * internal_width, internal_width, bias=bias)
		self.gelu = nn.GELU()
		self.fc_dropout = nn.Dropout(0.5)
		self.x_proj = nn.Linear(width, internal_width)

	def forward(self, zL, zH, x=None):

		if len(zL.shape) == 2:
			zL = zL.unsqueeze(dim=1)
		if len(zH.shape) == 2:
			zH = zH.unsqueeze(dim=1)
		if x != None and len(x.shape) == 2:
			x = x.unsqueeze(dim=1)

		if zL.shape[0] > zH.shape[0]:
			zH = zH.repeat(batch_size, 1, 1)
		if zL.shape[0] < zH.shape[0]:
			zL = zL.repeat(batch_size, 1, 1)

		if x == None:
			x = torch.concat([zL, zH], dim=1)
		else:
			if x.shape[0] > zL.shape[0]:
				zL = zL.repeat(batch_size, 1, 1)
			if x.shape[0] > zH.shape[0]:
				zH = zH.repeat(batch_size, 1, 1)
			if x.shape[0] < zL.shape[0]:
				x = x.repeat(batch_size, 1, 1)
			x = torch.concat([zL, zH, x], dim=1)

		# Causal Self-Attention
		B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
		q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
		k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
		q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
		v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
		y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
		y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
		y = self.resid_dropout(self.c_proj(y))

		# MLP
		y = self.c_fc(y)
		y = self.gelu(y)
		y = self.c_fc2(y)
		y = self.fc_dropout(y)

		y = y[:,-1:,:]

		return y # zL

class InputHead(nn.Module):
	def __init__(self):
		super(InputHead, self).__init__()
		self.fc1 = nn.Linear(width, internal_width)

	def forward(self, x):
		x = x.flatten(start_dim=1)
		x = self.fc1(x)
		return x

class OutputHead(nn.Module):
	def __init__(self, output_size=4):
		super(OutputHead, self).__init__()
		self.fc1 = nn.Linear(internal_width, output_size)

	def forward(self, x):
		x = x.flatten(start_dim=1)
		x = self.fc1(x)
		x = F.softmax(x, dim=1)
		return x

class HRM(nn.Module):
	def __init__(self, output_size=4):
		super(HRM, self).__init__()
		self.L_net = Net()
		self.H_net = Net()
		self.output_head = OutputHead(output_size)
		self.input_embedding = ConvNet()
	def forward(self, z, x, N=2, T=2):
		x = x.to(device)
		x = self.input_embedding(x)
		zH, zL = z
		zH = zH.to(device)
		zL = zL.to(device)
		with torch.no_grad():
			for _i in range(N * T - 1):
				zL = self.L_net(zL, zH, x)
				if (_i + 1) % T == 0:
					zH = self.H_net(zH, zL)

		# 1−step grad
		zL = self.L_net(zL, zH, x)
		zH = self.H_net(zH, zL, x)
		return (zH, zL), self.output_head(zH)
```
