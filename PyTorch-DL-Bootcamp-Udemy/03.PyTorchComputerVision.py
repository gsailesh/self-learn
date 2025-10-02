import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


train_data = datasets.FashionMNIST(
	root="Data",
	download=True,
	train=True,
	transform=ToTensor(),
	target_transform=None
)

test_data = datasets.FashionMNIST(
	root="Data",
	download=True,
	train=False,
	transform=ToTensor()
)


BATCH_SIZE = 32

train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)


class FashionMNISTModelV0(torch.nn.Module):
	def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
		super().__init__()
		self.layer_stack = torch.nn.Sequential(
			torch.nn.Flatten(),
			torch.nn.Linear(in_features=input_shape, out_features=hidden_units),
			torch.nn.Linear(in_features=hidden_units, out_features=output_shape)
		)

	def forward(self, x):
		return self.layer_stack(x)


torch.manual_seed(42)

model_v0 = FashionMNISTModelV0(input_shape=28*28, hidden_units=10, output_shape=len(train_data.classes))

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_v0.parameters(), lr=0.05)

torch.manual_seed(42)

n_epochs=10

for epoch in tqdm(range(n_epochs)):
	train_loss = 0.

	for batch, (X, y) in enumerate(train_dataloader):
		model_v0.train()

		y_pred = model_v0(X)
		loss = loss_fn(y_pred, y)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	train_loss /= len(train_dataloader)
	print(f"Average training loss: {train_loss:3f}")

	test_loss = 0.
	model_v0.eval()

	with torch.inference_mode():
		for X, y in test_dataloader:
			test_pred = model_v0(X)
			test_loss += loss_fn(test_pred, y)

		test_loss /= len(test_dataloader)
	print(f"Average test loss: {test_loss:3f}")


#--------------- Device agnostic code ----------------------#
device = "mps" if torch.backends.mps.is_available() else "cpu"


class FashionMNISTModelV1(torch.nn.Module):
	def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
		super().__init__()
		self.layer_stack = torch.nn.Sequential(
				torch.nn.Flatten(),
				torch.nn.Linear(in_features=input_shape, out_features=hidden_units),
				torch.nn.ReLU(),
				torch.nn.Linear(in_features=hidden_units, out_features=output_shape),
				torch.nn.ReLU()
			)

	def forward(self, x):
		return self.layer_stack(x)


torch.manual_seed(42)
model_v1 = FashionMNISTModelV1(input_shape=28*28, hidden_units=10, output_shape=len(train_data.classes)).to(device)

next(model_v1.parameters()).device # Check model device.


loss_fn = torch.nn.CrossEntropyLoss()
optimimzer = torch.optim.SGD(params=model_v1.parameters(), lr=0.05)


def train_step(
	model: torch.nn.Module, 
	data_loader: torch.utils.data.DataLoader, 
	loss_fn: torch.nn.Module, 
	optimizer: torch.optim.Optimizer, 
	device: torch.device = device
	):
	train_loss = 0.
	model.to(device)

	for batch, (X, y) in enumerate(data_loader):
		X, y = X.to(device), y.to(device)

		y_pred = model(X)
		loss = loss_fn(y_pred, y)
		train_loss += loss

		loss.backward()
		optimizer.zero_grad()
		optimizer.step()

	train_loss /= len(data_loader)

	print(f"Average Train loss: {train_loss:3f}")

def test_step(	model: torch.nn.Module, 
	data_loader: torch.utils.data.DataLoader, 
	loss_fn: torch.nn.Module, 
	device: torch.device = device
):
	test_loss = 0.
	model.to(device)

	for X, y in data_loader:
		X, y = X.to(device), y.to(device)

		y_pred_test = model(X)
		loss = loss_fn(y_pred_test, y)
		test_loss += loss

	test_loss /= len(data_loader)
	print(f"Average test loss: {test_loss:3f}")


# ------------ Training and Testing loop -------------------- #

n_epochs = 5

for epoch in tqdm(range(n_epochs)):
	train_step(model=model_v1, data_loader=train_dataloader, loss_fn=loss_fn, optimizer=optimizer)
	test_step(model=model_v1, data_loader=test_dataloader, loss_fn=loss_fn)


