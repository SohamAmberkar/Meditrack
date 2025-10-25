# src/gnn/train_gnn.py
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

# tiny graph: 4 nodes, edges undirected
edge_index = torch.tensor([[0,1,1,2,2,3],[1,0,2,1,3,2]], dtype=torch.long)
x = torch.randn((4,16))  # node features
y = torch.tensor([0.5, 1.0, 0.8, 1.2])  # target (e.g., speed)

data = Data(x=x, edge_index=edge_index)
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(16, 32)
        self.conv2 = GCNConv(32, 1)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x.squeeze()

model = Net()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
for epoch in range(200):
    model.train()
    pred = model(data)
    loss = F.mse_loss(pred, y)
    opt.zero_grad(); loss.backward(); opt.step()
    if epoch%50==0:
        print(epoch, loss.item())
print("Done")
