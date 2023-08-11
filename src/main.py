from torch.utils.data import DataLoader
from model import *
from adjacencymatricesbuilder import MatricesBuilder
from dataloader import Dataloader

data_loader = Dataloader()

raw_data_relations = data_loader.load_relations_triples(["estProcheDe"])
raw_data_entities = data_loader.load_instances("Echantillon")

matrice_builder = MatricesBuilder(raw_data_entities, raw_data_relations)

adjacencies_matrix = matrice_builder.construct_matrices()

def negative_sampling(original_data, split_ratio=0.7):
    split_index = math.floor(len(original_data) * split_ratio)
    training_data = original_data[:split_index]
    testing_data = original_data[split_index:]
    
    return training_data, testing_data
    

training_data, testing_data  = negative_sampling(raw_data_relations)

batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(testing_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


rgcn = BasicRGCN(2, 1)
print(rgcn)

loss_fn = Loss()
optimizer = torch.optim.SGD(rgcn.parameters(), lr=1e-2)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        print(pred)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, rgcn, loss_fn, optimizer)
    test(test_dataloader, rgcn, loss_fn)
print("Done!")

torch.save(rgcn.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")