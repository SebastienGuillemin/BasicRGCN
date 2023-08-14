from torch.utils.data import DataLoader
from model import *
from adjacencymatricesbuilder import MatricesBuilder
from linkdataset import LinkDataset
from dataloader import Dataloader

data_loader = Dataloader()

raw_data_relations = data_loader.load_relations_triples(["estProcheDe"])
raw_data_entities = data_loader.load_instances("Echantillon")

matrice_builder = MatricesBuilder(raw_data_entities, raw_data_relations)

adjacency_matrices = matrice_builder.construct_matrices()

def negative_sampling(original_data, split_ratio=0.7):
    # TODO : implement negative sampling
    training_data = []
    testing_data = []
    
    for relation_name in original_data:
        relation_list = original_data[relation_name]
        split_index = math.floor(len(relation_list) * split_ratio)
        
        training_data += relation_list[:split_index]
        testing_data += relation_list[split_index:]
    
    return LinkDataset(training_data), LinkDataset(testing_data)
    

training_data, testing_data  = negative_sampling(raw_data_relations)

batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(testing_data, batch_size=batch_size)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


rgcn = BasicRGCN(adjacency_matrices, 1, 1, matrice_builder)
print(rgcn)

loss_fn = Loss()
optimizer = torch.optim.SGD(rgcn.parameters(), lr=1e-2)

def train(train_dataloader, model, loss_fn, optimizer):
    size = len(train_dataloader.dataset)
    model.train()
    for batch in enumerate(train_dataloader):
        print(batch)
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

def test(test_dataloader, model, loss_fn):
    size = len(test_dataloader.dataset)
    num_batches = len(test_dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in test_dataloader:
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