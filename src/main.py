from torch.utils.data import DataLoader
from model import *
from config.config import *
from adjacencymatricesbuilder import MatricesBuilder
from linkdataset import LinkDataset
from dataloader import Dataloader

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

## Retrieve data
data_loader = Dataloader()

raw_data_relations = data_loader.load_relations_triples(relations_list)
raw_data_entities = data_loader.load_instances("Echantillon")

## Create matrice
matrice_builder = MatricesBuilder(raw_data_entities, raw_data_relations)

adjacency_matrices, feature_matrices = matrice_builder.construct_matrices()

## Create dataset
training_data, testing_data  = negative_sampling(raw_data_relations)

# Create data loaders.
# batch_size = 64
# train_dataloader = DataLoader(training_data, batch_size=batch_size)
# test_dataloader = DataLoader(testing_data, batch_size=batch_size)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

rgcn = BasicRGCN(adjacency_matrices, 2, 2, matrice_builder)
print(rgcn)

loss_fn = Loss(matrice_builder, 1)
optimizer = torch.optim.SGD(rgcn.parameters(), lr=1e-2)

def train(feature_matrices, training_data, model, loss_fn, optimizer):
    # size = len(train_dataloader.dataset)
    model.train()
    
    # for batch in enumerate(train_dataloader):
    #     print(batch)
    #     X, y = X.to(device), y.to(device)
    
    feature_matrices.to(device)

        # Compute prediction error
    pred = model(feature_matrices)
    loss = loss_fn(pred, training_data)

    # Backpropagation
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    loss = loss.item()
    print(f"loss: {loss:>7f}")

def test(feature_matrices, training_data, model, loss_fn):
    # size = len(test_dataloader.dataset)
    # num_batches = len(test_dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        pred = model(feature_matrices)
        test_loss += loss_fn(pred, training_data).item()
        correct += (pred.argmax(1) == training_data).type(torch.float).sum().item()
    # test_loss /= num_batches
    # correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(feature_matrices, training_data, rgcn, loss_fn, optimizer)
    test(feature_matrices, training_data, rgcn, loss_fn)
print("Done!")

torch.save(rgcn.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")