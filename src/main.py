from model import *
from config.config import *
from graphbuilder import GraphBuilder
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

def retrieve_data():
    data_loader = Dataloader()
    raw_data_relations = data_loader.load_relations_triples(relations_list)
    raw_data_entities = data_loader.load_instances("Echantillon")
    
    return raw_data_relations, raw_data_entities
    

def train(training_graph: Graph, model, loss_fn, optimizer, device):
    # size = len(train_dataloader.dataset)
    model.train()
    
    # for batch in enumerate(train_dataloader):
    #     print(batch)
    #     X, y = X.to(device), y.to(device)
    
    training_graph.get_adjacency_matrices().to(device)
    training_graph.get_features().to(device)

    # Compute prediction error
    pred = model(training_graph)
    loss = loss_fn(pred, training_graph)

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


if __name__ == '__main__':
    ## Retrieve data
    raw_data_relations, raw_data_entities = retrieve_data()

    ## Create matrice
    matrice_builder = GraphBuilder(raw_data_entities, raw_data_relations)
    training_graph, testing_graph = matrice_builder.construct_graphs()

    ## Create dataset
    # training_data, testing_data = negative_sampling(raw_data_relations)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    rgcn = BasicRGCN(2, 2, 2)
    print(rgcn)

    loss_fn = Loss()
    optimizer = torch.optim.SGD(rgcn.parameters(), lr=1e-2)
    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(training_graph, rgcn, loss_fn, optimizer, device)
        test(testing_graph, rgcn, loss_fn)
    print("Done!")

    torch.save(rgcn.state_dict(), "model.pth")
    print("Saved PyTorch Model State to model.pth")