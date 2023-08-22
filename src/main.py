from model import *
from config.config import relations_list, drug_type
from graphbuilder import GraphBuilder
from linkdataset import LinkDataset
from dataloader import Dataloader
import matplotlib.pyplot as plt

trainig_losses = []

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
    raw_data_relations = data_loader.load_relations_triplets(relations_list)
    raw_data_entities = data_loader.load_sample_by_drug_type(drug_type)
    
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
    trainig_losses.append(loss)
    print(f"loss: {loss:>7f}%")

def test(testing_graph: Graph, model, loss_fn):
    # size = len(test_dataloader.dataset)
    # num_batches = len(test_dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        pred = model(testing_graph)
        test_loss += loss_fn(pred, testing_graph).item()
        # correct += (pred.argmax(1) == training_data).type(torch.float).sum().item()
    # test_loss /= num_batches
    # correct /= size
    # print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    print(f"Test Error: Avg loss: {test_loss:>8f}% \n")


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    ## Retrieve data
    raw_data_relations, raw_data_entities = retrieve_data()

    if (raw_data_relations == None):
        raise Exception('Impossible to retrieve data for relations.')
    else:
        print('%d relation types retrieved :' % (len(raw_data_relations)))
        for key, value in raw_data_relations.items():
            print('     %s : %s triplets retrieved.' % (key, str(len(value))))

    if (raw_data_entities == None):
        raise Exception('Impossible to retrieve data for entities.')
    else:
        print('%d entities retrieved.\n' % (len(raw_data_entities)))


    ## Create matrice
    matrice_builder = GraphBuilder(raw_data_entities, raw_data_relations)
    training_graph, testing_graph = matrice_builder.construct_graphs()

    print(training_graph)
    print(testing_graph)

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

    rgcn = BasicRGCN(in_features=2, out_features=2, relations_count=2)
    print(rgcn)

    loss_fn = Loss()
    optimizer = torch.optim.SGD(rgcn.parameters(), lr=1e-2)
    
    loss_gain = 100
    epoch = 1
    while loss_gain > 0.01:
        print(f"Epoch {epoch}\n-------------------------------")
        train(training_graph, rgcn, loss_fn, optimizer, device)
        test(testing_graph, rgcn, loss_fn)

        if len(trainig_losses) == 1:
            loss_gain = trainig_losses[0]
        
        else:
            epoch = len(trainig_losses)
            loss_gain = trainig_losses[epoch - 2] - trainig_losses[epoch - 1]

    print("Done!")

    torch.save(rgcn.state_dict(), "model.pth")
    print("Saved PyTorch Model State to model.pth")

    plt.plot(trainig_losses)
    plt.title("Loss evolution")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (%)")
    plt.savefig('loss.png')