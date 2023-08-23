from model import *
from config.config import relations_list, drug_type
from graphbuilder import GraphBuilder
from dataloader import Dataloader
import matplotlib.pyplot as plt
from torch import max, min

trainig_losses = []

def retrieve_data():
    data_loader = Dataloader()
    raw_data_relations = data_loader.load_relations_triplets(relations_list)
    raw_data_entities = data_loader.load_sample_by_drug_type(drug_type)
    
    return raw_data_relations, raw_data_entities
    

def train(training_graph: Graph, model, loss_fn, optimizer, device):
    model.train()
    training_graph.to(device)
    # Compute prediction error
    pred = model(training_graph)
    loss = loss_fn(pred, training_graph).to(device)

    # Backpropagation
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    loss = loss.item()
    trainig_losses.append(loss)
    print(f"loss: {loss:>7f}%")

def test(testing_graph: Graph, model, loss_fn):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        pred = model(testing_graph)
        test_loss += loss_fn(pred, testing_graph).item()

    print(f"Test loss: {test_loss:>8f}% \n")


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

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


    ## Create graph
    graph_builder = GraphBuilder(raw_data_entities, raw_data_relations)
    training_graph, testing_graph = graph_builder.construct_graphs()
    training_graph.to(device)
    testing_graph.to(device)

    print(training_graph)
    print(testing_graph)

    rgcn = BasicRGCN(in_features=2, out_features=2, relations_count=2).to(device)
    print(rgcn)

    loss_fn = CustomLoss()
    optimizer = torch.optim.SGD(rgcn.parameters(), lr=1e-2)
    
    for epoch in range (1, 100):
        print(f"Epoch {epoch}\n-------------------------------")
        train(training_graph, rgcn, loss_fn, optimizer, device)
        test(testing_graph, rgcn, loss_fn)

    print("Done!")

    torch.save(rgcn.state_dict(), "model.pth")
    print("Saved PyTorch Model State to model.pth")

    plt.plot(trainig_losses)
    plt.title("Loss evolution")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (%)")
    plt.savefig('loss.png')