from model import *
from config.config import relations_list, drug_type
from datamanager import DataManager
from repository import Repository
import matplotlib.pyplot as plt
from relationdataset import RelationDataset
from torch.utils.data import DataLoader, random_split

device = ("cuda" if torch.cuda.is_available() else "cpu")

training_losses = []
testing_accuracy = []
batch_size = 100
epoch_number = 20

step_threshold = 0.7

def retrieve_data():
    repository = Repository()
    raw_data_entities = repository.query_sample_by_drug_type(drug_type)
    raw_data_relations = repository.query_relations_triples(relations_list, raw_data_entities)

    if (raw_data_relations == None):
        raise Exception('Impossible to retrieve data for relations.')
    else:
        print('-> %d relation types retrieved :' % (len(raw_data_relations)))
        for key, value in raw_data_relations.items():
            print('     %s : %s triples retrieved.' % (key, str(len(value))))

    if (raw_data_entities == None):
        raise Exception('Impossible to retrieve data for entities.')
    else:
        print('\n-> %d entities retrieved.\n' % (len(raw_data_entities)))
    
    return raw_data_relations, raw_data_entities

def construct_graph():
    ## Retrieve data
    raw_data_relations, raw_data_entities = retrieve_data()

    ## Create graph
    data_manager = DataManager(raw_data_entities, raw_data_relations)
    graph = data_manager.construct_graph()
    graph.to(device)

    ## Add selfLoop triples to raw_data_relations
    self_loop_relations = []
    entities_count = data_manager.get_entity_count()
    for i in range(entities_count):
        entity = data_manager.get_entity(i)
        self_loop_relations.append(((entity, 'selfLoop', entity), 1))

    raw_data_relations['selfLoop'] = self_loop_relations
    
    return graph, raw_data_relations, data_manager

def create_dataset(data_relations, data_manager, negative_samples_count):
    dataset = RelationDataset([])

    for _, triples in data_relations.items():
        dataset.add(triples)

    dataset.negative_sampling(negative_samples_count, data_manager)
    
    return dataset

def split_data_relations(relations_dataset, split_ratio=0.7):
    # generator = torch.Generator().manual_seed(42)
    train_size = int(split_ratio * len(relations_dataset))
    test_size = len(relations_dataset) - train_size

    return random_split(relations_dataset, [train_size, test_size])

def train(train_dataloader, model, loss_fn, optimizer, device):
    size = len(train_dataloader.dataset)
    model.train()

    passed_examples = 0
    average_loss = 0
    pred, loss = None, None
    for batch, (x, y) in enumerate(train_dataloader):
        try:
            y = y.to(device)
            # Compute prediction error
            pred = model(x).to(device)            

            ## Step
            pred[pred > step_threshold] = 1.0
            pred[pred <= step_threshold] = 0.0
            # print(pred)
            loss = loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_value = loss.item()
            passed_examples += len(x[0])
            average_loss += loss_value
            print(f"Loss: {loss_value:>7f}  [{passed_examples:>5d}/{size:>5d}]")
        
        except RuntimeError as re:
            print(re)
            print('Prdicted values :', pred)
            print('Loss :', loss)
            exit()

    average_loss /= batch + 1
    print(f'-> Average loss : {average_loss:>7f}')

    training_losses.append(average_loss)

def test(test_dataloader, model, loss_fn):
    size = len(test_dataloader.dataset)
    num_batches = len(test_dataloader)
    model.eval()
    test_loss, correct = 0, 0

    with torch.no_grad():
        for x, y in test_dataloader:
            y = y.to(device)
            pred = model(x).to(device)

            ## Step
            pred[pred >= step_threshold] = 1.0
            pred[pred < step_threshold] = 0.0

            correct += (pred.argmax(0) == y).type(torch.float).sum().item()

    correct /= size
    accuracy = 100*correct
    testing_accuracy.append(accuracy)

    print(f"\nTest Error:\nAccuracy: {(accuracy):>0.1f}%\n")

if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)

    ## Print device
    print(f"Using {device} device")

    ## Construct graph and retrive relations triples
    graph, data_relations, data_manager = construct_graph()
    print(graph)

    ## Construct Dataset
    relations_dataset = create_dataset(data_relations, data_manager, negative_samples_count=800)

    rgcn = BasicRGCN(graph=graph, in_features=2, out_features=2, data_manager=data_manager, layer_count=2).to(device)
    print(rgcn, '\n')

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(rgcn.parameters(), lr=0.1)
    
    for epoch in range (epoch_number):
        training_dataset, testing_dataset = split_data_relations(relations_dataset)
        train_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(testing_dataset, batch_size=batch_size, shuffle=True)

        print(f"Epoch {epoch + 1}\n-------------------------------")
        train(train_dataloader, rgcn, loss_fn, optimizer, device)
        test(test_dataloader, rgcn, loss_fn)

    print("Done!")

    torch.save(rgcn.state_dict(), "./model.pth")
    print("Saved PyTorch Model State to model.pth")

    plt.plot(training_losses)
    plt.title("Loss evolution")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig('./loss.png')
    plt.close()

    plt.plot(testing_accuracy)
    plt.title("Accuracy evolution")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig('./accuracy.png')