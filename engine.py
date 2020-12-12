import torch
from tqdm import tqdm
from sklearn.metrics import f1_score

def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()
    final_loss = 0
    actual = []
    predicted = []
    for data in tqdm(data_loader, total=len(data_loader)):
        for k, v in data.items():
            data[k] = v.to(device)
        optimizer.zero_grad()
        _, loss, labels, active_labels = model(**data)
        labels = [i for i in list(labels.data.cpu().numpy()) if i>=0]
        active_labels = [i for i in list(active_labels.data.cpu().numpy()) if i>=0]
        actual+=active_labels
        predicted+=labels
        loss.backward()
        optimizer.step()
        scheduler.step()
        final_loss += loss.item()
    return final_loss / len(data_loader), f1_score(actual, predicted, average='micro')


def eval_fn(data_loader, model, device):
    model.eval()
    final_loss = 0
    actual = []
    predicted = []
    for data in tqdm(data_loader, total=len(data_loader)):
        for k, v in data.items():
            data[k] = v.to(device)
        _, loss, labels, active_labels = model(**data)
        final_loss += loss.item()
        labels = [i for i in list(labels.data.cpu().numpy()) if i >= 0]
        active_labels = [i for i in list(active_labels.data.cpu().numpy()) if i >= 0]
        actual += active_labels
        predicted += labels
    return final_loss / len(data_loader), f1_score(actual, predicted, average='micro')
