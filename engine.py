import torch
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
from time import sleep
def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()
    final_loss = 0
    actual = []
    predicted = []
    for data in tqdm(data_loader, total=len(data_loader)):
        for k, v in data.items():
            data[k] = v.to(device)
        optimizer.zero_grad()
        # _, loss, labels, active_labels = model(**data)
        # labels = [i for i in list(labels.data.cpu().numpy()) if i>=0]
        # active_labels = [i for i in list(active_labels.data.cpu().numpy()) if i>=0]
        # actual+=active_labels
        # predicted+=labels
        loss = model(**data)
        loss.backward()
        optimizer.step()
        scheduler.step()
        final_loss += loss.item()
    return final_loss / len(data_loader)
    #f1_score(actual, predicted, average='macro')


def eval_fn(data_loader, model, device):
    model.eval()
    final_loss = 0
    actual = []
    predicted = []
    for data in tqdm(data_loader, total=len(data_loader)):
        for k, v in data.items():
            data[k] = v.to(device)
        # _, loss, labels, active_labels = model(**data)
        # final_loss += loss.item()
        # labels = [i for i in list(labels.data.cpu().numpy()) if i >= 0]
        # active_labels = [i for i in list(active_labels.data.cpu().numpy()) if i >= 0]
        # actual += active_labels
        # predicted += labels
        loss = model(**data)
        final_loss += loss.item()
    return final_loss / len(data_loader)
    #f1_score(actual, predicted, average='macro')

def backtrack(mat, ind):
    seq = [ind]
    _, sent_length = mat.shape
    for i in range(sent_length-1, 0, -1):
        seq.append(mat[seq[-1],i])
    return seq


def mat2seq(dp_ind, max_idx_n, max_pos):
    batch_length, _, _ = dp_ind.shape
    seq = []
    for i in range(batch_length):
        mat = dp_ind[i, :max_pos[i], :]
        seq.append(backtrack(np.transpose(mat), max_idx_n[i]))
    return seq

def train_fn_crf(data_loader, model, optimizer, device, scheduler):
    model.train()
    final_loss = 0
    for data in tqdm(data_loader, total=len(data_loader), desc='Training'):
        for k, v in data.items():
            data[k] = v.to(device)
        optimizer.zero_grad()
        loss = model(**data)
        loss.backward()
        optimizer.step()
        scheduler.step()
        final_loss += loss.item()
    return final_loss / len(data_loader)


def eval_fn_crf(data_loader, model, device):
    model.eval()
    final_loss = 0
    actual = []
    predicted = []
    for data in tqdm(data_loader, total=len(data_loader), desc='Evaluating'):
        for k, v in data.items():
            data[k] = v.to(device)
        loss = model(**data)
        dp_ind, max_idx_n, max_pos, target_tag = model.predict(**data)
        dp_ind, max_idx_n, max_pos, target_tag = dp_ind.data.cpu().numpy(), max_idx_n.data.cpu().numpy(), \
                                              max_pos.data.cpu().numpy(), list(target_tag.data.cpu().numpy())
        predicted_tag = mat2seq(dp_ind, max_idx_n, max_pos)
        target_tag = [x for x in target_tag if x>=0]
        for p in predicted_tag: predicted += p
        actual += target_tag
        final_loss += loss.item()
    return final_loss / len(data_loader), f1_score(actual, predicted, average='macro'), accuracy_score(actual, predicted)
