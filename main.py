import config
import pickle
import dataset
import torch
import tqdm
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
import warnings
import engine
import numpy as np
from model import *
from crf_model import *
warnings.filterwarnings('ignore')

def read_data(name):
    with open(f'../data/{name}/train.p', 'rb') as f:
        train_data = pickle.load(f)
    with open(f'../data/{name}/test.p', 'rb') as f:
        test_data = pickle.load(f)
    with open(f'../data/{name}/valid.p', 'rb') as f:
        valid_data = pickle.load(f)
    with open(f'../data/{name}/encoder.p', 'rb') as f:
        tag_enc = pickle.load(f)

    return train_data, test_data, valid_data, len(list(tag_enc.classes_))

train_data, test_data, valid_data, n_tags = read_data(config.DATASET)

train_dataset = dataset.EntityDataset(train_data)
train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=1)

test_dataset = dataset.EntityDataset(test_data)
test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=1)

valid_dataset = dataset.EntityDataset(valid_data)
valid_data_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=1)

#model = EntityModel_crf(num_tag=n_tags)
model = torch.load('model.pt')
print(model)
model.to(config.DEVICE)

param_optimizer = list(model.named_parameters())
no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
optimizer_parameters = [
    {
        "params": [
            p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
        ],
        "weight_decay": 0.001,
    },
    {
        "params": [
            p for n, p in param_optimizer if any(nd in n for nd in no_decay)
        ],
        "weight_decay": 0.0,
    },
]

num_train_steps = int(len(train_data) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
optimizer = AdamW(optimizer_parameters, lr=3e-5)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
)

best_loss = np.inf
for epoch in range(config.EPOCHS):
    train_loss = engine.train_fn_crf(train_data_loader, model, optimizer, config.DEVICE, scheduler)
    test_loss, test_f1, test_acc = engine.eval_fn_crf(valid_data_loader, model, config.DEVICE)
    print(f"Train Loss = {train_loss} Valid Loss = {test_loss}")
    print(f"Test F1 = {test_f1} Test acc = {test_acc}")
    if test_loss < best_loss:
        torch.save(model, config.MODEL_PATH)
        best_loss = test_loss

