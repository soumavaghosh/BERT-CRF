import config
import torch
import transformers
import torch.nn as nn


def loss_fn(output, target, mask, num_labels):
    lfn = nn.CrossEntropyLoss()
    active_loss = mask.view(-1) == 1
    active_logits = output.view(-1, num_labels)
    active_labels = torch.where(
        active_loss,
        target.view(-1),
        torch.tensor(lfn.ignore_index).type_as(target)
    )
    loss = lfn(active_logits, active_labels)
    labels = torch.argmax(active_logits, dim=1)
    labels = torch.where(active_loss, labels, torch.tensor(lfn.ignore_index).type_as(target))
    return loss, labels, active_labels


class EntityModel(nn.Module):
    def __init__(self, num_tag):
        super(EntityModel, self).__init__()
        self.num_tag = num_tag
        self.bert = transformers.BertModel.from_pretrained(config.BASE_MODEL)
        self.bert_drop = nn.Dropout(0.3)
        self.out_tag = nn.Linear(768, self.num_tag)

    def forward(self, ids, mask, token_type_ids, target_tag):
        o1, _ = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        bo_tag = self.bert_drop(o1)
        tag = self.out_tag(bo_tag)
        loss, labels, active_labels = loss_fn(tag, target_tag, mask, self.num_tag)

        return tag, loss, labels, active_labels
