import torch
import torch.nn as nn
import config
import transformers

class EntityModel_crf(nn.Module):
    def __init__(self, num_tag):
        super(EntityModel_crf, self).__init__()
        self.num_tag = num_tag
        self.bert = transformers.BertModel.from_pretrained(config.BASE_MODEL)
        self.bert_drop = nn.Dropout(0.3)
        self.out_tag = nn.Linear(768, self.num_tag)
        self.transitions = nn.Parameter(torch.randn(self.num_tag, self.num_tag))

    def _score_sentence(self, feat, mask, tag):
        emiss_score = torch.gather(feat, dim = 2, index = tag.unsqueeze(2)).squeeze(2)
        trans_score = self.transitions[tag[:,:-1],tag[:,1:]]
        emiss_score[:, 1:] += trans_score
        total_score = (emiss_score * mask.type(torch.float)).sum(dim=1)
        return  total_score

    def _calc_scaling_factor(self, feat, mask):
        batch_size, sent_length, _ = feat.shape
        dp = torch.unsqueeze(feat[:,0], dim=1) #b, 1, K
        # vec = torch.unsqueeze(feat[:,0], dim=1)#b, 1, K

        for i in range(1, sent_length):
            emit_trans = feat[:,i].unsqueeze(dim=1) + self.transitions #b, K, K
            vec = dp[:, -1, :].unsqueeze(1)
            log_sum = vec.transpose(1, 2) + emit_trans #b, K, K
            max_v = log_sum.max(dim=1)[0].unsqueeze(dim=1) #b, 1, K
            log_sum = log_sum - max_v
            vec = max_v + torch.logsumexp(log_sum, dim=1).unsqueeze(dim=1)
            dp = torch.cat((dp, vec), dim = 1)

        mask = torch.sum(mask, dim = 1)-1
        ind = torch.tensor(list(range(batch_size)), device=config.DEVICE)
        d = dp[ind, mask, :]

        max_d = d.max(dim=-1)[0]
        d = max_d + torch.logsumexp(d - max_d.unsqueeze(dim=1), dim=1)
        return d

    def predict(self, ids, mask, token_type_ids, target_tag):

        o1, _ = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        bo_tag = self.bert_drop(o1)
        feat = self.out_tag(bo_tag)

        batch_size, sent_length, tag_num = feat.shape
        ind = torch.tensor(list(range(tag_num)), device=config.DEVICE)
        dp_ind = ind.repeat(batch_size, 1, 1) # b, 1, K
        dp_val = torch.unsqueeze(feat[:, 0], dim=1) # b, 1, K

        for i in range(1, sent_length):
            emit_trans = feat[:, i].unsqueeze(dim=1) + self.transitions  # b, K, K
            vec = dp_val[:, -1, :].unsqueeze(1)
            log_sum = vec.transpose(1, 2) + emit_trans  # b, K, K
            max_val, max_ind = torch.max(log_sum, dim=1) # b, K
            max_val = max_val.unsqueeze(dim=1) # b, 1, K
            max_ind = max_ind.unsqueeze(dim=1) # b, 1, K

            dp_val = torch.cat((dp_val, max_val), dim = 1)
            dp_ind = torch.cat((dp_ind, max_ind), dim = 1)

        max_pos = torch.sum(mask, dim=1)
        ind = torch.tensor(list(range(batch_size)), device=config.DEVICE)
        d = dp_val[ind, max_pos-1, :]
        _, max_idx_n = torch.max(d, dim=1)

        mask = mask.view(-1)
        active_loss = mask == 1
        target_tag = target_tag.view(-1)
        target_tag = torch.where(active_loss, target_tag, torch.ones(target_tag.shape).type_as(target_tag)*-1)
        return dp_ind, max_idx_n, max_pos, target_tag

    def forward(self, ids, mask, token_type_ids, target_tag):
        o1, _ = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        bo_tag = self.bert_drop(o1)
        feat = self.out_tag(bo_tag)
        gold_score = self._score_sentence(feat, mask, target_tag)
        sent_score = self._calc_scaling_factor(feat, mask)
        loss = torch.sum(sent_score - gold_score)

        return loss
