import torch
import torch.nn.functional as F
import torch.nn as nn

def compute_simclr_loss(logits_a, logits_b, logits_a_gathered, logits_b_gathered, labels, temperature):
    sim_aa = logits_a @ logits_a_gathered.t() / temperature
    sim_ab = logits_a @ logits_b_gathered.t() / temperature
    sim_ba = logits_b @ logits_a_gathered.t() / temperature
    sim_bb = logits_b @ logits_b_gathered.t() / temperature
    masks = torch.where(F.one_hot(labels, logits_a_gathered.size(0)) == 0, 0, float('-inf'))
    sim_aa += masks
    sim_bb += masks
    sim_a = torch.cat([sim_ab, sim_aa], 1)
    sim_b = torch.cat([sim_ba, sim_bb], 1)
    loss_a = F.cross_entropy(sim_a, labels)
    loss_b = F.cross_entropy(sim_b, labels)
    return (loss_a + loss_b) * 0.5

class SupConLoss(nn.Module):
    def __init__(self, device):
        super(SupConLoss, self).__init__()
        self.device = device
        self.temperature = 1.0
    def forward(self, text_features, image_features, t_label, i_targets): 
        batch_size = text_features.shape[0] 
        batch_size_N = image_features.shape[0] 
        mask = torch.eq(t_label.unsqueeze(1).expand(batch_size, batch_size_N), \
            i_targets.unsqueeze(0).expand(batch_size,batch_size_N)).float().to(self.device) 

        logits = torch.div(torch.matmul(text_features, image_features.T),self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach() 
        exp_logits = torch.exp(logits) 
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True)) 
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1) 
        loss = - mean_log_prob_pos.mean()

        return loss