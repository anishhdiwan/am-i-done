import torch
import torch.nn as nn

class BOLoss(nn.Module):
    def __init__(self, phase_intervals):
        super(BOLoss, self).__init__()
        self.phase_intervals = phase_intervals

    def forward(self, predictions, targets):
        N = predictions.shape[0]
        errors = torch.abs(predictions - targets)
        potentials = []

        for (l_k, u_k) in self.phase_intervals:
            m_k = (l_k + u_k) / 2
            r_k = (u_k - l_k) / 2
            e_k = torch.min(
                torch.tensor(1.0),
                ((targets - m_k) / (r_k * torch.sqrt(torch.tensor(2.0)))) ** 2 +
                ((predictions - m_k) / (r_k * torch.sqrt(torch.tensor(2.0)))) ** 2
            )
            potentials.append(e_k)

        min_potentials = torch.stack(potentials).min(dim=0).values
        bo_loss = torch.mean(min_potentials * errors)

        return bo_loss
        
if __name__ == '__main__':
    phase_intervals = [(0.0, 0.5), (0.5, 1.0)]
    bo_loss = BOLoss(phase_intervals)
    predictions = torch.tensor([0.5, 0.7, 0.1], requires_grad=True)
    targets = torch.tensor([0.6, 0.8, 0.2])
    loss = bo_loss(predictions, targets)
    print("Loss:", loss.item())

