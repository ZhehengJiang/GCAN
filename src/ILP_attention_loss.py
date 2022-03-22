import torch
import torch.nn as nn
import torch.nn.functional as F
from src.lap_solvers.ILP import ILP_solver
# from src.lap_solvers.ILP_varied import ILP_solver as ILP_solver_varied
from torch import Tensor

class ILP_attention_loss(nn.Module):

    def __init__(self,varied_size=False):
        super(ILP_attention_loss, self).__init__()
        self.varied_size = varied_size

    def forward(self, pred_dsmat: Tensor, gt_perm: Tensor, src_ns: Tensor, tgt_ns: Tensor) -> Tensor:

        batch_num = pred_dsmat.shape[0]

        pred_dsmat = pred_dsmat.to(dtype=torch.float32)

        try:
            assert torch.all((gt_perm >= 0) * (gt_perm <= 1))
        except AssertionError as err:
            print(pred_dsmat)
            raise err

        if self.varied_size:
            dis_pred = ILP_solver(pred_dsmat+1, src_ns+1, tgt_ns+1, dummy=True)
        else:
            dis_pred = ILP_solver(pred_dsmat, src_ns, tgt_ns)
        ali_perm = dis_pred + gt_perm
        ali_perm[ali_perm >= 1.0] = 1
        pred_dsmat = torch.mul(ali_perm, pred_dsmat)
        gt_perm = torch.mul(ali_perm, gt_perm)

        loss = torch.tensor(0.).to(pred_dsmat.device)
        n_sum = torch.zeros_like(loss)
        for b in range(batch_num):
            batch_slice = [b, slice(src_ns[b]), slice(tgt_ns[b])]
            loss += F.binary_cross_entropy(
                pred_dsmat[batch_slice],
                gt_perm[batch_slice],
                reduction='sum')
            n_sum += src_ns[b].to(n_sum.dtype).to(pred_dsmat.device)

        return loss/n_sum

