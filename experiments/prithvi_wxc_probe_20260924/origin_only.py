"""Exact origin-state path without unused concentration forecasting."""
import torch
from haze_forecast.regional_recovery_network import RecoveryPM25GNN


class RecoveryOriginOnly(RecoveryPM25GNN):
    """Same parameters and hazards; return None instead of untrained curves."""

    def forward_with_state(self, x, active):
        if (x.ndim != 4 or x.shape[1:] != (self.history,self.node_count,self.input_dim)
                or x.shape[0] < 1):
            raise ValueError('expected nonempty [batch,history,node,feature]')
        batch, _, nodes, _ = x.shape
        if (active.shape != (batch,nodes,2) or active.dtype != torch.bool
                or active.device != x.device or not active.any(dim=1).all()):
            raise ValueError('each batch/target needs active nodes')
        pooled = []
        for target, (index, model) in enumerate(zip(self.target_indices,self.heads,strict=True)):
            model.graph_gnn.device = x.device
            hidden = x.new_zeros(batch*nodes,model.hid_dim)
            # Original lead-zero step consumes the last observation, not a forecast.
            for hour in range(self.history):
                observed = torch.cat([x[:,hour,:,index:index+1],x[:,hour]],dim=-1)
                spatial = model.graph_gnn(observed.contiguous())
                hidden = model.gru_cell(torch.cat([spatial,observed],dim=-1),hidden)
            encoded = hidden.view(batch,nodes,model.hid_dim)
            mask = active[:,:,target].unsqueeze(-1)
            pooled.append((encoded*mask).sum(dim=1)/mask.sum(dim=1))
        return None, torch.cat(pooled,dim=-1)
