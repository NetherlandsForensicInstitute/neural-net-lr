from torch import Tensor, nn
from torch.nn import Module


class PairToOutput(nn.Module):
    """
    Connects a 'suspect_net' and a 'crime_related' net to a top_layer that takes two inputs
    (the output of the 'suspect_net' and the output of the 'crimen_related' net) and returns a prediction.
    """

    def __init__(self, susp_net: Module, cr_net: Module, top_layer: Module):
        super().__init__()
        self.susp_net = susp_net
        self.cr_net = cr_net
        self.final_layer = top_layer

    def forward(self, susp: Tensor, cr: Tensor) -> Tensor:
        susp_emb = self.susp_net(susp)
        cr_emb = self.cr_net(cr)
        final_output = self.final_layer(susp_emb, cr_emb)
        return final_output
