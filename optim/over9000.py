# from https://raw.githubusercontent.com/mgrankin/over9000/master/over9000.py

from SignProdJoey.optim.lookahead import Lookahead
from SignProdJoey.optim.ralamb import Ralamb

# Lookahead implementation from https://github.com/lonePatient/lookahead_pytorch/blob/master/optimizer.py
# RAdam + LARS implementation from https://gist.github.com/redknightlois/c4023d393eb8f92bb44b2ab582d7ec20


# RAdam + LARS + LookAHead
def Over9000(params, alpha=0.5, k=6, *args, **kwargs):
    ralamb = Ralamb(params, *args, **kwargs)
    return Lookahead(ralamb, alpha, k)
