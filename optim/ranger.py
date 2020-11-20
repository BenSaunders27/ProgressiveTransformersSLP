# from https://github.com/mgrankin/over9000/blob/master/ranger.py

from SignProdJoey.optim.lookahead import Lookahead
from SignProdJoey.optim.radam import RAdam


def Ranger(params, alpha=0.5, k=6, *args, **kwargs):
    radam = RAdam(params, *args, **kwargs)
    return Lookahead(radam, alpha, k)
