from core.solver import SOLVER
from core.test import test_class
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--eval_only', default=False,help='eval only',action='store_true')
parser.add_argument('--id', type=int,default=1,help='id of res')
parser.add_argument('--batch_size', type=int,default=128,help='batch size')
parser.add_argument('--lr', type=float,default=5e-4,help='learning rate')
parser.add_argument('--num_traj', type=int,default=3,help='# of state/action stacked')
parser.add_argument('--loss', type=str,default='simclr',help='Encoder Loss',choices=['simclr','BT'])
args = parser.parse_args()
if args.eval_only:
    test = test_class(args.id)
    test.test()
else:
    sol = SOLVER(args)
    sol.train()
    sol.plot_loss()