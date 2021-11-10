from core.solver import SOLVER
from core.test import test_class
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str,default = 'syn', help='dataset',choices=['syn','torcs'])
parser.add_argument('--eval', default=False,help='eval only',action='store_true')
parser.add_argument('--PN', default=False,help='only plot expert vs negative',action='store_true')
parser.add_argument('--id', type=int,default=1,help='id of res')
parser.add_argument('--simulate', default=False,help='simulate',action='store_true')
parser.add_argument('--batch_size', type=int,default=32,help='batch size')
parser.add_argument('--lr', type=float,default=1e-3,help='learning rate')
parser.add_argument('--num_traj', type=int,default=50,help='# of state/action stacked')
parser.add_argument('--loss', type=str,default='simclr',help='Encoder Loss',choices=['simclr','BT'])
parser.add_argument('--recon_loss', action='store_true',default=False,help='use recon loss')
parser.add_argument('--policy', type=str,default='mlp',help='Polcy struture',choices=['mlp','mdn'])
parser.add_argument('--encoder', type=str,default='mlp',help='Polcy struture',choices=['mlp','mdn'])
args = parser.parse_args()
if args.eval:
    test = test_class(args.id,args.PN)
    if args.simulate:
        test.simulate_syn()
    else:
       test.test()
else:
    sol = SOLVER(args)
    sol.train()
    sol.plot_loss()