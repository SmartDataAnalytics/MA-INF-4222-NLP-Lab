from torch.utils.data import DataLoader
from utils import TripleDataset, sample_negatives, preprocess
import numpy as np
from model import DistMult
from torch.autograd import Variable
from evaluation import evaluate_model, stats
import argparse
import pdb

parser = argparse.ArgumentParser(
				description='Demo for Commonsense KB Reasoning: Bilinear Averaging Model, Bilinear LSTM Model, \
					DistMult Averaging Model, DistMult LSTM Model, ER-MLP Averaging Model, ER-MLP LSTM Model'
)

parser.add_argument('--model', default='BilinearAvg', metavar='',
					help='model to run: {BilinearAvg, BilinearLstm, DistMultAvg, DistMultLstm, \
						ErmlpLstm, ErmlpAvg} (default: BilinearAvg)')
parser.add_argument('--sub', required=True, metavar='',
					help='Term1 to evaluate')
parser.add_argument('--pred', required=True, metavar='',
					help='Term2 to evaluate')
parser.add_argument('--eval_type', default='topfive', metavar='',
					help='Type of evaluation')

args = parser.parse_args()

np.random.seed(4086)
torch.manual_seed(4086)
torch.cuda.manual_seed(4086)
gpu = True
embedding_rel_dim = 150
embedding_dim = 200

maxlen_s = 20
maxlen_o = 25
sub_ = args.sub
pred_ = args.pred
eval_type = args.eval_type

# Load Trained Model
if args.model == 'BilinearAvg':
	model = BilinearModel(embedding_dim=embedding_dim, embedding_rel_dim=embedding_rel_dim, weights=pretrained_weights, n_r=n_r, lw=lw, batch_size=batch_size, input_dropout=dropout_p, gpu=gpu)
elif args.model == 'BilinearLstm':
	model = LSTM_BilinearModel(embedding_dim=embedding_dim, embedding_rel_dim=embedding_rel_dim, weights=pretrained_weights, n_r=n_r, lw=lw, batch_size=batch_size, input_dropout=dropout_p, gpu=gpu)
elif args.model == 'DistMultAvg':
	model = Avg_DistMult(embedding_dim=embedding_dim, embedding_rel_dim=embedding_rel_dim, weights=pretrained_weights, n_r=n_r, lw=lw, batch_size=batch_size, input_dropout=dropout_p, gpu=gpu)	
elif args.model == 'DistMultLstm':
	model = LSTM_DistMult(embedding_dim=embedding_dim, embedding_rel_dim=embedding_rel_dim, weights=pretrained_weights, n_r=n_r, lw=lw, batch_size=batch_size, input_dropout=dropout_p, gpu=gpu)
elif args.model == 'ErmlpLstm':
	model = LSTM_ERMLP(embedding_dim=embedding_dim, embedding_rel_dim=embedding_rel_dim, mlp_hidden=mlp_hidden, weights=pretrained_weights, n_r=n_r, lw=lw, batch_size=batch_size, input_dropout=dropout_p, gpu=gpu)
elif args.model == 'ErmlpAvg':
	model = ERMLP_avg(embedding_dim=embedding_dim, embedding_rel_dim=embedding_rel_dim, mlp_hidden=mlp_hidden, weights=pretrained_weights, n_r=n_r, lw=lw, batch_size=batch_size, input_dropout=dropout_p, gpu=gpu)
else:
	raise Exception('Unknown model!')

model_name = 'models/ConceptNet/model.bin'
state = torch.load(model_name, map_location=lambda storage, loc: storage)
model.load_state_dict(state)

'''
relations = ['HasPainIntensity','HasPainCharacter','LocationOfAction','LocatedNear',
'DesireOf','NotMadeOf','InheritsFrom','InstanceOf','RelatedTo','NotDesires',
'NotHasA','NotIsA','NotHasProperty','NotCapableOf']
'''

with open(DATA_ROOT + 'word_id_map.dict','r') as f:
	word_id_map = json.load(f)

with open(DATA_ROOT + 'rel_id_map.dict','r') as f:
	rel_id_map = json.load(f)


evaluate_model(sub_, obj_, word_id_map, rel_id_map, eval_type='topfive')