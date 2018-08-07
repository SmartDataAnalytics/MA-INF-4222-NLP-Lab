from torch.utils.data import DataLoader
from utils import TripleDataset, sample_negatives, preprocess
import numpy as np
from model import DistMult
from torch.autograd import Variable
from evaluation import get_accuracy, auc, find_clf_threshold, stats
import argparse
import pdb

np.random.seed(4086)
torch.manual_seed(4086)

parser = argparse.ArgumentParser(
				description='Test Models for Commonsense KB Reasoning: Bilinear Averaging Model, Bilinear LSTM Model, \
					DistMult Averaging Model, DistMult LSTM Model, ER-MLP Averaging Model, ER-MLP LSTM Model'
)

parser.add_argument('--model', default='BilinearAvg', metavar='',
					help='model to run: {BilinearAvg, BilinearLstm, DistMultAvg, DistMultLstm, \
						ErmlpLstm, ErmlpAvg} (default: BilinearAvg)')
parser.add_argument('--test_file', type=str, default='dev2.txt', metavar='',
					help='test dataset to be used: {test.txt, dev2.txt} (default: dev2.txt)')
parser.add_argument('--rel_file', type=str, default='rel.txt', metavar='',
					help='file containing ConceptNet relation')
parser.add_argument('--pretrained_weights_file', default='embeddings.txt', type=str, help='name of pretrained weights file')
parser.add_argument('--k', type=int, default=150, metavar='',
					help='embedding relation dim (default: 150)')
parser.add_argument('--dropout_p', type=float, default=0.2, metavar='',
					help='Probability of dropping out neuron(default: 0.2)')
parser.add_argument('--mlp_hidden', type=int, default=100, metavar='',
					help='size of ER-MLP hidden layer (default: 100)')
parser.add_argument('--mb_size', type=int, default=100, metavar='',
					help='size of minibatch (default: 100)')
parser.add_argument('--negative_samples', type=int, default=3, metavar='',
					help='number of negative samples per positive sample  (default: 10)')
parser.add_argument('--nm_epoch', type=int, default=1000, metavar='',
					help='number of training epoch (default: 1000)')
parser.add_argument('--lr', type=float, default=0.01, metavar='',
					help='learning rate (default: 0.01)')
parser.add_argument('--lr_decay', type=float, default=1e-3, metavar='',
					help='decaying learning rate every n epoch (default: 1e-3)')
parser.add_argument('--weight_decay', type=float, default=1e-4, metavar='',
					help='L2 weight decay (default: 1e-4)')
parser.add_argument('--embeddings_lambda', type=float, default=1e-2, metavar='',
					help='prior strength for embeddings. Constraints embeddings norms to at most one  (default: 1e-2)')
parser.add_argument('--normalize_embed', default=False, type=bool, metavar='',
					help='whether to normalize embeddings to unit euclidean ball (default: False)')
parser.add_argument('--checkpoint_dir', default='models/', metavar='',
                    help='directory to save model checkpoint, saved every epoch (default: models/)')
parser.add_argument('--use_gpu', default=False, action='store_true',
					help='whether to run in the GPU')

args = parser.parse_args()

if args.use_gpu:
	torch.cuda.manual_seed(4086)

np.random.seed(4086)
torch.manual_seed(4086)

# Read and Prepare DataSet
DATA_ROOT = 'data/ConceptNet/'
test_file = DATA_ROOT+args.test_file
rel_file = DATA_ROOT+args.rel_file
pretrained_file = DATA_ROOT+args.pretrained_weights_file
# Prepare Test DataSet
preprocessor.read_test_triples(test_file)
test_triples = preprocessor.test_triples
test_idx = preprocessor.triple_to_index(test_triples, dev=True)
test_data = preprocessor.pad_idx_data(test_idx, dev=True)
test_label = test_data[3]
test_data = test_data[0], test_data[1], test_data[2]

# Parameters of Model
batch_size = args.mb_size
# Embedding Normalization
lw = args.embeddings_lambda
gpu = args.use_gpu
epochs = args.nm_epoch
sampling_factor = args.negative_samples
lr = args.lr
lr_decay = args.lr_decay
embedding_rel_dim = args.k
weight_decay = args.weight_decay
mlp_hidden = args.mlp_hidden
dropout_p = args.dropout_p
normalize_embed = args.normalize_embed

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

# Test Model
test_s, test_o, test_p = test_data
score_test = model.forward(test_s, test_o, test_p)
score_test = score_test.cpu().data.numpy() if gpu else score_test.data.numpy()
test_acc = get_accuracy(score_test, thresh)
test_auc_score = auc(score_test, test_label)

print('Test Accuracy: {0}'.format(test_acc))
print('Test AUC Score: {0}'.format(test_auc_score))
