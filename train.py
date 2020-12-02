import sys

sys.path.append('../')

import os
import time
import yaml
import random
import numpy as np
import warnings
import logging
import pickle
from datetime import datetime
from tqdm import tqdm_notebook as tqdm

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import torch
from torch import einsum
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from src import utils, messenger

config_file = '../configs/config_code2seq.yml'

config = yaml.load(open(config_file), Loader=yaml.FullLoader)

# Data source
DATA_HOME = config['data']['home']
DICT_FILE = DATA_HOME + config['data']['dict']
TRAIN_DIR = DATA_HOME + config['data']['train']
VALID_DIR = DATA_HOME + config['data']['valid']
TEST_DIR = DATA_HOME + config['data']['test']

# Training parameter
batch_size = config['training']['batch_size']
num_epochs = config['training']['num_epochs']
lr = config['training']['lr']
teacher_forcing_rate = config['training']['teacher_forcing_rate']
nesterov = config['training']['nesterov']
weight_decay = config['training']['weight_decay']
momentum = config['training']['momentum']
decay_ratio = config['training']['decay_ratio']
save_name = config['training']['save_name']
warm_up = config['training']['warm_up']
patience = config['training']['patience']

# Model parameter
token_size = config['model']['token_size']
hidden_size = config['model']['hidden_size']
num_layers = config['model']['num_layers']
bidirectional = config['model']['bidirectional']
rnn_dropout = config['model']['rnn_dropout']
embeddings_dropout = config['model']['embeddings_dropout']
num_k = config['model']['num_k']

# etc
slack_url_path = config['etc']['slack_url_path']
info_prefix = config['etc']['info_prefix']

slack_url = None
if os.path.exists(slack_url_path):
    slack_url = yaml.load(open(slack_url_path), Loader=yaml.FullLoader)['slack_url']

warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(1)
random_state = 42

run_id = datetime.now().strftime('%Y-%m-%d--%H-%M-%S')
log_file = '../../logs/' + run_id + '.log'
exp_dir = '../../runs/' + run_id
os.mkdir(exp_dir)

logging.basicConfig(format='%(asctime)s | %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', filename=log_file,
                    level=logging.DEBUG)
msgr = messenger.Info(info_prefix, slack_url)

msgr.print_msg('run_id : {}'.format(run_id))
msgr.print_msg('log_file : {}'.format(log_file))
msgr.print_msg('exp_dir : {}'.format(exp_dir))
msgr.print_msg('device : {}'.format(device))
msgr.print_msg(str(config))

PAD_TOKEN = '<PAD>'
BOS_TOKEN = '<S>'
EOS_TOKEN = '</S>'
UNK_TOKEN = '<UNK>'
PAD = 0
BOS = 1
EOS = 2
UNK = 3

# load vocab dict
with open(DICT_FILE, 'rb') as file:
    subtoken_to_count = pickle.load(file)
    node_to_count = pickle.load(file)
    target_to_count = pickle.load(file)
    max_contexts = pickle.load(file)
    num_training_examples = pickle.load(file)
    msgr.print_msg('Dictionaries loaded.')

# making vocab dicts for terminal subtoken, nonterminal node and target.

word2id = {
    PAD_TOKEN: PAD,
    BOS_TOKEN: BOS,
    EOS_TOKEN: EOS,
    UNK_TOKEN: UNK,
}

vocab_subtoken = utils.Vocab(word2id=word2id)
vocab_nodes = utils.Vocab(word2id=word2id)
vocab_target = utils.Vocab(word2id=word2id)

vocab_subtoken.build_vocab(list(subtoken_to_count.keys()), min_count=0)
vocab_nodes.build_vocab(list(node_to_count.keys()), min_count=0)
vocab_target.build_vocab(list(target_to_count.keys()), min_count=0)

vocab_size_subtoken = len(vocab_subtoken.id2word)
vocab_size_nodes = len(vocab_nodes.id2word)
vocab_size_target = len(vocab_target.id2word)

msgr.print_msg('vocab_size_subtoken：' + str(vocab_size_subtoken))
msgr.print_msg('vocab_size_nodes：' + str(vocab_size_nodes))
msgr.print_msg('vocab_size_target：' + str(vocab_size_target))

num_length_train = num_training_examples
msgr.print_msg('num_examples : ' + str(num_length_train))


class DataLoader(object):

    def __init__(self, data_path, batch_size, num_k, vocab_subtoken, vocab_nodes, vocab_target, shuffle=True,
                 batch_time=False):

        """
        data_path : path for data
        num_examples : total lines of data file
        batch_size : batch size
        num_k : max ast pathes included to one examples
        vocab_subtoken : dict of subtoken and its id
        vocab_nodes : dict of node simbol and its id
        vocab_target : dict of target simbol and its id
        """

        self.data_path = data_path
        self.batch_size = batch_size

        self.num_examples = self.file_count(data_path)
        self.num_k = num_k

        self.vocab_subtoken = vocab_subtoken
        self.vocab_nodes = vocab_nodes
        self.vocab_target = vocab_target

        self.index = 0
        self.pointer = np.array(range(self.num_examples))
        self.shuffle = shuffle

        self.batch_time = batch_time

        self.reset()

    def __iter__(self):
        return self

    def __next__(self):

        if self.batch_time:
            t1 = time.time()

        if self.index >= self.num_examples:
            self.reset()
            raise StopIteration()

        ids = self.pointer[self.index: self.index + self.batch_size]
        seqs_S, seqs_N, seqs_E, seqs_Y = self.read_batch(ids)

        # length_k : (batch_size, k)
        lengths_k = [len(ex) for ex in seqs_N]

        # flattening (batch_size, k, l) to (batch_size * k, l)
        # this is useful to make torch.tensor
        seqs_S = [symbol for k in seqs_S for symbol in k]
        seqs_N = [symbol for k in seqs_N for symbol in k]
        seqs_E = [symbol for k in seqs_E for symbol in k]

        # Padding
        lengths_S = [len(s) for s in seqs_S]
        lengths_N = [len(s) for s in seqs_N]
        lengths_E = [len(s) for s in seqs_E]
        lengths_Y = [len(s) for s in seqs_Y]

        max_length_S = max(lengths_S)
        max_length_N = max(lengths_N)
        max_length_E = max(lengths_E)
        max_length_Y = max(lengths_Y)

        padded_S = [utils.pad_seq(s, max_length_S) for s in seqs_S]
        padded_N = [utils.pad_seq(s, max_length_N) for s in seqs_N]
        padded_E = [utils.pad_seq(s, max_length_E) for s in seqs_E]
        padded_Y = [utils.pad_seq(s, max_length_Y) for s in seqs_Y]

        # index for split (batch_size * k, l) into (batch_size, k, l)
        index_N = range(len(lengths_N))

        # sort for rnn
        seq_pairs = sorted(zip(lengths_N, index_N, padded_N, padded_S, padded_E), key=lambda p: p[0], reverse=True)
        lengths_N, index_N, padded_N, padded_S, padded_E = zip(*seq_pairs)

        batch_S = torch.tensor(padded_S, dtype=torch.long, device=device)
        batch_E = torch.tensor(padded_E, dtype=torch.long, device=device)

        # transpose for rnn
        batch_N = torch.tensor(padded_N, dtype=torch.long, device=device).transpose(0, 1)
        batch_Y = torch.tensor(padded_Y, dtype=torch.long, device=device).transpose(0, 1)

        # update index
        self.index += self.batch_size

        if self.batch_time:
            t2 = time.time()
            elapsed_time = t2 - t1
            print(f"batching time：0.0535")

        return batch_S, batch_N, batch_E, batch_Y, lengths_S, lengths_N, lengths_E, lengths_Y, max_length_S, max_length_N, max_length_E, max_length_Y, lengths_k, index_N

    def reset(self):
        if self.shuffle:
            self.pointer = shuffle(self.pointer)
        self.index = 0

    def file_count(self, path):
        lst = [name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))]
        return len(lst)

    def read_batch(self, ids):

        seqs_S = []
        seqs_E = []
        seqs_N = []
        seqs_Y = []

        for i in ids:
            path = self.data_path + '/{:0>6d}.txt'.format(i)
            with open(path, 'r') as f:
                seq_S = []
                seq_N = []
                seq_E = []

                target, *syntax_path = f.readline().split(' ')
                target = target.split('|')
                target = utils.sentence_to_ids(self.vocab_target, target)

                # remove '' and '\n' in sequence, java-small dataset contains many '' in a line.
                syntax_path = [s for s in syntax_path if s != '' and s != '\n']

                # if the amount of ast path exceed the k,
                # uniformly sample ast pathes, as described in the paper.
                if len(syntax_path) > self.num_k:
                    sampled_path_index = random.sample(range(len(syntax_path)), self.num_k)
                else:
                    sampled_path_index = range(len(syntax_path))

                for j in sampled_path_index:
                    terminal1, ast_path, terminal2 = syntax_path[j].split(',')

                    terminal1 = utils.sentence_to_ids(self.vocab_subtoken, terminal1.split('|'))
                    ast_path = utils.sentence_to_ids(self.vocab_nodes, ast_path.split('|'))
                    terminal2 = utils.sentence_to_ids(self.vocab_subtoken, terminal2.split('|'))

                    seq_S.append(terminal1)
                    seq_E.append(terminal2)
                    seq_N.append(ast_path)

                seqs_S.append(seq_S)
                seqs_E.append(seq_E)
                seqs_N.append(seq_N)
                seqs_Y.append(target)

        return seqs_S, seqs_N, seqs_E, seqs_Y


class Encoder(nn.Module):
    def __init__(self, input_size_subtoken, input_size_node, token_size, hidden_size, bidirectional=True, num_layers=2,
                 rnn_dropout=0.5, embeddings_dropout=0.25):
        """
        input_size_subtoken : # of unique subtoken
        input_size_node : # of unique node symbol
        token_size : embedded token size
        hidden_size : size of initial state of decoder
        rnn_dropout = 0.5 : rnn drop out ratio
        embeddings_dropout = 0.25 : dropout ratio for context vector
        """

        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.token_size = token_size

        self.embedding_subtoken = nn.Embedding(input_size_subtoken, token_size, padding_idx=PAD)
        self.embedding_node = nn.Embedding(input_size_node, token_size, padding_idx=PAD)

        self.lstm = nn.LSTM(token_size, token_size, num_layers=num_layers, bidirectional=bidirectional,
                            dropout=rnn_dropout)
        self.out = nn.Linear(token_size * 4, hidden_size)

        self.dropout = nn.Dropout(embeddings_dropout)
        self.num_directions = 2 if bidirectional else 1
        self.num_layers = num_layers

    def forward(self, batch_S, batch_N, batch_E, lengths_k, index_N, hidden=None):
        """
        batch_S : (B * k, l) start terminals' subtoken of each ast path
        batch_N : (l, B*k) nonterminals' nodes of each ast path
        batch_E : (B * k, l) end terminals' subtoken of each ast path

        lengths_k : length of k in each example
        index_N : index for unsorting,
        """

        bk_size = batch_N.shape[1]
        output_bag = []
        hidden_batch = []

        # (B * k, l, d)
        encode_S = self.embedding_subtoken(batch_S)
        encode_E = self.embedding_subtoken(batch_E)

        # encode_S (B * k, d) token_representation of each ast path
        encode_S = encode_S.sum(1)
        encode_E = encode_E.sum(1)

        """
        LSTM Outputs: output, (h_n, c_n)
        output (seq_len, batch, num_directions * hidden_size)
        h_n    (num_layers * num_directions, batch, hidden_size) : tensor containing the hidden state for t = seq_len.
        c_n    (num_layers * num_directions, batch, hidden_size)
        """

        # emb_N :(l, B*k, d)
        emb_N = self.embedding_node(batch_N)
        packed = pack_padded_sequence(emb_N, lengths_N)
        output, (hidden, cell) = self.lstm(packed, hidden)
        # output, _ = pad_packed_sequence(output)

        # hidden (num_layers * num_directions, batch, hidden_size)
        # only last layer, (num_directions, batch, hidden_size)
        hidden = hidden[-self.num_directions:, :, :]

        # -> (Bk, num_directions, hidden_size)
        hidden = hidden.transpose(0, 1)

        # -> (Bk, 1, hidden_size * num_directions)
        hidden = hidden.contiguous().view(bk_size, 1, -1)

        # encode_N (Bk, hidden_size * num_directions)
        encode_N = hidden.squeeze(1)

        # encode_SNE  : (B*k, hidden_size * num_directions + 2)
        encode_SNE = torch.cat([encode_N, encode_S, encode_E], dim=1)

        # encode_SNE  : (B*k, d)
        encode_SNE = self.out(encode_SNE)

        # unsort as example
        # index = torch.tensor(index_N, dtype=torch.long, device=device)
        # encode_SNE = torch.index_select(encode_SNE, dim=0, index=index)
        index = np.argsort(index_N)
        encode_SNE = encode_SNE[[index]]

        # as is in  https://github.com/tech-srl/code2seq/blob/ec0ae309efba815a6ee8af88301479888b20daa9/model.py#L511
        encode_SNE = self.dropout(encode_SNE)

        # output_bag  : [ B, (k, d) ]
        output_bag = torch.split(encode_SNE, lengths_k, dim=0)

        # hidden_0  : (1, B, d)
        # for decoder initial state
        hidden_0 = [ob.mean(0).unsqueeze(dim=0) for ob in output_bag]
        hidden_0 = torch.cat(hidden_0, dim=0).unsqueeze(dim=0)

        return output_bag, hidden_0


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, rnn_dropout):
        """
        hidden_size : decoder unit size,
        output_size : decoder output size,
        rnn_dropout : dropout ratio for rnn
        """

        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.embedding = nn.Embedding(output_size, hidden_size, padding_idx=PAD)
        self.gru = nn.GRU(hidden_size, hidden_size, dropout=rnn_dropout)
        self.out = nn.Linear(hidden_size * 2, output_size)

    def forward(self, seqs, hidden, attn):
        emb = self.embedding(seqs)
        _, hidden = self.gru(emb, hidden)

        output = torch.cat((hidden, attn), 2)
        output = self.out(output)

        return output, hidden


class EncoderDecoder_with_Attention(nn.Module):
    """Conbine Encoder and Decoder"""

    def __init__(self, input_size_subtoken, input_size_node, token_size, output_size, hidden_size, bidirectional=True,
                 num_layers=2, rnn_dropout=0.5, embeddings_dropout=0.25):

        super(EncoderDecoder_with_Attention, self).__init__()
        self.encoder = Encoder(input_size_subtoken, input_size_node, token_size, hidden_size,
                               bidirectional=bidirectional, num_layers=num_layers, rnn_dropout=rnn_dropout,
                               embeddings_dropout=embeddings_dropout)
        self.decoder = Decoder(hidden_size, output_size, rnn_dropout)

        self.W_a = torch.rand((hidden_size, hidden_size), dtype=torch.float, device=device, requires_grad=True)

        nn.init.xavier_uniform_(self.W_a)

    def forward(self, batch_S, batch_N, batch_E, lengths_S, lengths_N, lengths_E, lengths_Y, max_length_S, max_length_N,
                max_length_E, max_length_Y, lengths_k, index_N, terget_max_length, batch_Y=None,
                use_teacher_forcing=False):

        # Encoder
        encoder_output_bag, encoder_hidden = \
            self.encoder(batch_S, batch_N, batch_E, lengths_k, index_N)

        _batch_size = len(encoder_output_bag)
        decoder_hidden = encoder_hidden

        # make initial input for decoder
        decoder_input = torch.tensor([BOS] * _batch_size, dtype=torch.long, device=device)
        decoder_input = decoder_input.unsqueeze(0)  # (1, batch_size)

        # output holder
        decoder_outputs = torch.zeros(terget_max_length, _batch_size, self.decoder.output_size, device=device)

        # print('=' * 20)
        for t in range(terget_max_length):

            # ct
            ct = self.attention(encoder_output_bag, decoder_hidden, lengths_k)

            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, ct)

            # print(decoder_output.max(-1)[1])

            decoder_outputs[t] = decoder_output

            # Teacher Forcing
            if use_teacher_forcing and batch_Y is not None:
                decoder_input = batch_Y[t].unsqueeze(0)
            else:
                decoder_input = decoder_output.max(-1)[1]

        return decoder_outputs

    def attention(self, encoder_output_bag, hidden, lengths_k):

        """
        encoder_output_bag : (batch, k, hidden_size) bag of embedded ast path
        hidden : (1 , batch, hidden_size):
        lengths_k : (batch, 1) length of k in each example
        """

        # e_out : (batch * k, hidden_size)
        e_out = torch.cat(encoder_output_bag, dim=0)

        # e_out : (batch * k(i), hidden_size(j))
        # self.W_a  : [hidden_size(j), hidden_size(k)]
        # ha -> : [batch * k(i), hidden_size(k)]
        ha = einsum('ij,jk->ik', e_out, self.W_a)

        # ha -> : [batch, (k, hidden_size)]
        ha = torch.split(ha, lengths_k, dim=0)

        # dh = [batch, (1, hidden_size)]
        hd = hidden.transpose(0, 1)
        hd = torch.unbind(hd, dim=0)

        # _ha : (k(i), hidden_size(j))
        # _hd : (1(k), hidden_size(j))
        # at : [batch, ( k(i) ) ]
        at = [F.softmax(torch.einsum('ij,kj->i', _ha, _hd), dim=0) for _ha, _hd in zip(ha, hd)]

        # a : ( k(i) )
        # e : ( k(i), hidden_size(j))
        # ct : [batch, (hidden_size(j)) ] -> [batch, (1, hidden_size) ]
        ct = [torch.einsum('i,ij->j', a, e).unsqueeze(0) for a, e in zip(at, encoder_output_bag)]

        # ct [batch, hidden_size(k)]
        # -> (1, batch, hidden_size)
        ct = torch.cat(ct, dim=0).unsqueeze(0)

        return ct


mce = nn.CrossEntropyLoss(size_average=False, ignore_index=PAD)


def masked_cross_entropy(logits, target):
    return mce(logits.view(-1, logits.size(-1)), target.view(-1))


batch_time = False
train_dataloader = DataLoader(TRAIN_DIR, batch_size, num_k, vocab_subtoken, vocab_nodes, vocab_target,
                              batch_time=batch_time, shuffle=True)
valid_dataloader = DataLoader(VALID_DIR, batch_size, num_k, vocab_subtoken, vocab_nodes, vocab_target, shuffle=False)

model_args = {
    'input_size_subtoken': vocab_size_subtoken,
    'input_size_node': vocab_size_nodes,
    'output_size': vocab_size_target,
    'hidden_size': hidden_size,
    'token_size': token_size,
    'bidirectional': bidirectional,
    'num_layers': num_layers,
    'rnn_dropout': rnn_dropout,
    'embeddings_dropout': embeddings_dropout
}

model = EncoderDecoder_with_Attention(**model_args).to(device)

# optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov = nesterov)
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: decay_ratio ** epoch)

fname = exp_dir + save_name
early_stopping = utils.EarlyStopping(fname, patience, warm_up, verbose=True)


def compute_loss(batch_S, batch_N, batch_E, batch_Y, lengths_S, lengths_N, lengths_E, lengths_Y, max_length_S,
                 max_length_N, max_length_E, max_length_Y, lengths_k, index_N, model, optimizer=None, is_train=True):
    model.train(is_train)

    use_teacher_forcing = is_train and (random.random() < teacher_forcing_rate)

    target_max_length = batch_Y.size(0)
    pred_Y = model(batch_S, batch_N, batch_E, lengths_S, lengths_N, lengths_E, lengths_Y, max_length_S, max_length_N,
                   max_length_E, max_length_Y, lengths_k, index_N, target_max_length, batch_Y, use_teacher_forcing)

    loss = masked_cross_entropy(pred_Y.contiguous(), batch_Y.contiguous())

    if is_train:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    batch_Y = batch_Y.transpose(0, 1).contiguous().data.cpu().tolist()
    pred = pred_Y.max(dim=-1)[1].data.cpu().numpy().T.tolist()

    return loss.item(), batch_Y, pred


#
# Training Loop
#
progress_bar = False  # progress bar is visible in progress_bar = False

for epoch in range(1, num_epochs + 1):
    print('运行开始:')
    print(epoch)
    train_loss = 0.
    train_refs = []
    train_hyps = []
    valid_loss = 0.
    valid_refs = []
    valid_hyps = []

    # train

    for batch in tqdm(train_dataloader, total=train_dataloader.num_examples // train_dataloader.batch_size + 1,
                      desc='TRAIN'):
        print('第一次训练开始。。。')
        batch_S, batch_N, batch_E, batch_Y, lengths_S, lengths_N, lengths_E, lengths_Y, max_length_S, max_length_N, max_length_E, max_length_Y, lengths_k, index_N = batch

        loss, gold, pred = compute_loss(
            batch_S, batch_N, batch_E, batch_Y,
            lengths_S, lengths_N, lengths_E, lengths_Y,
            max_length_S, max_length_N, max_length_E, max_length_Y,
            lengths_k, index_N, model, optimizer,
            is_train=True
        )

        train_loss += loss
        train_refs += gold
        train_hyps += pred

    # valid
    for batch in tqdm(valid_dataloader, total=valid_dataloader.num_examples // valid_dataloader.batch_size + 1,
                      desc='VALID'):
        batch_S, batch_N, batch_E, batch_Y, lengths_S, lengths_N, lengths_E, lengths_Y, max_length_S, max_length_N, max_length_E, max_length_Y, lengths_k, index_N = batch

        loss, gold, pred = compute_loss(
            batch_S, batch_N, batch_E, batch_Y,
            lengths_S, lengths_N, lengths_E, lengths_Y,
            max_length_S, max_length_N, max_length_E, max_length_Y,
            lengths_k, index_N, model, optimizer,
            is_train=False
        )

        valid_loss += loss
        valid_refs += gold
        valid_hyps += pred

    train_loss = np.sum(train_loss) / train_dataloader.num_examples
    valid_loss = np.sum(valid_loss) / valid_dataloader.num_examples

    # F1 etc
    train_precision, train_recall, train_f1 = utils.calculate_results_set(train_refs, train_hyps)
    valid_precision, valid_recall, valid_f1 = utils.calculate_results_set(valid_refs, valid_hyps)

    early_stopping(valid_f1, model, epoch)
    if early_stopping.early_stop:
        msgr.print_msg("Early stopping")
        break

    msgr.print_msg('Epoch {}: train_loss: {:5.2f}  train_f1: {:2.4f}  valid_loss: {:5.2f}  valid_f1: {:2.4f}'.format(
        epoch, train_loss, train_f1, valid_loss, valid_f1))

    print('-' * 80)

    scheduler.step()

# evaluation
print('训练结束，开始评估')
model = EncoderDecoder_with_Attention(**model_args).to(device)

fname = exp_dir + save_name
ckpt = torch.load(fname)
model.load_state_dict(ckpt)

model.eval()

test_dataloader = DataLoader(TEST_DIR, batch_size, num_k, vocab_subtoken, vocab_nodes, vocab_target,
                             batch_time=batch_time, shuffle=True)

refs_list = []
hyp_list = []

for batch in tqdm(test_dataloader,
                  total=test_dataloader.num_examples // test_dataloader.batch_size + 1,
                  desc='TEST'):
    batch_S, batch_N, batch_E, batch_Y, lengths_S, lengths_N, lengths_E, lengths_Y, max_length_S, max_length_N, max_length_E, max_length_Y, lengths_k, index_N = batch
    target_max_length = batch_Y.size(0)
    use_teacher_forcing = False

    pred_Y = model(batch_S, batch_N, batch_E, lengths_S, lengths_N, lengths_E, lengths_Y, max_length_S, max_length_N,
                   max_length_E, max_length_Y, lengths_k, index_N, target_max_length, batch_Y, use_teacher_forcing)

    refs = batch_Y.transpose(0, 1).contiguous().data.cpu().tolist()[0]
    pred = pred_Y.max(dim=-1)[1].data.cpu().numpy().T.tolist()[0]

    refs_list.append(refs)
    hyp_list.append(pred)

msgr.print_msg('Tested model : ' + fname)

test_precision, test_recall, test_f1 = utils.calculate_results(refs_list, hyp_list)
msgr.print_msg('Test : precision {:1.5f}, recall {:1.5f}, f1 {:1.5f}'.format(test_precision, test_recall, test_f1))

test_precision, test_recall, test_f1 = utils.calculate_results_set(refs_list, hyp_list)
msgr.print_msg('Test(set) : precision {:1.5f}, recall {:1.5f}, f1 {:1.5f}'.format(test_precision, test_recall, test_f1))

batch_time = False
test_dataloader = DataLoader(TEST_DIR, 1, num_k, vocab_subtoken, vocab_nodes, vocab_target, batch_time=batch_time,
                             shuffle=True)

model.eval()

batch_S, batch_N, batch_E, batch_Y, lengths_S, lengths_N, lengths_E, lengths_Y, max_length_S, max_length_N, max_length_E, max_length_Y, lengths_k, index_N = next(
    test_dataloader)

sentence_Y = ' '.join(utils.ids_to_sentence(vocab_target, batch_Y.data.cpu().numpy()[:-1, 0]))
msgr.print_msg('tgt: {}'.format(sentence_Y))

target_max_length = batch_Y.size(0)
use_teacher_forcing = False
output = model(batch_S, batch_N, batch_E, lengths_S, lengths_N, lengths_E, lengths_Y, max_length_S, max_length_N,
               max_length_E, max_length_Y, lengths_k, index_N, target_max_length, batch_Y, use_teacher_forcing)

output = output.max(dim=-1)[1].view(-1).data.cpu().tolist()
output_sentence = ' '.join(utils.ids_to_sentence(vocab_target, utils.trim_eos(output)))
msgr.print_msg('out: {}'.format(output_sentence))