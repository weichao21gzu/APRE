import torch
import torch.nn as nn
from arguments import get_model_classes, get_args
import numpy as np


class LabelGRUCell(nn.RNNCellBase):
    """自定义GRUCell"""
    def __init__(self, input_size, hidden_size, bias):
        super(LabelGRUCell, self).__init__(input_size, hidden_size, bias=True, num_chunks=1)
        # 输入变量的线性变换过程是 x @ W.T + b (@代表矩阵乘法， .T代表矩阵转置)
        # in2hid_w 的原始形状应是 (hidden_size, input_size), 为了编程的方便, 这里改成(input_size, hidden_size)torch
        # lb, ub = -torch.sqrt(torch.tensor((1/hidden_size))), torch.sqrt(torch.tensor((1/hidden_size)))
        # self.in2hid_w = nn.ParameterList([self.__init(lb, ub, input_size, hidden_size) for _ in range(3)])
        # self.hid2hid_w = nn.ParameterList([self.__init(lb, ub, hidden_size, hidden_size) for _ in range(3)])
        # self.in2hid_b = nn.ParameterList([self.__init(lb, ub, hidden_size) for _ in range(3)])
        # self.hid2hid_b = nn.ParameterList([self.__init(lb, ub, hidden_size) for _ in range(3)])

    # @staticmethod
    # def __init(low, upper, dim1, dim2=None):
    #     if dim2 is None:
    #         return nn.Parameter(torch.rand(dim1) * (upper - low) + low)  # 按照官方的初始化方法来初始化网络参数
    #     else:
    #         return nn.Parameter(torch.rand(dim1, dim2) * (upper - low) + low)

    def forward(self, x, hid):
        self.check_forward_input(x)
        if hid is None:
            hid = torch.zeros(x.size(0), self.hidden_size, dtype=x.dtype, device=x.device)
        self.check_forward_hidden(x, hid, '')

        # self.in2hid_w[0]代表Wr, x即是R描述输入， hid即原型P输入
        r = torch.sigmoid(torch.mm(x, self.weight_ih) + self.bias_ih +
                          torch.mm(hid, self.weight_ih) + self.bias_ih)
        r_remain = torch.mul((1-r), x)
        r_replace = torch.mul(r, x)

        # self.in2hid_w[1]代表Wp
        p = torch.sigmoid(torch.mm(r_replace, self.weight_hh) + self.bias_hh +
                          torch.mm(hid, self.weight_hh) + self.bias_hh)
        p_ori_remain = torch.mul(p, hid)
        # p_rec = r_remain + p_ori_remain
        p_rec = r_replace + hid

        return p_rec



class biaffine(nn.Module):
    def __init__(self, in_size, out_size, bias_x=True, bias_y=True):
        super().__init__()
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.out_size = out_size
        self.U = torch.nn.Parameter(torch.randn(in_size + int(bias_x), out_size, in_size + int(bias_y)))
        # self.U1 = self.U.view(size=(in_size + int(bias_x),-1))
        # U.shape = [in_size,out_size,in_size]

    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), dim=-1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), dim=-1)

        """
        batch_size,seq_len,hidden=x.shape
        bilinar_mapping=torch.matmul(x,self.U)
        bilinar_mapping=bilinar_mapping.view(size=(batch_size,seq_len*self.out_size,hidden))
        y=torch.transpose(y,dim0=1,dim1=2)
        bilinar_mapping=torch.matmul(bilinar_mapping,y)
        bilinar_mapping=bilinar_mapping.view(size=(batch_size,seq_len,self.out_size,seq_len))
        bilinar_mapping=torch.transpose(bilinar_mapping,dim0=2,dim1=3)
        """
        bilinar_mapping = torch.einsum('bi,ioj,bj->bo', x, self.U, y)
        return bilinar_mapping


class BiaffineSelfAttention(nn.Module):
    def __init__(self, input_dim, bias=True):
        super(BiaffineSelfAttention, self).__init__()
        self.input_dim = input_dim
        self.bias = bias
        self.W = nn.Parameter(torch.Tensor(input_dim, input_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(input_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        if self.bias:
            nn.init.zeros_(self.b)

    def forward(self, x, attn_mask=None):
        batch_size, seq_len, dim = x.shape
        attention = torch.einsum("ijk,kl->ijl", x, self.W)
        attention = torch.einsum("ijk,ijl->ikl", x, attention)
        if self.bias:
            attention = attention + self.b[None, None, :]
        if attn_mask:
            attention = attention.masked_fill(attn_mask, -1e9)
        attention = nn.functional.softmax(attention, dim=2)
        context = torch.bmm(x, attention.transpose(1, 2))
        return context, attention


class Model(torch.nn.Module):

    def __init__(self, args, tokenizer = None, prompt_label_idx = None):
        super().__init__()
        model_classes = get_model_classes()
        model_config = model_classes[args.model_type]

        # --------------- -----------------------------
        self.prompt_label_idx = prompt_label_idx

        self.model = model_config['model'].from_pretrained(
            args.model_name_or_path,
            return_dict=False,
            cache_dir=args.cache_dir if args.cache_dir else None)

        self.lstm = nn.LSTM(bidirectional=True, num_layers=2, input_size=1024, hidden_size=1024//2, batch_first=True)

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.model.config.hidden_size * 3, self.model.config.hidden_size),
            torch.nn.ReLU(),
            # nn.Dropout(p=args.dropout_prob),
            torch.nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size),
        )

        self.mlp2 = torch.nn.Sequential(
            torch.nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size),
            torch.nn.ReLU(),
            # nn.Dropout(p=args.dropout_prob),
            torch.nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size * 3),
        )

        self.new_biaffine = BiaffineSelfAttention(self.model.config.hidden_size, bias=True)

        self.classifier = torch.nn.Linear(self.model.config.hidden_size, 40)

        # self.biaffine_linear_head = torch.nn.Linear(self.model.config.hidden_size*3, self.model.config.hidden_size)
        # self.biaffine_linear_tail = torch.nn.Linear(self.model.config.hidden_size*3, self.model.config.hidden_size)
        #
        # self.biaffine = biaffine(self.model.config.hidden_size, 19, bias_x=False, bias_y=False)

        self.gru = LabelGRUCell(self.model.config.hidden_size, self.model.config.hidden_size, bias=True)

        self.extra_token_embeddings = nn.Embedding(args.new_tokens, self.model.config.hidden_size)

        self.temp_idx_list = []
        for temp_word_l in self.prompt_label_idx:
            for temp_word_idx in temp_word_l:
                self.temp_idx_list.append(temp_word_idx)
        self.temp_idx_list = torch.Tensor(np.array(self.temp_idx_list)).long()


    def forward(self, input_ids, attention_mask, token_type_ids, input_flags, mlm_labels, labels):

        raw_embeddings = self.model.embeddings.word_embeddings(input_ids)
        new_token_embeddings = self.extra_token_embeddings.weight
        new_embeddings = new_token_embeddings[input_flags]
        inputs_embeds = torch.where(input_flags.unsqueeze(-1) > 0, new_embeddings, raw_embeddings)


        hidden_states, cls = self.model(inputs_embeds=inputs_embeds,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids)

        # hidden_states1, _ = self.lstm(hidden_states)
        # hidden_states2 = self.mlp(hidden_states)
        # hidden_states = torch.add(hidden_states1, hidden_states2)
        # hidden_states = torch.div(hidden_states, 2)


        hidden_states = hidden_states[mlm_labels >= 0].view(input_ids.size(0), len(self.prompt_label_idx), -1)

        hidden_states, attention = self.new_biaffine(hidden_states)
        hidden_states = hidden_states.view(input_ids.size(0), -1)

        # hidden_states = self.mlp(hidden_states)
        # hidden_states = self.gatenet(cls, hidden_states)

        # hidden_states = hidden_states.view(input_ids.size(0), -1)

        hidden_states = self.mlp(hidden_states)
        hidden_states = self.gru(cls, hidden_states)
        hidden_states = self.mlp2(hidden_states)
        hidden_states = hidden_states.view(input_ids.size(0), len(self.prompt_label_idx),-1)


        logits = [
            torch.mm(
                hidden_states[:,index,:],
                self.model.embeddings.word_embeddings.weight[i].transpose(0,1)
            )
            for index, i in enumerate(self.prompt_label_idx)
        ]

        # print(self.model.embeddings.word_embeddings.weight[self.prompt_label_idx[0]].shape[0],
        #       self.model.embeddings.word_embeddings.weight[self.prompt_label_idx[0]].shape[1])


        # logits = [
        #     torch.cdist(
        #         hidden_states[:, index, :].unsqueeze(1),
        #         self.model.embeddings.word_embeddings.weight[i].unsqueeze(0),
        #         p=2,
        #         compute_mode='use_mm_for_euclid_dist'
        #     ).squeeze(1)
        #     for index, i in enumerate(self.prompt_label_idx)
        # ]

        return logits

def get_model(tokenizer, prompt_label_idx):
    args = get_args()
    model = Model(args, tokenizer, prompt_label_idx)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.cuda()
    return model

def get_tokenizer(special=[]):
    args = get_args()
    model_classes = get_model_classes()
    model_config = model_classes[args.model_type]
    tokenizer = model_config['tokenizer'].from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None)
    tokenizer.add_tokens(special)
    return tokenizer

