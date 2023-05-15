import math

from arguments import get_args_parser
from templating import  get_semeval_temps, get_temps
from modeling import get_model, get_tokenizer
from data_prompt import REPromptDataset
from optimizing import get_optimizer
import torch
import torch.nn as nn
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
from tqdm import tqdm, trange
import numpy as np
from collections import Counter
import random
import os
import time
from sklearn.metrics import f1_score as F1_source


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

P_list = []
Recall_list = []
f1_list = []


def f1_score(output, label, rel_num, na_num):
    correct_by_relation = Counter()
    guess_by_relation = Counter()
    gold_by_relation = Counter()

    for i in range(len(output)):
        guess = output[i]
        gold = label[i]

        if guess == na_num:
            guess = 0
        elif guess < na_num:
            guess += 1

        if gold == na_num:
            gold = 0
        elif gold < na_num:
            gold += 1

        if gold == 0 and guess == 0:
            continue
        if gold == 0 and guess != 0:
            guess_by_relation[guess] += 1
        if gold != 0 and guess == 0:
            gold_by_relation[gold] += 1
        if gold != 0 and guess != 0:
            guess_by_relation[guess] += 1
            gold_by_relation[gold] += 1
            if gold == guess:
                correct_by_relation[gold] += 1

    f1_by_relation = Counter()
    recall_by_relation = Counter()
    prec_by_relation = Counter()
    for i in range(1, rel_num):
        recall = 0
        if gold_by_relation[i] > 0:
            recall = correct_by_relation[i] / gold_by_relation[i]
        precision = 0
        if guess_by_relation[i] > 0:
            precision = correct_by_relation[i] / guess_by_relation[i]
        if recall + precision > 0:
            f1_by_relation[i] = 2 * recall * precision / (recall + precision)
        recall_by_relation[i] = recall
        prec_by_relation[i] = precision

    micro_f1 = 0
    if sum(guess_by_relation.values()) != 0 and sum(correct_by_relation.values()) != 0:
        recall = sum(correct_by_relation.values()) / sum(gold_by_relation.values())
        prec = sum(correct_by_relation.values()) / sum(guess_by_relation.values())
        micro_f1 = 2 * recall * prec / (recall + prec)

        Recall_list.append(recall)
        P_list.append(prec)
        f1_list.append(micro_f1)

        print("recall: {}, prection: {}".format(recall, prec))

    return micro_f1, sum(f1_by_relation.values()) / (len(f1_by_relation) + 0.00000001)


def evaluate(model, dataset, dataloader):
    model.eval()
    scores = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            logits = model(**batch)
            res = []
            for i in dataset.prompt_id_2_label:
                _res = 0.0
                for j in range(len(i)):
                    _res += logits[j][:, i[j]]
                _res = _res.detach().cpu()
                res.append(_res)
            logits = torch.stack(res, 0).transpose(1, 0)  # torch.stack()沿着新的维度连接一系列数组，所有的数组都需要具有相同的大小
            labels = batch['labels'].detach().cpu().tolist()
            all_labels += labels
            scores.append(logits.cpu().detach())
        scores = torch.cat(scores, 0)
        scores = scores.detach().cpu().numpy()
        all_labels = np.array(all_labels)
        np.save("scores.npy", scores)
        np.save("all_labels.npy", all_labels)

        pred = np.argmax(scores, axis=-1)
        mi_f1, ma_f1 = f1_score(pred, all_labels, dataset.num_class, dataset.NA_NUM)
        print('his')
        print(mi_f1)
        print(ma_f1)
        miF1 = F1_source(all_labels.tolist(), pred.tolist(), labels=[i for i in range(dataset.num_class) if i != dataset.NA_NUM], average='micro')
        maF1 = F1_source(all_labels.tolist(), pred.tolist(), labels=[i for i in range(dataset.num_class) if i != dataset.NA_NUM], average='macro')
        # miF1 = F1_source(all_labels.tolist(), pred.tolist(), average='micro')
        # maF1 = F1_source(all_labels.tolist(), pred.tolist(), average='macro')
        return miF1, maF1


args = get_args_parser()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


set_seed(args.seed)

tokenizer = get_tokenizer(special=[])
temps = get_semeval_temps(tokenizer)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


train_dataset = REPromptDataset(
    path=args.data_dir,
    name='train.txt',
    rel2id=args.data_dir + "/" + "rel2id.json",
    temps=temps,
    tokenizer=tokenizer
)
# train_dataset.save(path=args.output_dir, name="train")

# val_dataset = REPromptDataset(
#     path=args.data_dir,
#     name='val.txt',
#     rel2id=args.data_dir + "/" + "rel2id.json",
#     temps=temps,
#     tokenizer=tokenizer
# )
# val_dataset.save(path=args.output_dir, name="val")

test_dataset = REPromptDataset(
    path=args.data_dir,
    name='test.txt',
    rel2id=args.data_dir + "/" + "rel2id.json",
    temps=temps,
    tokenizer=tokenizer
)
# test_dataset.save(path=args.output_dir, name="test")

train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

train_dataset.cuda()
train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)

# val_dataset.cuda()
# val_sampler = SequentialSampler(val_dataset)
# val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=train_batch_size // 2)

test_dataset.cuda()
test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=train_batch_size // 2)

model = get_model(tokenizer, train_dataset.prompt_label_idx)
optimizer, scheduler, optimizer_new_token, scheduler_new_token = get_optimizer(model, train_dataloader)
criterion = nn.CrossEntropyLoss()

mi_f1 = 0.0
hist_mi_f1 = []
hist_ma_f1 = []
mx_epoch = None
last_epoch = None
start_time = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime(time.time()))
max_f1 = 0.0

for epoch in range(int(args.num_train_epochs)):
    model.train()
    model.zero_grad()
    tr_loss = 0.0
    global_step = 0
    last_loss = 0
    avg_loss = 0
    with tqdm(train_dataloader) as tr_loop:
        for step, batch in enumerate(tr_loop):
            tr_loop.set_description(str(epoch + 1) + ' epoch')
            if len(batch['labels']) < args.per_gpu_train_batch_size:
                continue
            logits = model(**batch)
            labels = train_dataset.prompt_id_2_label[batch['labels']]

            loss = 0.0
            for index, i in enumerate(logits):
                loss += criterion(i, labels[:, index])
            loss /= len(logits)

            res = []
            for i in train_dataset.prompt_id_2_label:
                _res = 0.0
                for j in range(len(i)):
                    _res += logits[j][:, i[j]]
                res.append(_res)
            final_logits = torch.stack(res, 0).transpose(0, 1)

            loss += criterion(final_logits, batch['labels'])

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            if last_loss != 0:
                avg_loss += loss.item() - last_loss
            last_loss = loss.item()
            tr_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer_new_token.step()
                scheduler_new_token.step()
                model.zero_grad()
                # print(args)
                global_step += 1
                tr_loop.set_postfix({'loss': '{0:1.4f}'.format(tr_loss / global_step), 'down': '{0:1.4f}'.format(avg_loss / step if step != 0 else 0)})

    model.eval()
    mi_f1, ma_f1 = evaluate(model, test_dataset, test_dataloader)
    print('mi f1: ', mi_f1)
    print('ma f1: ', ma_f1)
    if mi_f1 > max_f1:
        max_f1 = mi_f1
    hist_mi_f1.append(mi_f1)
    hist_ma_f1.append(ma_f1)
    print("max_f1 = {}".format(max_f1))
    # if mi_f1 > mx_res:
    #     mx_res = mi_f1


print(hist_mi_f1)
print(hist_ma_f1)


print("P: {}".format(P_list))
print("R: {}".format(Recall_list))
print("F1: {}".format(F1_source))

