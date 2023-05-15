import argparse
import torch
import transformers
from transformers import BertConfig, BertTokenizer, BertModel, \
                         RobertaConfig, RobertaTokenizer, RobertaModel, \
                         AlbertTokenizer, AlbertConfig, AlbertModel

_GLOBAL_ARGS = None

_MODEL_CLASSES = {
    'bert': {
        'config': BertConfig,
        'tokenizer': BertTokenizer,
        'model':BertModel,
    },
    'roberta': {
        'config': RobertaConfig,
        'tokenizer': RobertaTokenizer,
        'model':RobertaModel,
    },
    'albert': {
        'config': AlbertConfig,
        'tokenizer': AlbertTokenizer,
        'model':AlbertModel,
    }
}

# import torch
# from transformers import BertTokenizer, BertModel
# tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
# roberta = BertModel.from_pretrained("hfl/chinese-roberta-wwm-ext")

def get_args_parser():

    parser = argparse.ArgumentParser(description="Command line interface for Relation Extraction.")

    # Required parameters
    parser.add_argument("--data_dir", default='../datasets/semeval', type=str, required=False,
                        help="The input data dir. Should contain the data files for the task.")
    parser.add_argument("--model_type", default='roberta', type=str, required=False, choices=_MODEL_CLASSES.keys(),
                        help="The type of the pretrained language model to use")
    parser.add_argument("--model_name_or_path", default="roberta-large", type=str, required=False,
                        help="Path to the pre-trained model or shortcut name")
    parser.add_argument("--output_dir", default='../results/tacred', type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written")

    parser.add_argument("--new_tokens", default=4, type=int,
                        help="The output directory where the model predictions and checkpoints will be written")

    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization for PET. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--per_gpu_train_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for PET training.")
    # parser.add_argument("--per_gpu_eval_batch_size", default=32, type=int,
    #                     help="Batch size per GPU/CPU for PET evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                        help="Number of updates steps to accumulate before performing a backward/update pass in PET.")
    parser.add_argument("--num_train_epochs", default=30, type=float,
                        help="Total number of training epochs to perform in PET.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform in PET. Override num_train_epochs.")

    # Other optional parameters
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where to store the pre-trained models downloaded from S3.")
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--learning_rate_for_new_token", default=1e-4, type=float,
                        help="The initial learning rate for Adam.")

    parser.add_argument("--weight_decay", default=1e-2, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0.1, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--seed', type=int, default=31415926,
                        help="random seed for initialization")
    parser.add_argument("--dropout_prob", type=float, default=0.1)
    parser.add_argument("--temps", default="temp.txt", type=str,
                        help="Where to store the pre-trained models downloaded from S3.")



    args = parser.parse_args()
    args.n_gpu = torch.cuda.device_count()

    global _GLOBAL_ARGS
    _GLOBAL_ARGS = args
    return args

def get_args():
    return get_args_parser()

def get_model_classes():
    return _MODEL_CLASSES