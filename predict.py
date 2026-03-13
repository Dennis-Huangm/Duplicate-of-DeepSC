# Denis
# coding:UTF-8
import argparse
import json

import torch
from torch.utils.data import DataLoader

from datasets import EurDataset, collate_data
from models import Transceiver
from train import val_epoch
from utils import build_transceiver_config, resolve_device, resolve_repo_path, validate_vocab


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100, help='the epochs of training')
    parser.add_argument('--batch-size', type=int, default=128, help='total batch size for all GPUs')
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu', help='device for evaluation')
    parser.add_argument('--ffn-num-input', type=int, default=128, help='ffn\'s input dim')
    parser.add_argument('--ffn-num-hiddens', type=int, default=256, help='the hidden size of transformers\'s ffn')
    parser.add_argument('--num-hiddens', type=int, default=128, help='the dimension of channel encoding')
    parser.add_argument('--key-size', type=int, default=128, help='the dimension of key')
    parser.add_argument('--query-size', type=int, default=128, help='the dimension of query')
    parser.add_argument('--value-size', type=int, default=128, help='the dimension of value')
    parser.add_argument('--num-layers', type=int, default=3, help='the layers of encoder and decoder')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--num_heads', type=int, default=8, help='multiple head of attention')
    parser.add_argument('--norm-shape', nargs='+', type=int, default=[128], help='layer norm shape values')
    parser.add_argument('--vocab', type=str, default='./content/vocab.json', help='path to vocab json')
    parser.add_argument('--checkpoint-path', type=str, default='./artifacts/model.pt', help='path to the saved checkpoint')
    parser.add_argument('--save-csv', action='store_true', help='save the result as csv file')
    parser.add_argument('--save-img', action='store_true', help='save the loss arc as img')
    return parser.parse_args()



def predict(opt):
    device = resolve_device(opt.device)
    vocab_path = resolve_repo_path(opt.vocab)
    checkpoint_path = resolve_repo_path(opt.checkpoint_path)

    test_datasets = EurDataset(split='test')
    test_loader = DataLoader(test_datasets, shuffle=True, batch_size=opt.batch_size, collate_fn=collate_data)
    with vocab_path.open('r', encoding='utf-8') as file:
        vocab = json.load(file)
        token_to_idx = validate_vocab(vocab)
        vocab_size = len(token_to_idx)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if 'model_state_dict' in checkpoint:
        model_config = checkpoint.get('model_config', build_transceiver_config(opt, vocab_size))
        checkpoint_vocab_size = checkpoint.get('vocab_size')
        checkpoint_start_token_id = checkpoint.get('start_token_id')
        current_start_token_id = token_to_idx['<START>']
        if checkpoint_vocab_size is not None and checkpoint_vocab_size != vocab_size:
            raise ValueError(
                f'Checkpoint vocab size ({checkpoint_vocab_size}) does not match current vocab size ({vocab_size}).'
            )
        if checkpoint_start_token_id is not None and checkpoint_start_token_id != current_start_token_id:
            raise ValueError('Checkpoint <START> token id does not match the current vocab.')
        model_state_dict = checkpoint['model_state_dict']
    else:
        model_config = build_transceiver_config(opt, vocab_size)
        model_state_dict = checkpoint

    transceiver = Transceiver(**model_config).to(device)
    try:
        transceiver.load_state_dict(model_state_dict)
    except RuntimeError as exc:
        raise RuntimeError(
            'Failed to load checkpoint. For legacy raw state_dict checkpoints, ensure the prediction '
            'architecture flags match the training configuration.'
        ) from exc

    l = val_epoch(transceiver, test_loader, device, vocab, 12)
    print(l)


if __name__ == '__main__':
    args = parse_opt()
    predict(args)
