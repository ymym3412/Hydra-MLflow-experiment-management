import logging
import os

import chariot.transformer as ct
import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from allennlp.common import Params
from allennlp.data import Instance
from allennlp.data.fields import LabelField, TextField
from allennlp.data.iterators import BucketIterator
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.seq2seq_encoders import (PytorchSeq2SeqWrapper,
                                               Seq2SeqEncoder)
from allennlp.modules.text_field_embedders import (BasicTextFieldEmbedder,
                                                   TextFieldEmbedder)
from allennlp.modules.token_embedders import Embedding
from allennlp.training.trainer import Trainer
from chariot.preprocessor import Preprocessor
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score, precision_score, recall_score
from tqdm import tqdm

from mlflow_writer import MlflowWriter
from model.model import ClassifierWithAttn

logger = logging.getLogger(__name__)


# AllenNLP用に文章からInstanceを生成する
def text_to_instance(word_list, label):
    tokens = [Token(word) for word in word_list]
    word_sentence_field = TextField(tokens, {"tokens": SingleIdTokenIndexer()})
    fields = {"tokens": word_sentence_field}
    if label is not None:
        label_field = LabelField(label, skip_indexing=True)
        fields["label"] = label_field
    return Instance(fields)


def load_dataset(path, dataset):
    if dataset not in ['train', 'val', 'test']:
        raise ValueError('"dataset" parametes must be train/val/test')

    data, labels = pd.read_csv(f'{path}/{dataset}.csv'), pd.read_csv(f'{path}/{dataset}_label.csv', header=None, squeeze=True)
    return data, labels


# 学習
def train(train_dataset, val_dataset, cfg):
    # Vocabularyを生成
    VOCAB_SIZE = cfg.w2v.vocab_size
    vocab = Vocabulary.from_instances(train_dataset + val_dataset, max_vocab_size=VOCAB_SIZE)

    BATCH_SIZE = cfg.training.batch_size

    # パディング済みミニバッチを生成してくれるIterator
    iterator = BucketIterator(batch_size=BATCH_SIZE, sorting_keys=[("tokens", "num_tokens")])
    iterator.index_with(vocab)

    # 東北大が提供している学習済み日本語 Wikipedia エンティティベクトルを使用する
    # http://www.cl.ecei.tohoku.ac.jp/~m-suzuki/jawiki_vector/
    model_name = cfg.w2v.model_name
    norm = cfg.w2v.norm
    cwd = hydra.utils.get_original_cwd()
    params = Params({
          'embedding_dim': 200,
          'padding_index': 0,
          'pretrained_file': os.path.join(cwd, f'embs/jawiki.{model_name}_vectors.200d.txt'),
          'norm_type': norm})

    token_embedding = Embedding.from_params(vocab=vocab, params=params)
    HIDDEN_SIZE = cfg.model.hidden_size
    dropout = cfg.model.dropout

    word_embeddings: TextFieldEmbedder = BasicTextFieldEmbedder({"tokens": token_embedding})
    encoder: Seq2SeqEncoder = PytorchSeq2SeqWrapper(nn.LSTM(word_embeddings.get_output_dim(),
                                                    HIDDEN_SIZE, bidirectional=True, batch_first=True))
    model = ClassifierWithAttn(word_embeddings, encoder, vocab, dropout)
    model.train()

    USE_GPU = True

    if USE_GPU and torch.cuda.is_available():
        model = model.cuda(0)

    LR = cfg.training.learning_rate
    EPOCHS = cfg.training.epoch
    patience = cfg.training.patience if cfg.training.patience > 0 else None

    optimizer = optim.Adam(model.parameters(), lr=LR)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        iterator=iterator,
        train_dataset=train_dataset,
        validation_dataset=val_dataset,
        patience=patience,
        cuda_device=0 if USE_GPU else -1,
        num_epochs=EPOCHS
    )
    metrics = trainer.train()
    logger.info(metrics)

    return model, metrics


def test(test_dataset, model, writer):
    # 推論
    model.eval()
    with torch.no_grad():
        predicted = [model.forward_on_instance(d)['logits'].argmax() for d in tqdm(test_dataset)]

    # Accuracyの計算
    target = np.array([ins.fields['label'].label for ins in test_dataset])
    predict = np.array(predicted)
    accuracy = accuracy_score(target, predict)
    # Precision/Recallの計算
    macro_precision = precision_score(target, predict, average='macro')
    micro_precision = precision_score(target, predict, average='micro')
    macro_recall = recall_score(target, predict, average='macro')
    micro_recall = recall_score(target, predict, average='micro')
    # MLflowに記録
    writer.log_metric('accuracy', accuracy)
    writer.log_metric('macro-precision', macro_precision)
    writer.log_metric('micro-precision', micro_precision)
    writer.log_metric('macro-recall', macro_recall)
    writer.log_metric('micro-recall', micro_recall)
    model.cpu()
    writer.log_torch_model(model)


def preprocess(X, y, preprocessor=None):
    if preprocessor is None:
        preprocessor = Preprocessor()
        preprocessor\
            .stack(ct.text.UnicodeNormalizer())\
            .stack(ct.Tokenizer("ja"))\
            .fit(X['article'])

    processed = preprocessor.transform(X['article'])
    dataset = [text_to_instance([token.surface for token in document], int(label)) for document, label in zip(processed, y)]
    return dataset, preprocessor


@hydra.main(config_path='config.yaml')
def main(cfg: DictConfig):
    # https://medium.com/pytorch/hydra-a-fresh-look-at-configuration-for-machine-learning-projects-50583186b710
    cwd = hydra.utils.get_original_cwd()
    train_X, train_y = load_dataset(os.path.join(cwd, 'data'), 'train')
    val_X, val_y = load_dataset(os.path.join(cwd, 'data'), 'val')
    test_X, test_y = load_dataset(os.path.join(cwd, 'data'), 'test')

    train_dataset, preprocessor = preprocess(train_X, train_y)
    val_dataset, preprocessor = preprocess(val_X, val_y, preprocessor)
    test_dataset, preprocessor = preprocess(test_X, test_y, preprocessor)

    EXPERIMENT_NAME = 'livedoor-news-hydra-exp'
    writer = MlflowWriter(EXPERIMENT_NAME)
    writer.log_params_from_omegaconf_dict(cfg)

    model, metrics = train(train_dataset, val_dataset, cfg)
    test(test_dataset, model, writer)
    # Hydraの成果物をArtifactに保存
    writer.log_artifact(os.path.join(os.getcwd(), '.hydra/config.yaml'))
    writer.log_artifact(os.path.join(os.getcwd(), '.hydra/hydra.yaml'))
    writer.log_artifact(os.path.join(os.getcwd(), '.hydra/overrides.yaml'))
    writer.log_artifact(os.path.join(os.getcwd(), 'main.log'))
    writer.set_terminated()


if __name__ == '__main__':
    main()
