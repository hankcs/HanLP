# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-01-22 19:31
from bert.tokenization.albert_tokenization import *


class FullTokenizer(object):
    """
    TODO Remove this file once https://github.com/kpe/bert-for-tf2/issues/47 is fixed
    Runs end-to-end tokenziation.
    """

    def __init__(self, vocab_file, do_lower_case=True, spm_model_file=None):
        self.vocab = None
        self.sp_model = None
        if spm_model_file:
            self.sp_model = spm.SentencePieceProcessor()
            self.sp_model.Load(spm_model_file)
            # Note(mingdachen): For the purpose of consisent API, we are
            # generating a vocabulary for the sentence piece tokenizer.
            self.vocab = {self.sp_model.IdToPiece(i): i for i
                          in range(self.sp_model.GetPieceSize())}
        else:
            self.vocab = load_vocab(vocab_file)
            self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
            self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def tokenize(self, text):
        if self.sp_model:
            split_tokens = encode_pieces(self.sp_model, text, return_unicode=False)
        else:
            split_tokens = []
            for token in self.basic_tokenizer.tokenize(text):
                for sub_token in self.wordpiece_tokenizer.tokenize(token):
                    split_tokens.append(sub_token)

        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        if self.sp_model:
            return [self.sp_model.PieceToId(
                printable_text(token)) for token in tokens]
        else:
            return convert_by_vocab(self.vocab, tokens)

    def convert_ids_to_tokens(self, ids):
        if self.sp_model:
            return [self.sp_model.IdToPiece(id_) for id_ in ids]
        else:
            return convert_by_vocab(self.inv_vocab, ids)
