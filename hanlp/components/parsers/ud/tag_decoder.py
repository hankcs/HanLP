# This file is modified from udify, which is licensed under the MIT license:
# MIT License
#
# Copyright (c) 2019 Dan Kondratyuk
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Decodes sequences of tags, e.g., POS tags, given a list of contextualized word embeddings
"""

from typing import Dict

import numpy
import torch
import torch.nn.functional as F
from alnlp.metrics import CategoricalAccuracy
from torch.nn.modules.adaptive import AdaptiveLogSoftmaxWithLoss
from torch.nn.modules.linear import Linear

from hanlp.components.parsers.ud.lemma_edit import apply_lemma_rule
from hanlp.components.parsers.ud.udify_util import sequence_cross_entropy, sequence_cross_entropy_with_logits


class TagDecoder(torch.nn.Module):
    """A basic sequence tagger that decodes from inputs of word embeddings"""

    def __init__(self,
                 input_dim,
                 num_classes,
                 label_smoothing: float = 0.03,
                 adaptive: bool = False) -> None:
        super(TagDecoder, self).__init__()

        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        self.adaptive = adaptive

        if self.adaptive:
            adaptive_cutoffs = [round(self.num_classes / 15), 3 * round(self.num_classes / 15)]
            self.task_output = AdaptiveLogSoftmaxWithLoss(input_dim,
                                                          self.num_classes,
                                                          cutoffs=adaptive_cutoffs,
                                                          div_value=4.0)
        else:
            self.task_output = Linear(self.output_dim, self.num_classes)

    def forward(self,
                encoded_text: torch.FloatTensor,
                mask: torch.LongTensor,
                gold_tags: torch.LongTensor,
                ) -> Dict[str, torch.Tensor]:
        hidden = encoded_text

        batch_size, sequence_length, _ = hidden.size()
        output_dim = [batch_size, sequence_length, self.num_classes]

        loss_fn = self._adaptive_loss if self.adaptive else self._loss

        output_dict = loss_fn(hidden, mask, gold_tags, output_dim)

        return output_dict

    def _adaptive_loss(self, hidden, mask, gold_tags, output_dim):
        logits = hidden
        reshaped_log_probs = logits.reshape(-1, logits.size(2))

        class_probabilities = self.task_output.log_prob(reshaped_log_probs).view(output_dim)

        output_dict = {"logits": logits, "class_probabilities": class_probabilities}

        if gold_tags is not None:
            output_dict["loss"] = sequence_cross_entropy(class_probabilities,
                                                         gold_tags,
                                                         mask,
                                                         label_smoothing=self.label_smoothing)

        return output_dict

    def _loss(self, hidden, mask, gold_tags, output_dim):
        logits = self.task_output(hidden)
        reshaped_log_probs = logits.view(-1, self.num_classes)
        class_probabilities = F.softmax(reshaped_log_probs, dim=-1).view(output_dim)

        output_dict = {"logits": logits, "class_probabilities": class_probabilities}

        if gold_tags is not None:
            output_dict["loss"] = sequence_cross_entropy_with_logits(logits,
                                                                     gold_tags,
                                                                     mask,
                                                                     label_smoothing=self.label_smoothing)
        return output_dict

    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        all_words = output_dict["words"]

        all_predictions = output_dict["class_probabilities"][self.task].cpu().data.numpy()
        if all_predictions.ndim == 3:
            predictions_list = [all_predictions[i] for i in range(all_predictions.shape[0])]
        else:
            predictions_list = [all_predictions]
        all_tags = []
        for predictions, words in zip(predictions_list, all_words):
            argmax_indices = numpy.argmax(predictions, axis=-1)
            tags = [self.vocab.get_token_from_index(x, namespace=self.task)
                    for x in argmax_indices]

            if self.task == "lemmas":
                def decode_lemma(word, rule):
                    if rule == "_":
                        return "_"
                    if rule == "@@UNKNOWN@@":
                        return word
                    return apply_lemma_rule(word, rule)

                tags = [decode_lemma(word, rule) for word, rule in zip(words, tags)]

            all_tags.append(tags)
        output_dict[self.task] = all_tags

        return output_dict
