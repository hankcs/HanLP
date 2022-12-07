# coding:utf-8
# MIT License
#
# Copyright (c) 2022 xfbai
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

class AMRParsingDataSet(object):

    @staticmethod
    def tokenize(sample: dict, tokenizer, max_src_length=400, max_tgt_length=1024, unified_input=True, amr="src",
                 text="tgt"):
        amr = sample.get(amr, None)  # AMR tokens
        txt = sample[text]  # Text tokens

        if amr is not None:
            sample['labels'] = tokenizer.tokenize_amr(amr.split())[:max_src_length - 2] + [tokenizer.amr_eos_token_id]

        raw_txt_ids = tokenizer(
            txt, max_length=max_tgt_length, padding=False, truncation=True
        )["input_ids"]
        if unified_input:
            txt_ids = raw_txt_ids[:max_tgt_length - 3] + [tokenizer.amr_bos_token_id, tokenizer.mask_token_id,
                                                          tokenizer.amr_eos_token_id]
        else:
            txt_ids = raw_txt_ids
        sample['input_ids'] = txt_ids
        return sample


class AMR2TextDataSet(object):

    @staticmethod
    def tokenize(sample: dict, tokenizer, max_src_length=400, max_tgt_length=1024, unified_input=True, amr="src",
                 text="tgt"):
        src = sample[amr]  # AMR tokens
        tgt = sample.get(text, None)  # Text tokens
        if not unified_input:
            src_ids = [tokenizer.amr_bos_token_id] + tokenizer.tokenize_amr(src.split())[
                                                     :max_src_length - 2] + [tokenizer.amr_eos_token_id]

        else:
            # [<s>[mask]</s><AMR>xxx</AMR>]
            src_ids = [tokenizer.bos_token_id, tokenizer.mask_token_id, tokenizer.eos_token_id] + [
                tokenizer.amr_bos_token_id] + tokenizer.tokenize_amr(src.split())[:max_src_length - 5] + [
                          tokenizer.amr_eos_token_id]
        sample["input_ids"] = src_ids

        if tgt is not None:
            with tokenizer.as_target_tokenizer():
                tgt_ids = tokenizer(
                    tgt, max_length=max_tgt_length, padding=False, truncation=True
                )
                tgt_ids["input_ids"] = [
                    label[1:] for label in tgt_ids["input_ids"]
                ]
            sample["labels"] = tgt_ids["input_ids"]
        return sample
