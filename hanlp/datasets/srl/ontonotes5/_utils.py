#!/usr/bin/env python
import codecs
import collections
import glob
import json
import os
import re
import sys
from pprint import pprint

from hanlp.datasets.parsing._ctb_utils import remove_all_ec, convert_to_stanford_dependency_330
from hanlp.utils.io_util import merge_files, get_resource, pushd, run_cmd, read_tsv_as_sents, replace_ext
from hanlp.utils.log_util import flash
from hanlp_common.io import eprint

BEGIN_DOCUMENT_REGEX = re.compile(r"#begin document \((.*)\); part (\d+)")


def flatten(l):
    return [item for sublist in l for item in sublist]


def get_doc_key(doc_id, part):
    return "{}_{}".format(doc_id, int(part))


class DocumentState(object):
    def __init__(self):
        self.doc_key = None
        self.text = []
        self.text_speakers = []
        self.speakers = []
        self.sentences = []
        self.pos = []
        self.lemma = []
        self.pos_buffer = []
        self.lemma_buffer = []
        self.constituents = []  # {}
        self.const_stack = []
        self.const_buffer = []
        self.ner = []
        self.ner_stack = []
        self.ner_buffer = []
        self.srl = []
        self.argument_stacks = []
        self.argument_buffers = []
        self.predicate_buffer = []
        self.clusters = collections.defaultdict(list)
        self.coref_stacks = collections.defaultdict(list)

    def assert_empty(self):
        assert self.doc_key is None
        assert len(self.text) == 0
        assert len(self.text_speakers) == 0
        assert len(self.speakers) == 0
        assert len(self.sentences) == 0
        assert len(self.srl) == 0
        assert len(self.predicate_buffer) == 0
        assert len(self.argument_buffers) == 0
        assert len(self.argument_stacks) == 0
        assert len(self.constituents) == 0
        assert len(self.const_stack) == 0
        assert len(self.const_buffer) == 0
        assert len(self.ner) == 0
        assert len(self.lemma_buffer) == 0
        assert len(self.pos_buffer) == 0
        assert len(self.ner_stack) == 0
        assert len(self.ner_buffer) == 0
        assert len(self.coref_stacks) == 0
        assert len(self.clusters) == 0

    def assert_finalizable(self):
        assert self.doc_key is not None
        assert len(self.text) == 0
        assert len(self.text_speakers) == 0
        assert len(self.speakers) > 0
        assert len(self.sentences) > 0
        assert len(self.constituents) > 0
        assert len(self.const_stack) == 0
        assert len(self.ner_stack) == 0
        assert len(self.predicate_buffer) == 0
        assert all(len(s) == 0 for s in list(self.coref_stacks.values()))

    def finalize_sentence(self):
        self.sentences.append(tuple(self.text))
        del self.text[:]
        self.lemma.append(tuple(self.lemma_buffer))
        del self.lemma_buffer[:]
        self.pos.append(tuple(self.pos_buffer))
        del self.pos_buffer[:]
        self.speakers.append(tuple(self.text_speakers))
        del self.text_speakers[:]

        assert len(self.predicate_buffer) == len(self.argument_buffers)
        self.srl.append([])
        for pred, args in zip(self.predicate_buffer, self.argument_buffers):
            for start, end, label in args:
                self.srl[-1].append((pred, start, end, label))
        self.predicate_buffer = []
        self.argument_buffers = []
        self.argument_stacks = []
        self.constituents.append([c for c in self.const_buffer])
        self.const_buffer = []
        self.ner.append([c for c in self.ner_buffer])
        self.ner_buffer = []

    def finalize(self):
        merged_clusters = []
        for c1 in list(self.clusters.values()):
            existing = None
            for m in c1:
                for c2 in merged_clusters:
                    if m in c2:
                        existing = c2
                        break
                if existing is not None:
                    break
            if existing is not None:
                print("Merging clusters (shouldn't happen very often.)")
                existing.update(c1)
            else:
                merged_clusters.append(set(c1))
        merged_clusters = [list(c) for c in merged_clusters]
        all_mentions = flatten(merged_clusters)
        assert len(all_mentions) == len(set(all_mentions))
        assert len(self.sentences) == len(self.srl)
        assert len(self.sentences) == len(self.constituents)
        assert len(self.sentences) == len(self.ner)
        return {
            "doc_key": self.doc_key,
            "sentences": self.sentences,
            "lemma": self.lemma,
            "pos": self.pos,
            "speakers": self.speakers,
            "srl": self.srl,
            "constituents": self.constituents,
            "ner": self.ner,
            "clusters": merged_clusters
        }


def filter_data(v5_input_file, doc_ids_file, output_file):
    """Filter OntoNotes5 data based on CoNLL2012 (coref) doc ids.
    https://github.com/bcmi220/unisrl/blob/master/scripts/filter_conll2012_data.py

    Args:
      v5_input_file: param doc_ids_file:
      output_file: 
      doc_ids_file: 

    Returns:

    """
    doc_count = 0
    sentence_count = 0
    srl_count = 0
    ner_count = 0
    cluster_count = 0
    word_count = 0
    doc_ids = []
    doc_ids_to_keys = {}
    filtered_examples = {}

    with open(doc_ids_file, "r") as f:
        for line in f:
            doc_id = line.strip().split("annotations/")[1]
            doc_ids.append(doc_id)
            doc_ids_to_keys[doc_id] = []
        f.close()

    with codecs.open(v5_input_file, "r", "utf8") as f:
        for jsonline in f:
            example = json.loads(jsonline)
            doc_key = example["doc_key"]
            dk_prefix = "_".join(doc_key.split("_")[:-1])
            if dk_prefix not in doc_ids_to_keys:
                continue
            doc_ids_to_keys[dk_prefix].append(doc_key)
            filtered_examples[doc_key] = example

            sentences = example["sentences"]
            word_count += sum([len(s) for s in sentences])
            sentence_count += len(sentences)
            srl_count += sum([len(srl) for srl in example["srl"]])
            ner_count += sum([len(ner) for ner in example["ner"]])
            coref = example["clusters"]
            cluster_count += len(coref)
            doc_count += 1
        f.close()

    print(("Documents: {}\nSentences: {}\nWords: {}\nNER: {}, PAS: {}, Clusters: {}".format(
        doc_count, sentence_count, word_count, ner_count, srl_count, cluster_count)))

    with codecs.open(output_file, "w", "utf8") as f:
        for doc_id in doc_ids:
            for key in doc_ids_to_keys[doc_id]:
                f.write(json.dumps(filtered_examples[key], ensure_ascii=False))
                f.write("\n")
        f.close()


def normalize_word(word, language):
    if language == "arabic":
        word = word[:word.find("#")]
    if word == "/." or word == "/?":
        return word[1:]
    else:
        return word


def handle_bit(word_index, bit, stack, spans, label_set):
    asterisk_idx = bit.find("*")
    if asterisk_idx >= 0:
        open_parens = bit[:asterisk_idx]
        close_parens = bit[asterisk_idx + 1:]
    else:
        open_parens = bit[:-1]
        close_parens = bit[-1]

    current_idx = open_parens.find("(")
    while current_idx >= 0:
        next_idx = open_parens.find("(", current_idx + 1)
        if next_idx >= 0:
            label = open_parens[current_idx + 1:next_idx]
        else:
            label = open_parens[current_idx + 1:]
        label_set.add(label)
        stack.append((word_index, label))
        current_idx = next_idx

    for c in close_parens:
        try:
            assert c == ")"
        except AssertionError:
            print(word_index, bit, spans, stack)
            continue
        open_index, label = stack.pop()
        spans.append((open_index, word_index, label))
        ''' current_span = (open_index, word_index)
        if current_span in spans:
          spans[current_span] += "_" + label
        else:
          spans[current_span] = label
        spans[current_span] = label '''


def handle_line(line, document_state: DocumentState, language, labels, stats):
    begin_document_match = re.match(BEGIN_DOCUMENT_REGEX, line)
    if begin_document_match:
        document_state.assert_empty()
        document_state.doc_key = get_doc_key(begin_document_match.group(1), begin_document_match.group(2))
        return None
    elif line.startswith("#end document"):
        document_state.assert_finalizable()
        finalized_state = document_state.finalize()
        stats["num_clusters"] += len(finalized_state["clusters"])
        stats["num_mentions"] += sum(len(c) for c in finalized_state["clusters"])
        # labels["{}_const_labels".format(language)].update(l for _, _, l in finalized_state["constituents"])
        # labels["ner"].update(l for _, _, l in finalized_state["ner"])
        return finalized_state
    else:
        row = line.split()
        # Starting a new sentence.
        if len(row) == 0:
            stats["max_sent_len_{}".format(language)] = max(len(document_state.text),
                                                            stats["max_sent_len_{}".format(language)])
            stats["num_sents_{}".format(language)] += 1
            document_state.finalize_sentence()
            return None
        assert len(row) >= 12

        doc_key = get_doc_key(row[0], row[1])
        word = normalize_word(row[3], language)
        pos = row[4]
        parse = row[5]
        lemma = row[6]
        predicate_sense = row[7]
        speaker = row[9]
        ner = row[10]
        args = row[11:-1]
        coref = row[-1]

        word_index = len(document_state.text) + sum(len(s) for s in document_state.sentences)
        document_state.text.append(word)
        document_state.text_speakers.append(speaker)
        document_state.pos_buffer.append(pos)
        document_state.lemma_buffer.append(lemma)

        handle_bit(word_index, parse, document_state.const_stack, document_state.const_buffer, labels["categories"])
        handle_bit(word_index, ner, document_state.ner_stack, document_state.ner_buffer, labels["ner"])

        if len(document_state.argument_stacks) < len(args):
            document_state.argument_stacks = [[] for _ in args]
            document_state.argument_buffers = [[] for _ in args]

        for i, arg in enumerate(args):
            handle_bit(word_index, arg, document_state.argument_stacks[i], document_state.argument_buffers[i],
                       labels["srl"])
        if predicate_sense != "-":
            document_state.predicate_buffer.append(word_index)
        if coref != "-":
            for segment in coref.split("|"):
                if segment[0] == "(":
                    if segment[-1] == ")":
                        cluster_id = int(segment[1:-1])
                        document_state.clusters[cluster_id].append((word_index, word_index))
                    else:
                        cluster_id = int(segment[1:])
                        document_state.coref_stacks[cluster_id].append(word_index)
                else:
                    cluster_id = int(segment[:-1])
                    start = document_state.coref_stacks[cluster_id].pop()
                    document_state.clusters[cluster_id].append((start, word_index))
        return None


def ontonotes_document_generator(input_path, language, labels, stats):
    with open(input_path, "r") as input_file:
        document_state = DocumentState()
        for line in input_file.readlines():
            document = handle_line(line, document_state, language, labels, stats)
            if document is not None:
                yield document
                document_state = DocumentState()


def convert_to_jsonlines(input_path, output_path, language, labels=None, stats=None):
    if labels is None:
        labels = collections.defaultdict(set)
    if stats is None:
        stats = collections.defaultdict(int)
    count = 0
    with open(output_path, "w") as output_file:
        for document in ontonotes_document_generator(input_path, language, labels, stats):
            output_file.write(json.dumps(document, ensure_ascii=False))
            output_file.write("\n")
            count += 1

    return labels, stats


def make_ontonotes_jsonlines(conll12_ontonotes_path, output_path, languages=None):
    if languages is None:
        languages = ['english', 'chinese', 'arabic']
    for language in languages:
        make_ontonotes_language_jsonlines(conll12_ontonotes_path, output_path, language)


def make_ontonotes_language_jsonlines(conll12_ontonotes_path, output_path=None, language='english'):
    conll12_ontonotes_path = get_resource(conll12_ontonotes_path)
    if output_path is None:
        output_path = os.path.dirname(conll12_ontonotes_path)
    for split in ['train', 'development', 'test']:
        pattern = f'{conll12_ontonotes_path}/data/{split}/data/{language}/annotations/*/*/*/*gold_conll'
        files = sorted(glob.glob(pattern, recursive=True))
        assert files, f'No gold_conll files found in {pattern}'
        version = os.path.basename(files[0]).split('.')[-1].split('_')[0]
        if version.startswith('v'):
            assert all([version in os.path.basename(f) for f in files])
        else:
            version = 'v5'
        lang_dir = f'{output_path}/{language}'
        if split == 'conll-2012-test':
            split = 'test'
        full_file = f'{lang_dir}/{split}.{language}.{version}_gold_conll'
        os.makedirs(lang_dir, exist_ok=True)
        print(f'Merging {len(files)} files to {full_file}')
        merge_files(files, full_file)
        v5_json_file = full_file.replace(f'.{version}_gold_conll', f'.{version}.jsonlines')
        print(f'Converting CoNLL file {full_file} to json file {v5_json_file}')
        labels, stats = convert_to_jsonlines(full_file, v5_json_file, language)
        print('Labels:')
        pprint(labels)
        print('Statistics:')
        pprint(stats)
        conll12_json_file = f'{lang_dir}/{split}.{language}.conll12.jsonlines'
        print(f'Applying CoNLL 12 official splits on {v5_json_file} to {conll12_json_file}')
        id_file = get_resource(f'http://conll.cemantix.org/2012/download/ids/'
                               f'{language}/coref/{split}.id')
        filter_data(v5_json_file, id_file, conll12_json_file)


def make_gold_conll(ontonotes_path, language):
    ontonotes_path = os.path.abspath(get_resource(ontonotes_path))
    to_conll = get_resource(
        'https://gist.githubusercontent.com/hankcs/46b9137016c769e4b6137104daf43a92/raw/66369de6c24b5ec47696ae307591f0d72c6f3f02/ontonotes_to_conll.sh')
    to_conll = os.path.abspath(to_conll)
    # shutil.rmtree(os.path.join(ontonotes_path, 'conll-2012'), ignore_errors=True)
    with pushd(ontonotes_path):
        try:
            flash(f'Converting [blue]{language}[/blue] to CoNLL format, '
                  f'this might take half an hour [blink][yellow]...[/yellow][/blink]')
            run_cmd(f'bash {to_conll} {ontonotes_path} {language}')
            flash('')
        except RuntimeError as e:
            flash(f'[red]Failed[/red] to convert {language} of {ontonotes_path} to CoNLL. See exceptions for detail')
            raise e


def convert_jsonlines_to_IOBES(json_file, output_file=None, doc_level_offset=True):
    json_file = get_resource(json_file)
    if not output_file:
        output_file = os.path.splitext(json_file)[0] + '.ner.tsv'
    with open(json_file) as src, open(output_file, 'w', encoding='utf-8') as out:
        for line in src:
            doc = json.loads(line)
            offset = 0
            for sent, ner in zip(doc['sentences'], doc['ner']):
                tags = ['O'] * len(sent)
                for start, end, label in ner:
                    if doc_level_offset:
                        start -= offset
                        end -= offset
                    if start == end:
                        tags[start] = 'S-' + label
                    else:
                        tags[start] = 'B-' + label
                        for i in range(start + 1, end + 1):
                            tags[i] = 'I-' + label
                        tags[end] = 'E-' + label
                offset += len(sent)
                for token, tag in zip(sent, tags):
                    out.write(f'{token}\t{tag}\n')
                out.write('\n')


def make_ner_tsv_if_necessary(json_file):
    json_file = get_resource(json_file)
    output_file = os.path.splitext(json_file)[0] + '.ner.tsv'
    if not os.path.isfile(output_file):
        convert_jsonlines_to_IOBES(json_file, output_file)
    return output_file


def batch_make_ner_tsv_if_necessary(json_files):
    for each in json_files:
        make_ner_tsv_if_necessary(each)


def make_pos_tsv_if_necessary(json_file):
    json_file = get_resource(json_file)
    output_file = os.path.splitext(json_file)[0] + '.pos.tsv'
    if not os.path.isfile(output_file):
        make_pos_tsv(json_file, output_file)
    return output_file


def make_pos_tsv(json_file, output_file):
    with open(json_file) as src, open(output_file, 'w', encoding='utf-8') as out:
        for line in src:
            doc = json.loads(line)
            for sent, pos in zip(doc['sentences'], doc['pos']):
                for token, tag in zip(sent, pos):
                    out.write(f'{token}\t{tag}\n')
                out.write('\n')


def batch_make_pos_tsv_if_necessary(json_files):
    for each in json_files:
        make_pos_tsv_if_necessary(each)


def make_con_txt(conll_file, output_file):
    with open(output_file, 'w') as out:
        for sent in read_tsv_as_sents(conll_file):
            tree = []
            pos_per_sent = []
            for cell in sent:
                if cell[0] == '#begin' or cell[0] == '#end':
                    continue
                if len(cell) < 8:
                    print(cell)
                filename, sentence_id, token_id, word, POS, parse, framefile, roleset, *_ = cell
                parse = parse.replace('*', f'({POS} {word})')
                tree.append(parse)
                pos_per_sent.append(POS)
            bracketed = ' '.join(tree)
            out.write(bracketed)
            out.write('\n')


def make_con_txt_if_necessary(json_file):
    json_file = get_resource(json_file)
    output_file = os.path.splitext(json_file)[0] + '.con.txt'
    if not os.path.isfile(output_file):
        make_con_txt(json_file, output_file)
    return output_file


def batch_make_con_txt_if_necessary(json_files):
    for each in json_files:
        make_con_txt_if_necessary(each)


def batch_remove_empty_category_if_necessary(json_files):
    for each in json_files:
        src = get_resource(each)
        dst = replace_ext(src, '.noempty.txt')
        if not os.path.isfile(dst):
            remove_all_ec(src)


def make_dep_conllx(con_txt_file, output_file, language='en'):
    con_txt_file = get_resource(con_txt_file)
    convert_to_stanford_dependency_330(con_txt_file, output_file, language=language)


def make_dep_conllx_if_necessary(con_txt_file: str, language='en'):
    con_txt_file = get_resource(con_txt_file)
    output_file = con_txt_file.replace('.con.txt', '.dep.conllx', 1)
    if os.path.isfile(output_file):
        return
    make_dep_conllx(con_txt_file, output_file, language)


def batch_make_dep_conllx_if_necessary(con_txt_files, language='en'):
    for each in con_txt_files:
        make_dep_conllx_if_necessary(each, language)


def main():
    if len(sys.argv) != 3:
        eprint('2 arguments required: ontonotes_path output_path')
        exit(1)
    ontonotes_path = sys.argv[1]
    output_path = sys.argv[2]
    make_ontonotes_jsonlines(ontonotes_path, output_path)


if __name__ == "__main__":
    main()
