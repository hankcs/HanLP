from typing import DefaultDict, List, Optional, Iterator, Set, Tuple, Dict
from collections import defaultdict
import codecs
import os
import logging

from hanlp.utils.span_util import TypedSpan, enumerate_spans
from phrasetree.tree import Tree

logger = logging.getLogger(__name__)


class OntonotesSentence:
    """
    A class representing the annotations available for a single CONLL formatted sentence.

    # Parameters

    document_id : `str`
        This is a variation on the document filename
    sentence_id : `int`
        The integer ID of the sentence within a document.
    words : `List[str]`
        This is the tokens as segmented/tokenized in the Treebank.
    pos_tags : `List[str]`
        This is the Penn-Treebank-style part of speech. When parse information is missing,
        all parts of speech except the one for which there is some sense or proposition
        annotation are marked with a XX tag. The verb is marked with just a VERB tag.
    parse_tree : `nltk.Tree`
        An nltk Tree representing the parse. It includes POS tags as pre-terminal nodes.
        When the parse information is missing, the parse will be `None`.
    predicate_lemmas : `List[Optional[str]]`
        The predicate lemma of the words for which we have semantic role
        information or word sense information. All other indices are `None`.
    predicate_framenet_ids : `List[Optional[int]]`
        The PropBank frameset ID of the lemmas in `predicate_lemmas`, or `None`.
    word_senses : `List[Optional[float]]`
        The word senses for the words in the sentence, or `None`. These are floats
        because the word sense can have values after the decimal, like `1.1`.
    speakers : `List[Optional[str]]`
        The speaker information for the words in the sentence, if present, or `None`
        This is the speaker or author name where available. Mostly in Broadcast Conversation
        and Web Log data. When not available the rows are marked with an "-".
    named_entities : `List[str]`
        The BIO tags for named entities in the sentence.
    srl_frames : `List[Tuple[str, List[str]]]`
        A dictionary keyed by the verb in the sentence for the given
        Propbank frame labels, in a BIO format.
    coref_spans : `Set[TypedSpan]`
        The spans for entity mentions involved in coreference resolution within the sentence.
        Each element is a tuple composed of (cluster_id, (start_index, end_index)). Indices
        are `inclusive`.
    """

    def __init__(
        self,
        document_id: str,
        sentence_id: int,
        words: List[str],
        pos_tags: List[str],
        parse_tree: Optional[Tree],
        predicate_lemmas: List[Optional[str]],
        predicate_framenet_ids: List[Optional[str]],
        word_senses: List[Optional[float]],
        speakers: List[Optional[str]],
        named_entities: List[str],
        srl_frames: List[Tuple[str, List[str]]],
        coref_spans: Set[TypedSpan],
    ) -> None:

        self.document_id = document_id
        self.sentence_id = sentence_id
        self.words = words
        self.pos_tags = pos_tags
        self.parse_tree = parse_tree
        self.predicate_lemmas = predicate_lemmas
        self.predicate_framenet_ids = predicate_framenet_ids
        self.word_senses = word_senses
        self.speakers = speakers
        self.named_entities = named_entities
        self.srl_frames = srl_frames
        self.coref_spans = coref_spans


class Ontonotes:
    """
    This `DatasetReader` is designed to read in the English OntoNotes v5.0 data
    in the format used by the CoNLL 2011/2012 shared tasks. In order to use this
    Reader, you must follow the instructions provided [here (v12 release):]
    (https://cemantix.org/data/ontonotes.html), which will allow you to download
    the CoNLL style annotations for the  OntoNotes v5.0 release -- LDC2013T19.tgz
    obtained from LDC.

    Once you have run the scripts on the extracted data, you will have a folder
    structured as follows:

    ```
    conll-formatted-ontonotes-5.0/
     ── data
       ├── development
           └── data
               └── english
                   └── annotations
                       ├── bc
                       ├── bn
                       ├── mz
                       ├── nw
                       ├── pt
                       ├── tc
                       └── wb
       ├── test
           └── data
               └── english
                   └── annotations
                       ├── bc
                       ├── bn
                       ├── mz
                       ├── nw
                       ├── pt
                       ├── tc
                       └── wb
       └── train
           └── data
               └── english
                   └── annotations
                       ├── bc
                       ├── bn
                       ├── mz
                       ├── nw
                       ├── pt
                       ├── tc
                       └── wb
    ```

    The file path provided to this class can then be any of the train, test or development
    directories(or the top level data directory, if you are not utilizing the splits).

    The data has the following format, ordered by column.

    1.  Document ID : `str`
        This is a variation on the document filename
    2.  Part number : `int`
        Some files are divided into multiple parts numbered as 000, 001, 002, ... etc.
    3.  Word number : `int`
        This is the word index of the word in that sentence.
    4.  Word : `str`
        This is the token as segmented/tokenized in the Treebank. Initially the `*_skel` file
        contain the placeholder [WORD] which gets replaced by the actual token from the
        Treebank which is part of the OntoNotes release.
    5.  POS Tag : `str`
        This is the Penn Treebank style part of speech. When parse information is missing,
        all part of speeches except the one for which there is some sense or proposition
        annotation are marked with a XX tag. The verb is marked with just a VERB tag.
    6.  Parse bit : `str`
        This is the bracketed structure broken before the first open parenthesis in the parse,
        and the word/part-of-speech leaf replaced with a `*`. When the parse information is
        missing, the first word of a sentence is tagged as `(TOP*` and the last word is tagged
        as `*)` and all intermediate words are tagged with a `*`.
    7.  Predicate lemma : `str`
        The predicate lemma is mentioned for the rows for which we have semantic role
        information or word sense information. All other rows are marked with a "-".
    8.  Predicate Frameset ID : `int`
        The PropBank frameset ID of the predicate in Column 7.
    9.  Word sense : `float`
        This is the word sense of the word in Column 3.
    10. Speaker/Author : `str`
        This is the speaker or author name where available. Mostly in Broadcast Conversation
        and Web Log data. When not available the rows are marked with an "-".
    11. Named Entities : `str`
        These columns identifies the spans representing various named entities. For documents
        which do not have named entity annotation, each line is represented with an `*`.
    12. Predicate Arguments : `str`
        There is one column each of predicate argument structure information for the predicate
        mentioned in Column 7. If there are no predicates tagged in a sentence this is a
        single column with all rows marked with an `*`.
    -1. Co-reference : `str`
        Co-reference chain information encoded in a parenthesis structure. For documents that do
         not have co-reference annotations, each line is represented with a "-".
    """

    def dataset_iterator(self, file_path: str) -> Iterator[OntonotesSentence]:
        """
        An iterator over the entire dataset, yielding all sentences processed.
        """
        for conll_file in self.dataset_path_iterator(file_path):
            yield from self.sentence_iterator(conll_file)

    @staticmethod
    def dataset_path_iterator(file_path: str) -> Iterator[str]:
        """
        An iterator returning file_paths in a directory
        containing CONLL-formatted files.
        """
        logger.info("Reading CONLL sentences from dataset files at: %s", file_path)
        for root, _, files in list(os.walk(file_path)):
            for data_file in files:
                # These are a relic of the dataset pre-processing. Every
                # file will be duplicated - one file called filename.gold_skel
                # and one generated from the preprocessing called filename.gold_conll.
                if not data_file.endswith("gold_conll"):
                    continue

                yield os.path.join(root, data_file)

    def dataset_document_iterator(self, file_path: str) -> Iterator[List[OntonotesSentence]]:
        """
        An iterator over CONLL formatted files which yields documents, regardless
        of the number of document annotations in a particular file. This is useful
        for conll data which has been preprocessed, such as the preprocessing which
        takes place for the 2012 CONLL Coreference Resolution task.
        """
        with codecs.open(file_path, "r", encoding="utf8") as open_file:
            conll_rows = []
            document: List[OntonotesSentence] = []
            for line in open_file:
                line = line.strip()
                if line != "" and not line.startswith("#"):
                    # Non-empty line. Collect the annotation.
                    conll_rows.append(line)
                else:
                    if conll_rows:
                        document.append(self._conll_rows_to_sentence(conll_rows))
                        conll_rows = []
                if line.startswith("#end document"):
                    yield document
                    document = []
            if document:
                # Collect any stragglers or files which might not
                # have the '#end document' format for the end of the file.
                yield document

    def sentence_iterator(self, file_path: str) -> Iterator[OntonotesSentence]:
        """
        An iterator over the sentences in an individual CONLL formatted file.
        """
        for document in self.dataset_document_iterator(file_path):
            for sentence in document:
                yield sentence

    def _conll_rows_to_sentence(self, conll_rows: List[str]) -> OntonotesSentence:
        document_id: str = None
        sentence_id: int = None
        # The words in the sentence.
        sentence: List[str] = []
        # The pos tags of the words in the sentence.
        pos_tags: List[str] = []
        # the pieces of the parse tree.
        parse_pieces: List[str] = []
        # The lemmatised form of the words in the sentence which
        # have SRL or word sense information.
        predicate_lemmas: List[str] = []
        # The FrameNet ID of the predicate.
        predicate_framenet_ids: List[str] = []
        # The sense of the word, if available.
        word_senses: List[float] = []
        # The current speaker, if available.
        speakers: List[str] = []

        verbal_predicates: List[str] = []
        span_labels: List[List[str]] = []
        current_span_labels: List[str] = []

        # Cluster id -> List of (start_index, end_index) spans.
        clusters: DefaultDict[int, List[Tuple[int, int]]] = defaultdict(list)
        # Cluster id -> List of start_indices which are open for this id.
        coref_stacks: DefaultDict[int, List[int]] = defaultdict(list)

        for index, row in enumerate(conll_rows):
            conll_components = row.split()

            document_id = conll_components[0]
            sentence_id = int(conll_components[1])
            word = conll_components[3]
            pos_tag = conll_components[4]
            parse_piece = conll_components[5]

            # Replace brackets in text and pos tags
            # with a different token for parse trees.
            if pos_tag != "XX" and word != "XX":
                if word == "(":
                    parse_word = "-LRB-"
                elif word == ")":
                    parse_word = "-RRB-"
                else:
                    parse_word = word
                if pos_tag == "(":
                    pos_tag = "-LRB-"
                if pos_tag == ")":
                    pos_tag = "-RRB-"
                (left_brackets, right_hand_side) = parse_piece.split("*")
                # only keep ')' if there are nested brackets with nothing in them.
                right_brackets = right_hand_side.count(")") * ")"
                parse_piece = f"{left_brackets} ({pos_tag} {parse_word}) {right_brackets}"
            else:
                # There are some bad annotations in the CONLL data.
                # They contain no information, so to make this explicit,
                # we just set the parse piece to be None which will result
                # in the overall parse tree being None.
                parse_piece = None

            lemmatised_word = conll_components[6]
            framenet_id = conll_components[7]
            word_sense = conll_components[8]
            speaker = conll_components[9]

            if not span_labels:
                # If this is the first word in the sentence, create
                # empty lists to collect the NER and SRL BIO labels.
                # We can't do this upfront, because we don't know how many
                # components we are collecting, as a sentence can have
                # variable numbers of SRL frames.
                span_labels = [[] for _ in conll_components[10:-1]]
                # Create variables representing the current label for each label
                # sequence we are collecting.
                current_span_labels = [None for _ in conll_components[10:-1]]

            self._process_span_annotations_for_word(
                conll_components[10:-1], span_labels, current_span_labels
            )

            # If any annotation marks this word as a verb predicate,
            # we need to record its index. This also has the side effect
            # of ordering the verbal predicates by their location in the
            # sentence, automatically aligning them with the annotations.
            word_is_verbal_predicate = any("(V" in x for x in conll_components[11:-1])
            if word_is_verbal_predicate:
                verbal_predicates.append(word)

            self._process_coref_span_annotations_for_word(
                conll_components[-1], index, clusters, coref_stacks
            )

            sentence.append(word)
            pos_tags.append(pos_tag)
            parse_pieces.append(parse_piece)
            predicate_lemmas.append(lemmatised_word if lemmatised_word != "-" else None)
            predicate_framenet_ids.append(framenet_id if framenet_id != "-" else None)
            word_senses.append(float(word_sense) if word_sense != "-" else None)
            speakers.append(speaker if speaker != "-" else None)

        named_entities = span_labels[0]
        srl_frames = [
            (predicate, labels) for predicate, labels in zip(verbal_predicates, span_labels[1:])
        ]

        if all(parse_pieces):
            parse_tree = Tree.fromstring("".join(parse_pieces))
        else:
            parse_tree = None
        coref_span_tuples: Set[TypedSpan] = {
            (cluster_id, span) for cluster_id, span_list in clusters.items() for span in span_list
        }
        return OntonotesSentence(
            document_id,
            sentence_id,
            sentence,
            pos_tags,
            parse_tree,
            predicate_lemmas,
            predicate_framenet_ids,
            word_senses,
            speakers,
            named_entities,
            srl_frames,
            coref_span_tuples,
        )

    @staticmethod
    def _process_coref_span_annotations_for_word(
        label: str,
        word_index: int,
        clusters: DefaultDict[int, List[Tuple[int, int]]],
        coref_stacks: DefaultDict[int, List[int]],
    ) -> None:
        """
        For a given coref label, add it to a currently open span(s), complete a span(s) or
        ignore it, if it is outside of all spans. This method mutates the clusters and coref_stacks
        dictionaries.

        # Parameters

        label : `str`
            The coref label for this word.
        word_index : `int`
            The word index into the sentence.
        clusters : `DefaultDict[int, List[Tuple[int, int]]]`
            A dictionary mapping cluster ids to lists of inclusive spans into the
            sentence.
        coref_stacks : `DefaultDict[int, List[int]]`
            Stacks for each cluster id to hold the start indices of active spans (spans
            which we are inside of when processing a given word). Spans with the same id
            can be nested, which is why we collect these opening spans on a stack, e.g:

            [Greg, the baker who referred to [himself]_ID1 as 'the bread man']_ID1
        """
        if label != "-":
            for segment in label.split("|"):
                # The conll representation of coref spans allows spans to
                # overlap. If spans end or begin at the same word, they are
                # separated by a "|".
                if segment[0] == "(":
                    # The span begins at this word.
                    if segment[-1] == ")":
                        # The span begins and ends at this word (single word span).
                        cluster_id = int(segment[1:-1])
                        clusters[cluster_id].append((word_index, word_index))
                    else:
                        # The span is starting, so we record the index of the word.
                        cluster_id = int(segment[1:])
                        coref_stacks[cluster_id].append(word_index)
                else:
                    # The span for this id is ending, but didn't start at this word.
                    # Retrieve the start index from the document state and
                    # add the span to the clusters for this id.
                    cluster_id = int(segment[:-1])
                    start = coref_stacks[cluster_id].pop()
                    clusters[cluster_id].append((start, word_index))

    @staticmethod
    def _process_span_annotations_for_word(
        annotations: List[str],
        span_labels: List[List[str]],
        current_span_labels: List[Optional[str]],
    ) -> None:
        """
        Given a sequence of different label types for a single word and the current
        span label we are inside, compute the BIO tag for each label and append to a list.

        # Parameters

        annotations : `List[str]`
            A list of labels to compute BIO tags for.
        span_labels : `List[List[str]]`
            A list of lists, one for each annotation, to incrementally collect
            the BIO tags for a sequence.
        current_span_labels : `List[Optional[str]]`
            The currently open span per annotation type, or `None` if there is no open span.
        """
        for annotation_index, annotation in enumerate(annotations):
            # strip all bracketing information to
            # get the actual propbank label.
            label = annotation.strip("()*")

            if "(" in annotation:
                # Entering into a span for a particular semantic role label.
                # We append the label and set the current span for this annotation.
                bio_label = "B-" + label
                span_labels[annotation_index].append(bio_label)
                current_span_labels[annotation_index] = label
            elif current_span_labels[annotation_index] is not None:
                # If there's no '(' token, but the current_span_label is not None,
                # then we are inside a span.
                bio_label = "I-" + current_span_labels[annotation_index]
                span_labels[annotation_index].append(bio_label)
            else:
                # We're outside a span.
                span_labels[annotation_index].append("O")
            # Exiting a span, so we reset the current span label for this annotation.
            if ")" in annotation:
                current_span_labels[annotation_index] = None


def make_coref_instance(
        sentences: List[List[str]],
        max_span_width: int,
        gold_clusters: Optional[List[List[Tuple[int, int]]]] = None,
        max_sentences: int = None,
        remove_singleton_clusters: bool = True,
) -> dict:
    """
    # Parameters

    sentences : `List[List[str]]`, required.
        A list of lists representing the tokenised words and sentences in the document.
    token_indexers : `Dict[str, TokenIndexer]`
        This is used to index the words in the document.  See :class:`TokenIndexer`.
    max_span_width : `int`, required.
        The maximum width of candidate spans to consider.
    gold_clusters : `Optional[List[List[Tuple[int, int]]]]`, optional (default = None)
        A list of all clusters in the document, represented as word spans with absolute indices
        in the entire document. Each cluster contains some number of spans, which can be nested
        and overlap. If there are exact matches between clusters, they will be resolved
        using `_canonicalize_clusters`.
    wordpiece_modeling_tokenizer: `PretrainedTransformerTokenizer`, optional (default = None)
        If not None, this dataset reader does subword tokenization using the supplied tokenizer
        and distribute the labels to the resulting wordpieces. All the modeling will be based on
        wordpieces. If this is set to `False` (default), the user is expected to use
        `PretrainedTransformerMismatchedIndexer` and `PretrainedTransformerMismatchedEmbedder`,
        and the modeling will be on the word-level.
    max_sentences: int, optional (default = None)
        The maximum number of sentences in each document to keep. By default keeps all sentences.
    remove_singleton_clusters : `bool`, optional (default = True)
        Some datasets contain clusters that are singletons (i.e. no coreferents). This option allows
        the removal of them.

    # Returns

    An `Instance` containing the following `Fields`:
        text : `TextField`
            The text of the full document.
        spans : `ListField[SpanField]`
            A ListField containing the spans represented as `SpanFields`
            with respect to the document text.
        span_labels : `SequenceLabelField`, optional
            The id of the cluster which each possible span belongs to, or -1 if it does
                not belong to a cluster. As these labels have variable length (it depends on
                how many spans we are considering), we represent this a as a `SequenceLabelField`
                with respect to the spans `ListField`.
    """
    if max_sentences is not None and len(sentences) > max_sentences:
        sentences = sentences[:max_sentences]
        total_length = sum(len(sentence) for sentence in sentences)

        if gold_clusters is not None:
            new_gold_clusters = []

            for cluster in gold_clusters:
                new_cluster = []
                for mention in cluster:
                    if mention[1] < total_length:
                        new_cluster.append(mention)
                if new_cluster:
                    new_gold_clusters.append(new_cluster)

            gold_clusters = new_gold_clusters

    flattened_sentences = [_normalize_word(word) for sentence in sentences for word in sentence]
    flat_sentences_tokens = [word for word in flattened_sentences]

    text_field = flat_sentences_tokens

    cluster_dict = {}
    if gold_clusters is not None:
        gold_clusters = _canonicalize_clusters(gold_clusters)
        if remove_singleton_clusters:
            gold_clusters = [cluster for cluster in gold_clusters if len(cluster) > 1]

        for cluster_id, cluster in enumerate(gold_clusters):
            for mention in cluster:
                cluster_dict[tuple(mention)] = cluster_id

    spans: List = []
    span_labels: Optional[List[int]] = [] if gold_clusters is not None else None

    sentence_offset = 0
    for sentence in sentences:
        for start, end in enumerate_spans(
                sentence, offset=sentence_offset, max_span_width=max_span_width
        ):

            if span_labels is not None:
                if (start, end) in cluster_dict:
                    span_labels.append(cluster_dict[(start, end)])
                else:
                    span_labels.append(-1)

            spans.append((start, end))
        sentence_offset += len(sentence)

    span_field = spans

    # metadata: Dict[str, Any] = {"original_text": flattened_sentences}
    # if gold_clusters is not None:
    #     metadata["clusters"] = gold_clusters
    # metadata_field = MetadataField(metadata)

    fields: Dict[str, List] = {
        "text": text_field,
        "spans": span_field,
        'clusters': gold_clusters,
        # "metadata": metadata_field,
    }
    if span_labels is not None:
        fields["span_labels"] = span_labels

    return fields


def _normalize_word(word):
    if word in ("/.", "/?"):
        return word[1:]
    else:
        return word


def _canonicalize_clusters(clusters: List[List[Tuple[int, int]]]) -> List[List[Tuple[int, int]]]:
    """
    The data might include 2 annotated spans which are identical,
    but have different ids. This checks all clusters for spans which are
    identical, and if it finds any, merges the clusters containing the
    identical spans.
    """
    merged_clusters: List[Set[Tuple[int, int]]] = []
    for cluster in clusters:
        cluster_with_overlapping_mention = None
        for mention in cluster:
            # Look at clusters we have already processed to
            # see if they contain a mention in the current
            # cluster for comparison.
            for cluster2 in merged_clusters:
                if mention in cluster2:
                    # first cluster in merged clusters
                    # which contains this mention.
                    cluster_with_overlapping_mention = cluster2
                    break
            # Already encountered overlap - no need to keep looking.
            if cluster_with_overlapping_mention is not None:
                break
        if cluster_with_overlapping_mention is not None:
            # Merge cluster we are currently processing into
            # the cluster in the processed list.
            cluster_with_overlapping_mention.update(cluster)
        else:
            merged_clusters.append(set(cluster))
    return [list(c) for c in merged_clusters]