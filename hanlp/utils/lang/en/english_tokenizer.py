#!/usr/bin/env python
"""
Regex-based word tokenizers.

Note that small/full/half-width character variants are *not* covered.
If a text were to contains such characters, normalize it first.
A modified version of https://github.com/fnl/segtok

- dropped dependency on regex
- dropped web_tokenize
- supported concat word

"""

__author__ = 'Florian Leitner <florian.leitner@gmail.com>'
from re import compile, UNICODE, VERBOSE

SENTENCE_TERMINALS = '.!?\u203C\u203D\u2047\u2048\u2049\u3002' \
                     '\uFE52\uFE57\uFF01\uFF0E\uFF1F\uFF61'
"The list of valid Unicode sentence terminal characters."

# Note that Unicode the category Pd is NOT a good set for valid word-breaking hyphens,
# because it contains many dashes that should not be considered part of a word.
HYPHENS = '\u00AD\u058A\u05BE\u0F0C\u1400\u1806\u2010-\u2012\u2e17\u30A0-'
"Any valid word-breaking hyphen, including ASCII hyphen minus."

APOSTROPHES = '\'\u00B4\u02B9\u02BC\u2019\u2032'
"""All apostrophe-like marks, including the ASCII "single quote"."""

APOSTROPHE = r"[\u00B4\u02B9\u02BC\u2019\u2032]"
"""Any apostrophe-like marks, including "prime" but not the ASCII "single quote"."""

LINEBREAK = r'(?:\r\n|\n|\r|\u2028)'
"""Any valid linebreak sequence (Windows, Unix, Mac, or U+2028)."""

LETTER = r'[^\W\d_]'
"""Any Unicode letter character that can form part of a word: Ll, Lm, Lt, Lu."""

NUMBER = r'\d'
"""Any Unicode number character: Nd or Nl."""

POWER = r'\u207B?[\u00B9\u00B2\u00B3]'
"""Superscript 1, 2, and 3, optionally prefixed with a minus sign."""

SUBDIGIT = r'[\u2080-\u2089]'
"""Subscript digits."""

ALNUM = LETTER[:-1] + NUMBER + ']'
"""Any alphanumeric Unicode character: letter or number."""

HYPHEN = r'[%s]' % HYPHENS

SPACE = r'\s'
"""Any unicode space character plus the (horizontal) tab."""

APO_MATCHER = compile(APOSTROPHE, UNICODE)
"""Matcher for any apostrophe."""

HYPHENATED_LINEBREAK = compile(
    r'({alnum}{hyphen}){space}*?{linebreak}{space}*?({alnum})'.format(
        alnum=ALNUM, hyphen=HYPHEN, linebreak=LINEBREAK, space=SPACE
    ), UNICODE
)
"""
The pattern matches any alphanumeric Unicode character, followed by a hyphen,
a single line-break surrounded by optional (non-breaking) spaces,
and terminates with a alphanumeric character on this next line.
The opening char and hyphen as well as the terminating char are captured in two groups.
"""

IS_POSSESSIVE = compile(r"{alnum}+(?:{hyphen}{alnum}+)*(?:{apo}[sS]|[sS]{apo})$".format(
    alnum=ALNUM, hyphen=HYPHEN, apo="['" + APOSTROPHE[1:]
), UNICODE
)
"""A pattern that matches English words with a possessive s terminal form."""

IS_CONTRACTION = compile(r"{alnum}+(?:{hyphen}{alnum}+)*{apo}(?:d|ll|m|re|s|t|ve)$".format(
    alnum=ALNUM, hyphen=HYPHEN, apo="['" + APOSTROPHE[1:]
), UNICODE
)
"""A pattern that matches tokens with valid English contractions ``'(d|ll|m|re|s|t|ve)``."""

MAP_CONCAT_WORD = {'aint': [2, 4], 'arent': [3, 5], 'cant': [2, 4], 'cannot': [3, 6], 'coulda': [5, 6],
                   'couldnt': [5, 7], 'didnt': [3, 5], 'doncha': [2, 3, 6], 'dont': [2, 4],
                   'doesnt': [4, 6], 'dunno': [2, 3, 5], 'finna': [3, 5], 'gimme': [3, 5], 'gonna': [3, 5],
                   'gotta': [3, 5], 'hadnt': [3, 5], 'hasnt': [3, 5], 'havent': [4, 6], 'isnt': [2, 4],
                   'itd': [2, 3], 'itll': [2, 4], 'lemme': [3, 5], 'lets': [3, 4], 'mightnt': [5, 7],
                   'mustnt': [4, 6], 'shant': [3, 5], 'shoulda': [6, 7], 'shouldnt': [6, 8],
                   'thatd': [4, 5], 'thatll': [4, 6], 'thats': [4, 5], 'theyd': [4, 5], 'theyre': [4, 6],
                   'theyve': [4, 6], 'wanna': [3, 5], 'wasnt': [3, 5], 'weve': [2, 4], 'werent': [4, 6],
                   'whadya': [3, 4, 6], 'whatcha': [4, 7], 'whatre': [4, 6], 'whats': [4, 5],
                   'whatve': [4, 6], 'whatz': [4, 5], 'whod': [3, 4], 'wholl': [3, 5], 'woncha': [2, 3, 6],
                   'wont': [2, 4], 'woulda': [5, 6], 'wouldnt': [5, 7], 'youd': [3, 4], 'youll': [3, 5],
                   'youve': [3, 5], "'tis": [2, 4], "'twas": [2, 5], "d'ye": [2, 4], "don'cha": [2, 4, 7],
                   "i'mma": [1, 3, 5], "i'mmm": [1, 5], "more'n": [4, 6], '’tis': [2, 4], '’twas': [2, 5],
                   'd’ye': [2, 4], 'don’cha': [2, 4, 7], 'i’mma': [1, 3, 5], 'i’mmm': [1, 5],
                   'more’n': [4, 6]}

RE_APOSTROPHE = compile(r'(?i)[a-z](n[\'\u2019]t|[\'\u2019](ll|nt|re|ve|[dmstz]))(\W|$)')


def split_possessive_markers(tokens):
    """
    A function to split possessive markers at the end of alphanumeric (and hyphenated) tokens.

    Takes the output of any of the tagger functions and produces and updated list.
    To use it, simply wrap the tagger function, for example::

    >>> my_sentence = "This is Fred's latest book."
    >>> split_possessive_markers(tokenize_english(my_sentence))
    ['This', 'is', 'Fred', "'s", 'latest', 'book', '.']

    :param tokens: a list of tokens
    :returns: an updated list if a split was made or the original list otherwise
    """
    idx = -1

    for token in list(tokens):
        idx += 1

        if IS_POSSESSIVE.match(token) is not None:
            if token[-1].lower() == 's' and token[-2] in APOSTROPHES:
                tokens.insert(idx, token[:-2])
                idx += 1
                tokens[idx] = token[-2:]
            elif token[-2].lower() == 's' and token[-1] in APOSTROPHES:
                tokens.insert(idx, token[:-1])
                idx += 1
                tokens[idx] = token[-1:]

    return tokens


def split_contractions(tokens):
    """
    A function to split apostrophe contractions at the end of alphanumeric (and hyphenated) tokens.

    Takes the output of any of the tagger functions and produces and updated list.

    :param tokens: a list of tokens
    :returns: an updated list if a split was made or the original list otherwise
    """
    idx = -1

    for token in list(tokens):
        idx += 1

        if IS_CONTRACTION.match(token) is not None:
            length = len(token)

            if length > 1:
                for pos in range(length - 1, -1, -1):
                    if token[pos] in APOSTROPHES:
                        if 2 < length and pos + 2 == length and token[-1] == 't' and token[pos - 1] == 'n':
                            pos -= 1

                        tokens.insert(idx, token[:pos])
                        idx += 1
                        tokens[idx] = token[pos:]

    return tokens


def _matches(regex):
    """Regular expression compiling function decorator."""

    def match_decorator(fn):
        automaton = compile(regex, UNICODE | VERBOSE)
        fn.split = automaton.split
        fn.match = automaton.match
        return fn

    return match_decorator


@_matches(r'\s+')
def space_tokenizer(sentence):
    """
    For a given input `sentence`, return a list of its tokens.

    Split on Unicode spaces ``\\s+`` (i.e., any kind of **Unicode** space character).
    The separating space characters are not included in the resulting token list.
    """
    return [token for token in space_tokenizer.split(sentence) if token]


@_matches(r'(%s+)' % ALNUM)
def symbol_tokenizer(sentence):
    """
    The symbol tagger extends the :func:`space_tokenizer` by separating alphanumerics.

    Separates alphanumeric Unicode character sequences in already space-split tokens.
    """
    return [token for span in space_tokenizer(sentence) for
            token in symbol_tokenizer.split(span) if token]


@_matches(r"""((?:
    # Dots, except ellipsis
    {alnum} \. (?!\.\.)
    | # Comma, surrounded by digits (e.g., chemicals) or letters
    {alnum} , (?={alnum})
    | # Colon, surrounded by digits (e.g., time, references)
    {number} : (?={number})
    | # Hyphen, surrounded by digits (e.g., DNA endings: "5'-ACGT-3'") or letters
    {alnum} {apo}? {hyphen} (?={alnum})  # incl. optional apostrophe for DNA segments
    | # Apostophes, non-consecutive
    {apo} (?!{apo})
    | # ASCII single quote, surrounded by digits or letters (no dangling allowed)
    {alnum} ' (?={alnum})
    | # ASCII single quote after an s and at the token's end
    s ' $
    | # Terminal dimensions (superscript minus, 1, 2, and 3) attached to physical units
    #  size-prefix                 unit-acronym    dimension
    \b [yzafpn\u00B5mcdhkMGTPEZY]? {letter}{{1,3}} {power} $
    | # Atom counts (subscript numbers) and ionization states (optional superscript
    #   2 or 3 followed by a + or -) are attached to valid fragments of a chemical formula
    \b (?:[A-Z][a-z]?|[\)\]])+ {subdigit}+ (?:[\u00B2\u00B3]?[\u207A\u207B])?
    | # Any (Unicode) letter, digit, or the underscore
    {alnum}
    )+)""".format(alnum=ALNUM, apo=APOSTROPHE, power=POWER, subdigit=SUBDIGIT,
                  hyphen=HYPHEN, letter=LETTER, number=NUMBER))
def tokenize_english(sentence):
    """
    A modified version of the segtok tagger: https://github.com/fnl/segtok
    This tagger extends the alphanumeric :func:`symbol_tokenizer` by splitting fewer cases:

    1. Dots appearing after a letter are maintained as part of the word, except for the last word
       in a sentence if that dot is the sentence terminal. Therefore, abbreviation marks (words
       containing or ending in a ``.``, like "i.e.") remain intact and URL or ID segments remain
       complete ("www.ex-ample.com", "EC1.2.3.4.5", etc.). The only dots that never are attached
       are triple dots (``...``; ellipsis).
    2. Commas surrounded by alphanumeric characters are maintained in the word, too, e.g. ``a,b``.
       Colons surrounded by digits are maintained, e.g., 'at 12:30pm' or 'Isaiah 12:3'.
       Commas, semi-colons, and colons dangling at the end of a token are always spliced off.
    3. Any two alphanumeric letters that are separated by a single hyphen are joined together;
       Those "inner" hyphens may optionally be followed by a linebreak surrounded by spaces;
       The spaces will be removed, however. For example, ``Hel- \\r\\n \t lo`` contains a (Windows)
       linebreak and will be returned as ``Hel-lo``.
    4. Apostrophes are always allowed in words as long as they are not repeated; The single quote
       ASCII letter ``'`` is only allowed as a terminal apostrophe after the letter ``s``,
       otherwise it must be surrounded by letters. To support DNA and chemicals, a apostrophe
       (prime) may be located before the hyphen, as in the single token "5'-ACGT-3'" (if any
       non-ASCII hyphens are used instead of the shown single quote).
    5. Superscript 1, 2, and 3, optionally prefixed with a superscript minus, are attached to a
       word if it is no longer than 3 letters (optionally 4 if the first letter is a power prefix
       in the range from yocto, y (10^-24) to yotta, Y (10^+24)).
    6. Subscript digits are attached if prefixed with letters that look like a chemical formula.
    """
    if not sentence:
        return []
    flat = not isinstance(sentence, list)
    if flat:
        sents = [sentence]
    else:
        sents = sentence
    results = []
    for sentence in sents:
        pruned = HYPHENATED_LINEBREAK.sub(r'\1\2', sentence)
        tokens = [token for span in space_tokenizer(pruned) for
                  token in tokenize_english.split(span) if token]

        # splice the sentence terminal off the last word/token if it has any at its borders
        # only look for the sentence terminal in the last three tokens
        for idx, word in enumerate(reversed(tokens[-3:]), 1):
            if (tokenize_english.match(word) and not APO_MATCHER.match(word)) or \
                    any(t in word for t in SENTENCE_TERMINALS):
                last = len(word) - 1

                if 0 == last or u'...' == word:
                    # any case of "..." or any single char (last == 0)
                    pass  # leave the token as it is
                elif any(word.rfind(t) == last for t in SENTENCE_TERMINALS):
                    # "stuff."
                    tokens[-idx] = word[:-1]
                    tokens.insert(len(tokens) - idx + 1, word[-1])
                elif any(word.find(t) == 0 for t in SENTENCE_TERMINALS):
                    # ".stuff"
                    tokens[-idx] = word[0]
                    tokens.insert(len(tokens) - idx + 1, word[1:])

                break

        # keep splicing off any dangling commas and (semi-) colons
        dirty = True
        while dirty:
            dirty = False

            for idx, word in enumerate(reversed(tokens), 1):
                while len(word) > 1 and word[-1] in u',;:':
                    char = word[-1]  # the dangling comma/colon
                    word = word[:-1]
                    tokens[-idx] = word
                    tokens.insert(len(tokens) - idx + 1, char)
                    idx += 1
                    dirty = True
                if dirty:
                    break  # restart check to avoid index errors

        # split concat words
        chunks = []
        for token in tokens:
            t = MAP_CONCAT_WORD.get(token.lower(), None)
            if t:
                i = 0
                for j in t:
                    chunks.append(token[i:j])
                    i = j
            else:
                chunks.append(token)
        tokens = chunks
        # split APOSTROPHE
        chunks = []
        for token in tokens:
            m = RE_APOSTROPHE.search(token)
            if m:
                chunks.append(token[:m.start(1)])
                chunks.append(token[m.start(1):m.end(1)])
                if m.end(1) < len(token):
                    chunks.append(token[m.end(1):])
            else:
                chunks.append(token)
        tokens = chunks
        results.append(tokens)
    return results[0] if flat else results