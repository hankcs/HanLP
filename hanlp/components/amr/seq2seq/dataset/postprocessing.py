from collections import defaultdict, Counter
import enum
import re
import networkx as nx
import penman

from hanlp.components.amr.seq2seq.dataset.penman import pm_encode

BACKOFF = penman.Graph([
    penman.Triple('d2', ':instance', 'dog'),
    penman.Triple('b1', ':instance', 'bark-01'),
    penman.Triple('b1', ':ARG0', 'd2'), ])


def token_processing(tok):
    if tok is None:
        return None
    elif tok.isdigit():
        try:
            return eval(tok)
        except:
            return tok
    elif tok.startswith('"') and (not tok.endswith('"')):
        return tok + '"'
    elif tok.endswith('"') and (not tok.startswith('"')):
        return '"' + tok
    else:
        return tok


def decode_into_node_and_backreferences(subtoken_ids, tokenizer):
    rex_arg = re.compile(f"^{tokenizer.INIT}(op|snt|conj|prep)")
    rex_spc = re.compile(r"<(s|/s|lit|/lit|stop|unk|pad|mask)>")

    # get strings
    subtokens = tokenizer.convert_ids_to_tokens(subtoken_ids)
    # fix backreferences
    subtoken_backreferences = [max(t - len(tokenizer), -1) for t in subtoken_ids]
    # strip padding
    no_pad = [(s, b) for s, b in zip(subtokens, subtoken_backreferences) if s != (tokenizer.INIT + '<pad>')]
    if no_pad:
        subtokens, subtoken_backreferences = zip(*no_pad)
    else:
        subtokens, subtoken_backreferences = ['<s>'], [-1]

    # subword collapse
    tokens = []
    backreferences = []
    subword_to_token_map = {}
    current_token_i = 0
    for subw_i, (subw_backr, subtok) in enumerate(zip(subtoken_backreferences, subtokens)):
        subword_to_token_map[subw_i] = current_token_i

        # if empty you cannot do anything but add a new word
        if not tokens:
            tokens.append(subtok.lstrip(tokenizer.INIT))
            backreferences.append(-1)
            current_token_i += 1

        # backref can't be splitted
        elif subw_backr > -1:
            tokens.append(None)
            backreferences.append(subword_to_token_map[subw_backr])
            current_token_i += 1

        # after a special token release
        elif isinstance(tokens[-1], str) and rex_spc.match(tokens[-1]):
            tokens.append(subtok.lstrip(tokenizer.INIT))
            backreferences.append(-1)
            current_token_i += 1

        # after a subtoken ':' (which should be followed by the rest of the edge) ignore tokenizer.INIT
        # TODO: this is an ugly patch due to the fact that BART tokenizer splits after ':'
        elif (tokens[-1] == ':') and rex_arg.match(subtok):
            tokens[-1] = tokens[-1] + subtok[1:]

        # leading tokenizer.INIT
        elif subtok.startswith(tokenizer.INIT):
            tokens.append(subtok.lstrip(tokenizer.INIT))
            backreferences.append(-1)
            current_token_i += 1

        # very ugly patch for some cases in which tokenizer.INIT is not in the following token to the edge
        elif isinstance(tokens[-1], str) and tokens[-1].startswith(':') and tokens[-1][-1].isdigit() and (
                subtok != '-of'):
            tokens.append(subtok.lstrip(tokenizer.INIT))
            backreferences.append(-1)
            current_token_i += 1

        # in any other case attach to the previous
        else:
            tokens[-1] = tokens[-1] + subtok

    # strip INIT and fix byte-level
    tokens = [tokenizer.convert_tokens_to_string(list(t)).lstrip() if isinstance(t, str) else t for t in tokens]
    # tokens = [t.replace(tokenizer.INIT, '') if isinstance(t, str) else t for t in tokens]

    # unks are substituted with thing
    tokens = [t if t != '<unk>' else 'thing' for t in tokens]

    old_tokens = tokens
    old_backreferences = backreferences

    # <lit> Barack Obama </lit> -> "Barack Obama"
    tokens = []
    backreferences = []
    token_to_token_map = {}
    start_search = 0
    removed = 0
    while True:
        try:

            lit_start = old_tokens.index('<lit>', start_search)
            token_addition = old_tokens[start_search:lit_start]
            for i, t in enumerate(token_addition, start=start_search):
                token_to_token_map[i] = i - removed
            tokens += token_addition

            backreferences_addition = [token_to_token_map[b] if b > -1 else -1 for b in
                                       old_backreferences[start_search:lit_start]]
            backreferences += backreferences_addition

            lit_end = min(lit_start + 2, len(old_tokens) - 1)

            while lit_end < len(old_tokens):
                old_tok = old_tokens[lit_end]

                if isinstance(old_tok, str) and (
                        (old_tok.startswith(':') and len(old_tok) > 3) or (old_tok == '<stop>')):
                    res_tok = old_tokens[lit_start + 1:lit_end]
                    for i in range(lit_start, lit_end):
                        token_to_token_map[i] = len(tokens)

                    # Remove possible wrong None
                    res = old_tokens[lit_start + 1:lit_end]
                    res = [str(r) for r in res if r is not None]
                    res = '"' + '_'.join(res) + '"'

                    removed += len(res_tok)
                    start_search = lit_end
                    tokens += [res, old_tok]
                    backreferences += [-1, -1]
                    break

                elif old_tok == '</lit>':
                    res_tok = old_tokens[lit_start + 1:lit_end]
                    for i in range(lit_start, lit_end + 1):
                        token_to_token_map[i] = len(tokens)

                    # Remove possible wrong None
                    res = old_tokens[lit_start + 1:lit_end]
                    res = [str(r) for r in res if r is not None]
                    res = '"' + '_'.join(res) + '"'

                    removed += len(res_tok) + 1
                    start_search = lit_end + 1
                    tokens.append(res)
                    backreferences.append(-1)
                    break

                else:
                    lit_end += 1
                    start_search = lit_end

        except ValueError:
            token_addition = old_tokens[start_search:]
            for i, t in enumerate(token_addition, start=start_search):
                token_to_token_map[i] = i - removed
            backreferences_addition = [token_to_token_map[b] if b > -1 else b for b in
                                       old_backreferences[start_search:]]
            tokens += token_addition
            backreferences += backreferences_addition
            break

    tokens = [token_processing(t) for t in tokens]

    shift = 1
    if len(tokens) > 1 and tokens[1] == '<s>':
        shift = 2

    tokens = tokens[shift:]
    backreferences = [b if b == -1 else b - shift for b in backreferences[shift:]]

    if tokens and tokens[-1] == '</s>':
        tokens.pop()
        backreferences.pop()

    return tokens, backreferences


def decode_into_node_and_backreferences_without_space(subtoken_ids, tokenizer):
    rex_arg = re.compile(f"^{tokenizer.INIT}(op|snt|conj|prep)")
    rex_spc = re.compile(r"<(s|/s|lit|/lit|stop|unk|pad|mask)>")

    # get strings
    subtokens = tokenizer.convert_ids_to_tokens(subtoken_ids)
    # fix backreferences
    subtoken_backreferences = [max(t - len(tokenizer), -1) for t in subtoken_ids]
    # strip padding
    no_pad = [(s, b) for s, b in zip(subtokens, subtoken_backreferences) if s != (tokenizer.INIT + '<pad>')]
    if no_pad:
        subtokens, subtoken_backreferences = zip(*no_pad)
    else:
        subtokens, subtoken_backreferences = ['<s>'], [-1]

    # subword collapse
    tokens = []
    backreferences = []
    subword_to_token_map = {}
    current_token_i = 0
    prev_is_pointer = False
    prev_is_rel = False
    for subw_i, (subw_backr, subtok) in enumerate(zip(subtoken_backreferences, subtokens)):
        subword_to_token_map[subw_i] = current_token_i
        is_pointer = subtok.startswith('<pointer:') and subtok.endswith('>')
        is_rel = subtok.startswith(':') and len(subtok) > 1
        is_bracket = subtok in '()'

        # if empty you cannot do anything but add a new word
        if not tokens:
            tokens.append(subtok)
            backreferences.append(-1)
            current_token_i += 1

        # backref can't be splitted
        elif subw_backr > -1:
            tokens.append(None)
            backreferences.append(subword_to_token_map[subw_backr])
            current_token_i += 1

        # after a special token release
        elif isinstance(tokens[-1], str) and rex_spc.match(tokens[-1]):
            tokens.append(subtok)
            backreferences.append(-1)
            current_token_i += 1

        # after a subtoken ':' (which should be followed by the rest of the edge) ignore tokenizer.INIT
        # TODO: this is an ugly patch due to the fact that BART tokenizer splits after ':'
        elif (tokens[-1] == ':') and rex_arg.match(subtok):
            tokens[-1] = tokens[-1] + subtok[1:]

        # current or prev is a control token
        elif (is_pointer or is_rel or prev_is_pointer or prev_is_rel or is_bracket or subtok == '</s>') \
                and subtok != '-of':
            tokens.append(subtok)
            backreferences.append(-1)
            current_token_i += 1

        # very ugly patch for some cases in which tokenizer.INIT is not in the following token to the edge
        elif isinstance(tokens[-1], str) and tokens[-1].startswith(':') and tokens[-1][-1].isdigit() and (
                subtok != '-of'):
            tokens.append(subtok)
            backreferences.append(-1)
            current_token_i += 1

        # in any other case attach to the previous
        else:
            tokens[-1] = tokens[-1] + subtok

        prev_is_pointer = is_pointer
        prev_is_rel = is_rel

    # strip INIT and fix byte-level
    tokens = [tokenizer.convert_tokens_to_string(list(t)).lstrip() if isinstance(t, str) else t for t in tokens]
    # tokens = [t.replace(tokenizer.INIT, '') if isinstance(t, str) else t for t in tokens]

    # unks are substituted with thing
    tokens = [t if t != '<unk>' else 'thing' for t in tokens]

    old_tokens = tokens
    old_backreferences = backreferences

    # <lit> Barack Obama </lit> -> "Barack Obama"
    tokens = []
    backreferences = []
    token_to_token_map = {}
    start_search = 0
    removed = 0
    while True:
        try:

            lit_start = old_tokens.index('<lit>', start_search)
            token_addition = old_tokens[start_search:lit_start]
            for i, t in enumerate(token_addition, start=start_search):
                token_to_token_map[i] = i - removed
            tokens += token_addition

            backreferences_addition = [token_to_token_map[b] if b > -1 else -1 for b in
                                       old_backreferences[start_search:lit_start]]
            backreferences += backreferences_addition

            lit_end = min(lit_start + 2, len(old_tokens) - 1)

            while lit_end < len(old_tokens):
                old_tok = old_tokens[lit_end]

                if isinstance(old_tok, str) and (
                        (old_tok.startswith(':') and len(old_tok) > 3) or (old_tok == '<stop>')):
                    res_tok = old_tokens[lit_start + 1:lit_end]
                    for i in range(lit_start, lit_end):
                        token_to_token_map[i] = len(tokens)

                    # Remove possible wrong None
                    res = old_tokens[lit_start + 1:lit_end]
                    res = [str(r) for r in res if r is not None]
                    res = '"' + '_'.join(res) + '"'

                    removed += len(res_tok)
                    start_search = lit_end
                    tokens += [res, old_tok]
                    backreferences += [-1, -1]
                    break

                elif old_tok == '</lit>':
                    res_tok = old_tokens[lit_start + 1:lit_end]
                    for i in range(lit_start, lit_end + 1):
                        token_to_token_map[i] = len(tokens)

                    # Remove possible wrong None
                    res = old_tokens[lit_start + 1:lit_end]
                    res = [str(r) for r in res if r is not None]
                    res = '"' + '_'.join(res) + '"'

                    removed += len(res_tok) + 1
                    start_search = lit_end + 1
                    tokens.append(res)
                    backreferences.append(-1)
                    break

                else:
                    lit_end += 1
                    start_search = lit_end

        except ValueError:
            token_addition = old_tokens[start_search:]
            for i, t in enumerate(token_addition, start=start_search):
                token_to_token_map[i] = i - removed
            backreferences_addition = [token_to_token_map[b] if b > -1 else b for b in
                                       old_backreferences[start_search:]]
            tokens += token_addition
            backreferences += backreferences_addition
            break

    tokens = [token_processing(t) for t in tokens]

    shift = 0
    if len(tokens) > 1 and tokens[1] == '<s>':
        shift = 1

    tokens = tokens[shift:]
    backreferences = [b if b == -1 else b - shift for b in backreferences[shift:]]

    if tokens and tokens[-1] == '</s>':
        tokens.pop()
        backreferences.pop()

    return tokens, backreferences


def index_of(element, iterable, default=None, start=None, end=None):
    if not callable(element):
        def check(x):
            return element == x
    else:
        check = element
    if start is None:
        start = 0
    if end is None:
        end = len(iterable)
    item = start
    while item < end:
        if check(iterable[item]):
            return item
        item += 1
    return default


def separate_edges_nodes(edges_nodes_slice, *other):
    is_arg = lambda x: isinstance(x, str) and x.startswith(':')
    start = 0
    edges = []
    nodes = []
    l = len(edges_nodes_slice)
    while start < l:
        edge_index = index_of(
            is_arg,
            edges_nodes_slice,
            start=start)
        if edge_index is None or edge_index == (l - 1):
            break
        if is_arg(edges_nodes_slice[edge_index + 1]):
            start = edge_index + 1
            continue
        edges.append(edge_index)
        nodes.append(edge_index + 1)
        start = edge_index + 2
    ret = []
    for oth in other:
        edges_oth = [oth[i] for i in edges]
        nodes_oth = [oth[i] for i in nodes]
        ret.append((edges_oth, nodes_oth))
    return ret


def _split_name_ops(graph):
    # identify name triples
    name_vars = {}
    for i, (v1, rel, v2) in enumerate(graph.triples):
        if rel == ':instance' and v2 == 'name':
            name_vars[v1] = 1

    # check if they have ops
    name_vars_to_ops = defaultdict(list)
    for i, (v1, rel, v2) in enumerate(graph.triples):
        if v1 in name_vars and rel.startswith(':op'):
            name_vars_to_ops[v1].append((i, rel, v2.strip('"')))

    triples = graph.triples.copy()
    for nv, ops in name_vars_to_ops.items():
        ops = sorted(ops, key=lambda x: int(x[1][3:]))
        idx, _, lits = zip(*ops)
        for i in idx:
            triples[i] = None

        lits = ['"' + l + '"' for lit in lits for l in lit.split('_')]

        tt = []
        for i, l in enumerate(lits, start=1):
            rel = ':op' + str(i)
            tt.append(penman.Triple(nv, rel, l))

        triples[min(idx)] = tt

    triples = [t if isinstance(t, list) else [t] for t in triples if t is not None]
    triples = [t for tt in triples for t in tt]

    graph_ = penman.Graph(triples)
    graph_.metadata = graph.metadata
    return graph_


def _reconstruct_graph_from_nodes(nodes, backreferences):
    triples = []
    triples_added = set()

    variable2index = {}
    index2variable = {}
    start_index = 0

    cnt = defaultdict(Counter)

    while start_index < len(nodes):
        stop_index = index_of('<stop>', nodes, default=len(nodes) + 1, start=start_index)
        old_start_index = start_index
        start_index = stop_index + 1

        src_node, src_backr = nodes[old_start_index], backreferences[old_start_index]

        if src_node == '<stop>':
            continue

        trg_nodes_edges = nodes[old_start_index:stop_index]
        trg_nodes_edges_backr = backreferences[old_start_index:stop_index]
        trg_nodes_edges_indices = list(range(old_start_index, stop_index))

        if isinstance(src_node, str):
            if src_node in ('<s>', '</s>', '<stop>'):
                continue
            elif ('/' in src_node) or (':' in src_node) or ('(' in src_node) or (')' in src_node):
                src_node = 'thing'

        if src_node is not None:
            src_node = str(src_node)
            src_var = src_node[0].lower()
            if not src_var not in 'abcdefghijklmnopqrstuvwxyz':
                src_var = 'x'
            # src_var = f'{src_var}_{len(variable2index)}'
            src_var = f'{src_var}{len(variable2index)}'
            src_var_i = old_start_index
            variable2index[src_var] = src_var_i
            index2variable[src_var_i] = src_var
            triple = penman.Triple(src_var, ':instance', src_node)
            if triple not in triples_added:
                triples.append(triple)
                triples_added.add(triple)
        else:
            if src_backr in index2variable:
                src_var = index2variable[src_backr]
        # more resilient logic here
        (trg_edges, trg_nodes), (_, trg_nodes_backr), (_, trg_nodes_indices) = \
            separate_edges_nodes(
                trg_nodes_edges,
                trg_nodes_edges,
                trg_nodes_edges_backr,
                trg_nodes_edges_indices)

        for n, e, nb, ni in zip(trg_nodes, trg_edges, trg_nodes_backr, trg_nodes_indices):

            if isinstance(n, str) and n.startswith(':'):
                continue
            if isinstance(n, str) and n.startswith('<') and n.endswith('>'):
                continue
            if e == ':li':
                pass
            elif len(e) < 4 or (not e.startswith(':')):
                continue

            # same edge more than once
            num = cnt[src_var][e]
            # num = 0
            if num:

                if e.startswith(':op') or e.startswith(':snt'):
                    continue
                # elif e.startswith(':ARG'):
                #    continue
                elif num > 3:
                    continue

            if n is None:
                if nb not in index2variable:
                    continue
                trg_var = index2variable[nb]
                trg = trg_var
            elif e == ':mode':
                trg = n
            elif (not isinstance(n, str)) or re.match(r"^[+-]?\d+\.?\d*$", n) or (n == '-') or (n == '+'):
                trg = str(n)
            elif (n.startswith('"') and n.endswith('"') and len(n) > 2):
                trg = '"' + n.replace('"', '') + '"'
            elif ('/' in n) or (':' in n) or ('(' in n) or (')' in n) or ('=' in n):
                trg = f'"{n}"'
            elif n == '"':
                continue
            elif (n.startswith('"') and (not n.endswith('"'))) or (not n.startswith('"') and (n.endswith('"'))) or (
                    '"' in n):
                trg = '"' + n.replace('"', '') + '"'
            else:
                trg_var = n[0].lower()
                if trg_var not in 'abcdefghijklmnopqrstuvwxyz':
                    trg_var = 'x'
                # trg_var = f'{trg_var}_{len(variable2index)}'
                trg_var = f'{trg_var}{len(variable2index)}'
                trg_var_i = ni
                variable2index[trg_var] = trg_var_i
                index2variable[trg_var_i] = trg_var
                triple = penman.Triple(trg_var, ':instance', n)
                if triple not in triples_added:
                    triples.append(triple)
                    triples_added.add(triple)
                trg = trg_var

            triple = penman.Triple(src_var, e, trg)
            if triple not in triples_added:
                triples.append(triple)
                triples_added.add(triple)

            cnt[src_var][e] += 1

    return penman.Graph(triples)


def build_graph(nodes, backreferences, restore_name_ops=False):
    graph = _reconstruct_graph_from_nodes(nodes, backreferences)
    if restore_name_ops:
        graph = _split_name_ops(graph)
    return graph


class ParsedStatus(enum.Enum):
    OK = 0
    FIXED = 1
    BACKOFF = 2


def connect_graph_if_not_connected(graph):
    try:
        encoded = pm_encode(graph)
        return graph, ParsedStatus.OK
    except:
        pass

    nxgraph = nx.MultiGraph()
    variables = graph.variables()
    for v1, _, v2 in graph.triples:
        if v1 in variables and v2 in variables:
            nxgraph.add_edge(v1, v2)
        elif v1 in variables:
            nxgraph.add_edge(v1, v1)

    triples = graph.triples.copy()
    new_triples = []
    addition = f'a{len(variables) + 1}'
    triples.append(penman.Triple(addition, ':instance', 'and'))
    for i, conn_set in enumerate(nx.connected_components(nxgraph), start=1):
        edge = f':op{i}'
        conn_set = sorted(conn_set, key=lambda x: int(x[1:]))
        conn_set = [c for c in conn_set if c in variables]
        node = conn_set[0]
        new_triples.append(penman.Triple(addition, edge, node))
    triples = new_triples + triples
    metadata = graph.metadata
    graph = penman.Graph(triples)
    graph.metadata.update(metadata)
    pm_encode(graph)

    return graph, ParsedStatus.FIXED


def restore_backreferences_from_pointers(nodes):
    new_nodes, new_backreferences = [], []
    prev_pointer = None
    pointer2i = {}
    for n in nodes:
        is_pointer = isinstance(n, str) and n.startswith('<pointer:') and n.endswith('>')

        if not is_pointer:
            if prev_pointer is not None:
                if prev_pointer in pointer2i:
                    new_nodes.append(None)
                    new_backreferences.append(pointer2i[prev_pointer])
                    new_nodes.append(n)
                    new_backreferences.append(-1)

                else:
                    pointer2i[prev_pointer] = len(new_nodes)
                    new_nodes.append(n)
                    new_backreferences.append(-1)
            else:
                new_nodes.append(n)
                new_backreferences.append(-1)

            prev_pointer = None
        else:
            prev_pointer = n
    return new_nodes, new_backreferences
