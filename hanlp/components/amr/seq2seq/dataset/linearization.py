import abc
import itertools
from collections import deque, defaultdict
import re
from typing import List, Optional, Dict, Any, Set, TypeVar
from dataclasses import dataclass
import networkx as nx
import penman


@dataclass
class SemanticGraph:
    nodes_var: List[str]
    """
    List of linearized nodes, with special tokens.
    """
    edges: Optional[List[str]]
    """
    List of linearized edges, with special tokens.
    """
    backreferences: List[int]
    """
    List of backpointers to handle rentrancies and cycles.
    """
    var2instance: Dict[str, str]
    """
    Dict from var ids to 'lemmatized' readable strings qualifying the node (collapsing the :instance edge for AMR).
    """
    extra: Dict[str, Any]
    """
    Holds extra stuff that might be useful, e.g. alignments, NER, EL.
    """

    # @cached_property
    @property
    def variables(self) -> Set[str]:
        """Set of variables in this semantic graph"""
        variables = {v for v in self.nodes_var if not v.startswith('<')}
        return variables

    @property
    def resolved_nodes_var(self) -> List[str]:
        return [self.nodes_var[b] for b in self.backreferences]

    # @cached_property
    @property
    def nodes(self) -> List[str]:
        """Linearized nodes with varids replaced by instances"""
        return [self.var2instance.get(node, node) for node in self.nodes_var]

    @property
    def resolved_nodes(self) -> List[str]:
        return [self.nodes[b] for b in self.backreferences]

    def src_occurrence(self, var: str) -> int:
        pass


class BaseLinearizer(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def linearize(self, *args, **kwargs) -> SemanticGraph:
        pass


class AMRTokens:
    START, END = '<', '>'
    _TEMPL = START + '{}' + END

    BOS_N = _TEMPL.format('s')
    EOS_N = _TEMPL.format('/s')
    START_N = _TEMPL.format('start')
    STOP_N = _TEMPL.format('stop')
    PNTR_N = _TEMPL.format('pointer')

    LIT_START = _TEMPL.format('lit')
    LIT_END = _TEMPL.format('/lit')

    BACKR_SRC_N = _TEMPL.format('backr:src:XXX')
    BACKR_TRG_N = _TEMPL.format('backr:trg:XXX')

    BOS_E = _TEMPL.format('s')
    EOS_E = _TEMPL.format('/s')
    START_E = _TEMPL.format('start')
    STOP_E = _TEMPL.format('stop')

    _FIXED_SPECIAL_TOKENS_N = {
        BOS_N, EOS_N, START_N, STOP_N}
    _FIXED_SPECIAL_TOKENS_E = {
        BOS_E, EOS_E, START_E, STOP_E}
    _FIXED_SPECIAL_TOKENS = _FIXED_SPECIAL_TOKENS_N | _FIXED_SPECIAL_TOKENS_E

    # match and read backreferences
    _re_BACKR_SRC_N = re.compile(BACKR_SRC_N.replace('XXX', r'([0-9]+)'))
    _re_BACKR_TRG_N = re.compile(BACKR_TRG_N.replace('XXX', r'([0-9]+)'))

    @classmethod
    def is_node(cls, string: str) -> bool:
        if isinstance(string, str) and string.startswith(':'):
            return False
        elif string in cls._FIXED_SPECIAL_TOKENS_E:
            return False
        return True

    @classmethod
    def read_backr(cls, string: str) -> Optional:
        m_src = cls._re_BACKR_SRC_N.search(string)
        if m_src is not None:
            return m_src
        m_trg = cls._re_BACKR_TRG_N.search(string)
        if m_trg is not None:
            return m_trg
        return None


T = TypeVar('T')


def index_default(
        item: T, list_: List[T],
        start: Optional[int] = None,
        stop: Optional[int] = None,
        default: Optional[int] = None
):
    if start is None:
        start = 0
    if stop is None:
        stop = len(list_)
    return next((i for i, x in enumerate(list_[start:stop], start=start) if x == item), default)


class AMRLinearizer(BaseLinearizer):

    def __init__(
            self,
            use_pointer_tokens: bool = True,
            collapse_name_ops: bool = False,
    ):
        self.collapse_name_ops = collapse_name_ops
        self.interleave_edges = False
        self.use_pointer_tokens = use_pointer_tokens

    def _collapse_name_ops(self, amr):
        # identify name triples
        name_vars = {}
        for i, (v1, rel, v2) in enumerate(amr.triples):
            if rel == ':instance' and v2 == 'name':
                name_vars[v1] = 1

        # check if they have ops
        name_vars_to_ops = defaultdict(list)
        for i, (v1, rel, v2) in enumerate(amr.triples):
            if v1 in name_vars and rel.startswith(':op'):
                name_vars_to_ops[v1].append((i, rel, v2.strip('"')))

        triples = amr.triples.copy()
        for nv, ops in name_vars_to_ops.items():
            ops = sorted(ops, key=lambda x: int(x[1][3:]))
            idx, _, lits = zip(*ops)
            for i in idx:
                triples[i] = None
            lit = '"' + '_'.join(lits) + '"'
            triples[min(idx)] = penman.Triple(nv, ':op1', lit)

        triples = [t for t in triples if t is not None]
        amr_ = penman.Graph(triples)
        amr_.metadata = amr.metadata
        return amr_

    def linearize(self, amr: penman.Graph) -> SemanticGraph:
        if self.collapse_name_ops:
            amr = self._collapse_name_ops(amr)
        linearized = self._linearize(amr)
        linearized = self._interleave(linearized)
        if self.use_pointer_tokens:
            linearized = self._add_pointer_tokens(linearized)
        return linearized

    def _linearize(self, amr: penman.Graph) -> SemanticGraph:
        variables = set(amr.variables())
        variables = {'var:' + v for v in variables}
        var2instance = {}

        graph = nx.MultiDiGraph()

        triples2order = {k: i for i, k in enumerate(amr.triples)}

        for triple in amr.triples:
            var, rel, instance = triple
            order = triples2order[triple]
            if rel != ':instance':
                continue
            for expansion_candidate in itertools.chain(range(order - 1, -1), range(order + 1, len(amr.triples))):
                if var == amr.triples[expansion_candidate][2]:
                    expansion = expansion_candidate
                    break
            else:
                expansion = 0
            var = 'var:' + var
            var2instance[var] = instance
            graph.add_node(var, instance=instance, order=order, expansion=expansion)

        for triple in amr.edges():
            var1, rel, var2 = triple
            order = triples2order[triple]
            if rel == ':instance':
                continue
            var1 = 'var:' + var1
            var2 = 'var:' + var2
            graph.add_edge(var1, var2, rel=rel, order=order)

        for triple in amr.attributes():
            var, rel, attr = triple
            order = triples2order[triple]
            if rel == ':instance':
                continue
            var = 'var:' + var
            graph.add_edge(var, attr, rel=rel, order=order)

        # nodes that are not reachable from the root (e.g. because of reification)
        # will be present in the not_explored queue
        # undirected_graph = graph.to_undirected()
        # print(amr.variables())
        not_explored = deque(sorted(variables, key=lambda x: nx.get_node_attributes(graph, 'order')[x]))
        # (
        #     len(nx.shortest_path(undirected_graph, 'var:' + amr.top, x)),
        #     -graph.out_degree(x),
        # )

        first_index = {}
        explored = set()
        added_to_queue = set()
        nodes_visit = [AMRTokens.BOS_N]
        edges_visit = [AMRTokens.BOS_E]
        backreferences = [0]
        queue = deque()
        queue.append('var:' + amr.top)

        while queue or not_explored:

            if queue:
                node1 = queue.popleft()
            else:
                node1 = not_explored.popleft()
                if node1 in added_to_queue:
                    continue
                if not list(graph.successors(node1)):
                    continue

            if node1 in variables:
                if node1 in explored:
                    continue
                if node1 in first_index:
                    nodes_visit.append(AMRTokens.BACKR_TRG_N)
                    backreferences.append(first_index[node1])
                else:
                    backreferences.append(len(nodes_visit))
                    first_index[node1] = len(nodes_visit)
                    nodes_visit.append(node1)
                edges_visit.append(AMRTokens.START_E)

                successors = []
                for node2 in graph.successors(node1):
                    for edge_data in graph.get_edge_data(node1, node2).values():
                        rel = edge_data['rel']
                        order = edge_data['order']
                        successors.append((order, rel, node2))
                successors = sorted(successors)

                for order, rel, node2 in successors:
                    edges_visit.append(rel)

                    # node2 is a variable
                    if node2 in variables:
                        # ... which was mentioned before
                        if node2 in first_index:
                            nodes_visit.append(AMRTokens.BACKR_TRG_N)
                            backreferences.append(first_index[node2])

                        # .. which is mentioned for the first time
                        else:
                            backreferences.append(len(nodes_visit))
                            first_index[node2] = len(nodes_visit)
                            nodes_visit.append(node2)

                        # 1) not already in Q
                        # 2) has children
                        # 3) the edge right before its expansion has been encountered
                        if (node2 not in added_to_queue) and list(graph.successors(node2)) and (
                                nx.get_node_attributes(graph, 'expansion')[node2] <= order):
                            queue.append(node2)
                            added_to_queue.add(node2)

                    # node2 is a constant
                    else:
                        backreferences.append(len(nodes_visit))
                        nodes_visit.append(node2)

                backreferences.append(len(nodes_visit))
                nodes_visit.append(AMRTokens.STOP_N)
                edges_visit.append(AMRTokens.STOP_E)
                explored.add(node1)

            else:
                backreferences.append(len(nodes_visit))
                nodes_visit.append(node1)
                explored.add(node1)

        backreferences.append(len(nodes_visit))
        nodes_visit.append(AMRTokens.EOS_N)
        edges_visit.append(AMRTokens.EOS_E)
        assert len(nodes_visit) == len(edges_visit) == len(backreferences)
        return SemanticGraph(
            nodes_visit,
            edges_visit,
            backreferences,
            var2instance,
            extra={'graph': graph, 'amr': amr}
        )

    def _interleave(self, graph: SemanticGraph) -> SemanticGraph:

        new_backreferences_map = []
        new_nodes = []
        new_edges = None
        new_backreferences = []

        # to isolate sublist to the stop token
        start_i = 1
        end_i = index_default(AMRTokens.STOP_N, graph.nodes_var, start_i, -1, -1)

        def add_node(node, backr=None):
            old_n_node = len(new_backreferences_map)
            new_n_node = len(new_nodes)

            if backr is None:
                backr = old_n_node

            new_backreferences_map.append(new_n_node)
            new_nodes.append(node)
            if old_n_node == backr:
                new_backreferences.append(new_n_node)
            else:
                new_backreferences.append(new_backreferences_map[backr])

        def add_edge(edge):
            new_nodes.append(edge)
            new_backreferences.append(len(new_backreferences))

        add_node(AMRTokens.BOS_N)

        while end_i > -1:

            # src node
            add_node(graph.nodes_var[start_i], graph.backreferences[start_i])

            # edges and trg nodes, interleaved
            nodes = graph.nodes_var[start_i + 1:end_i]
            edges = graph.edges[start_i + 1:end_i]
            backr = graph.backreferences[start_i + 1:end_i]
            for n, e, b in zip(nodes, edges, backr):
                add_edge(e)
                add_node(n, b)

            # stop
            add_node(graph.nodes_var[end_i], graph.backreferences[end_i])

            start_i = end_i + 1
            end_i = index_default(AMRTokens.STOP_N, graph.nodes_var, start_i, -1, -1)

        add_node(AMRTokens.EOS_N)

        new_graph = SemanticGraph(
            new_nodes,
            None,
            new_backreferences,
            graph.var2instance,
            extra=graph.extra,
        )
        return new_graph

    def _add_pointer_tokens(self, graph: SemanticGraph) -> SemanticGraph:
        new_nodes = []
        var2pointer = {}
        for node, backr in zip(graph.nodes_var, graph.backreferences):

            if node == AMRTokens.BACKR_TRG_N:
                node = graph.nodes_var[backr]
                pointer = var2pointer[node]
                new_nodes.append(pointer)
            elif node in graph.var2instance:
                pointer = var2pointer.setdefault(node, f"<pointer:{len(var2pointer)}>")
                new_nodes.append(pointer)
                new_nodes.append(node)
            else:
                new_nodes.append(node)

        new_backreferences = list(range(len(new_nodes)))
        new_graph = SemanticGraph(
            new_nodes,
            None,
            new_backreferences,
            graph.var2instance,
            extra=graph.extra,
        )
        return new_graph


