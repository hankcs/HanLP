from typing import List

from penman import load as load_, Graph, Triple
from penman import loads as loads_
from penman import encode as encode_
from penman.model import Model
from penman.models.noop import NoOpModel
from penman.models import amr
import penman
import logging

op_model = Model()
noop_model = NoOpModel()
amr_model = amr.model
DEFAULT = op_model

# Mute loggers
penman.layout.logger.setLevel(logging.CRITICAL)
penman._parse.logger.setLevel(logging.CRITICAL)


def _get_model(dereify):
    if dereify is None:
        return DEFAULT
    elif dereify:
        return op_model
    else:
        return noop_model


def _remove_wiki(graph):
    metadata = graph.metadata
    triples = []
    for t in graph.triples:
        v1, rel, v2 = t
        if rel == ':wiki':
            t = Triple(v1, rel, '+')
        triples.append(t)
    graph = Graph(triples)
    graph.metadata = metadata
    return graph


def pm_load(source, dereify=None, remove_wiki=False) -> List[penman.Graph]:
    """

    Args:
        source:
        dereify: Restore reverted relations
        remove_wiki:

    Returns:

    """
    model = _get_model(dereify)
    out = load_(source=source, model=model)
    if remove_wiki:
        for i in range(len(out)):
            out[i] = _remove_wiki(out[i])
    return out


def loads(string, dereify=None, remove_wiki=False):
    model = _get_model(dereify)
    out = loads_(string=string, model=model)
    if remove_wiki:
        for i in range(len(out)):
            out[i] = _remove_wiki(out[i])
    return out


def pm_encode(g, top=None, indent=-1, compact=False):
    model = amr_model
    return encode_(g=g, top=top, indent=indent, compact=compact, model=model)


def role_is_reverted(role: str):
    if role.endswith('consist-of'):
        return False
    return role.endswith('-of')


class AMRGraph(penman.Graph):
    def __str__(self):
        return penman.encode(self)
