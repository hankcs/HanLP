import glob
from typing import List, Union, Iterable
from pathlib import Path
from .penman import pm_load as pm_load


def read_raw_amr_data(
        paths: List[Union[str, Path]],
        use_recategorization=False,
        dereify=True,
        remove_wiki=False,
):
    assert paths

    if not isinstance(paths, Iterable):
        paths = [paths]

    graphs = []
    for path_ in paths:
        for path in glob.glob(str(path_)):
            path = Path(path)
            assert path.exists(), f'{path} not exist'
            graphs.extend(pm_load(path, dereify=dereify, remove_wiki=remove_wiki))

    assert graphs, 'No graphs loaded'

    if use_recategorization:
        for g in graphs:
            metadata = g.metadata
            metadata['snt_orig'] = metadata['snt']
            tokens = eval(metadata['tokens'])
            metadata['snt'] = ' '.join(
                [t for t in tokens if not ((t.startswith('-L') or t.startswith('-R')) and t.endswith('-'))])

    return graphs
