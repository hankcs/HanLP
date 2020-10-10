# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-08-26 15:02
import glob
import json
import os
import pickle
import platform
import random
import shutil
import sys
from sys import exit
from contextlib import contextmanager
import tempfile
import time
import urllib
import zipfile
import tarfile
from typing import Dict, Tuple, Optional
from urllib.parse import urlparse
from urllib.request import urlretrieve
from pathlib import Path
import numpy as np
from pkg_resources import parse_version
from hanlp.utils import time_util
from hanlp.utils.log_util import logger
from hanlp.utils.string_util import split_long_sentence_into
from hanlp.utils.time_util import now_filename
from hanlp.common.constant import HANLP_URL
from hanlp.version import __version__


def save_pickle(item, path):
    with open(path, 'wb') as f:
        pickle.dump(item, f)


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_json(item: dict, path: str, ensure_ascii=False, cls=None, default=lambda o: repr(o)):
    with open(path, 'w', encoding='utf-8') as out:
        json.dump(item, out, ensure_ascii=ensure_ascii, indent=2, cls=cls, default=default)


def load_json(path):
    with open(path, encoding='utf-8') as src:
        return json.load(src)


def filename_is_json(filename):
    filename, file_extension = os.path.splitext(filename)
    return file_extension in ['.json', '.jsonl']


def make_debug_corpus(path, delimiter=None, percentage=0.1, max_samples=100):
    files = []
    if os.path.isfile(path):
        files.append(path)
    elif os.path.isdir(path):
        files += [os.path.join(path, f) for f in os.listdir(path) if
                  os.path.isfile(os.path.join(path, f)) and '.debug' not in f and not f.startswith('.')]
    else:
        raise FileNotFoundError(path)
    for filepath in files:
        filename, file_extension = os.path.splitext(filepath)
        if not delimiter:
            if file_extension in {'.tsv', '.conll', '.conllx', '.conllu'}:
                delimiter = '\n\n'
            else:
                delimiter = '\n'
        with open(filepath, encoding='utf-8') as src, open(filename + '.debug' + file_extension, 'w',
                                                           encoding='utf-8') as out:
            samples = src.read().strip().split(delimiter)
            max_samples = min(max_samples, int(len(samples) * percentage))
            out.write(delimiter.join(samples[:max_samples]))


def path_join(path, *paths):
    return os.path.join(path, *paths)


def makedirs(path):
    os.makedirs(path, exist_ok=True)
    return path


def tempdir(name=None):
    path = tempfile.gettempdir()
    if name:
        path = makedirs(path_join(path, name))
    return path


def tempdir_human():
    return tempdir(now_filename())


class NumpyEncoder(json.JSONEncoder):
    """
    Special json encoder for numpy types
    See https://interviewbubble.com/typeerror-object-of-type-float32-is-not-json-serializable/
    """

    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):  #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def hanlp_home_default():
    """

    :return: default data directory depending on the platform and environment variables
    """
    if windows():
        return os.path.join(os.environ.get('APPDATA'), 'hanlp')
    else:
        return os.path.join(os.path.expanduser("~"), '.hanlp')


def windows():
    system = platform.system()
    return system == 'Windows'


def hanlp_home():
    """

    :return: data directory in the filesystem for storage, for example when downloading models
    """
    return os.getenv('HANLP_HOME', hanlp_home_default())


def file_exist(filename) -> bool:
    return os.path.isfile(filename)


def remove_file(filename):
    if file_exist(filename):
        os.remove(filename)


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def parent_dir(path):
    return os.path.normpath(os.path.join(path, os.pardir))


def download(url, save_path=None, save_dir=hanlp_home(), prefix=HANLP_URL, append_location=True):
    if not save_path:
        save_path = path_from_url(url, save_dir, prefix, append_location)
    if os.path.isfile(save_path):
        eprint('Using local {}, ignore {}'.format(save_path, url))
        return save_path
    else:
        makedirs(parent_dir(save_path))
        eprint('Downloading {} to {}'.format(url, save_path))
        tmp_path = '{}.downloading'.format(save_path)
        remove_file(tmp_path)
        try:
            def reporthook(count, block_size, total_size):
                global start_time, progress_size
                if count == 0:
                    start_time = time.time()
                    progress_size = 0
                    return
                duration = time.time() - start_time
                duration = max(1e-8, duration)
                progress_size = int(count * block_size)
                if progress_size > total_size:
                    progress_size = total_size
                speed = int(progress_size / duration)
                ratio = progress_size / total_size
                ratio = max(1e-8, ratio)
                percent = ratio * 100
                eta = duration / ratio * (1 - ratio)
                speed = human_bytes(speed)
                progress_size = human_bytes(progress_size)
                sys.stderr.write("\r%.2f%%, %s/%s, %s/s, ETA %s      " %
                                 (percent, progress_size, human_bytes(total_size), speed,
                                  time_util.report_time_delta(eta)))
                sys.stderr.flush()

            import socket
            socket.setdefaulttimeout(10)
            opener = urllib.request.build_opener()
            opener.addheaders = [('User-agent', f'HanLP/{__version__}')]
            urllib.request.install_opener(opener)
            urlretrieve(url, tmp_path, reporthook)
            eprint()
        except BaseException as e:
            remove_file(tmp_path)
            hints_for_download = ''
            url = url.split('#')[0]
            if not windows():
                hints_for_download = f'e.g. \nwget {url} -O {save_path}\n'
            else:
                hints_for_download = ' Use some decent downloading tools.\n'
            if not url.startswith(HANLP_URL):
                hints_for_download += 'For third party data, you may find it on our mirror site:\n' \
                                      'https://od.hankcs.com/hanlp/data/\n'
            installed_version, latest_version = check_outdated()
            if installed_version != latest_version:
                hints_for_download += f'Or upgrade to the latest version({latest_version}):\npip install -U hanlp'
            eprint(f'Download failed due to {repr(e)}. Please download it to {save_path} by yourself. '
                   f'{hints_for_download}')
            exit(1)
        remove_file(save_path)
        os.rename(tmp_path, save_path)
    return save_path


def parse_url_path(url):
    parsed: urllib.parse.ParseResult = urlparse(url)
    path = os.path.join(*parsed.path.strip('/').split('/'))
    return parsed.netloc, path


def uncompress(path, dest=None, remove=True):
    """
    uncompress a file

    Parameters
    ----------
    path
        The path to a compressed file
    dest
        The dest folder
    remove
        Remove compressed file after unzipping
    Returns
    -------
        The folder which contains the unzipped content if the zip contains multiple files,
        otherwise the path to the unique file
    """
    # assert path.endswith('.zip')
    prefix, ext = os.path.splitext(path)
    folder_name = os.path.basename(prefix)
    file_is_zip = ext == '.zip'
    root_of_folder = None
    with zipfile.ZipFile(path, "r") if ext == '.zip' else tarfile.open(path, 'r:*') as archive:
        try:
            if not dest:
                namelist = sorted(archive.namelist() if file_is_zip else archive.getnames())
                root_of_folder = namelist[0].strip('/') if len(
                    namelist) > 1 else ''  # only one file, root_of_folder = ''
                if all(f.split('/')[0] == root_of_folder for f in namelist[1:]) or not root_of_folder:
                    dest = os.path.dirname(path)  # only one folder, unzip to the same dir
                else:
                    root_of_folder = None
                    dest = prefix  # assume zip contains more than one files or folders
            eprint('Extracting {} to {}'.format(path, dest))
            archive.extractall(dest)
            if root_of_folder:
                if root_of_folder != folder_name:
                    # move root to match folder name
                    os.rename(path_join(dest, root_of_folder), path_join(dest, folder_name))
                dest = path_join(dest, folder_name)
            elif len(namelist) == 1:
                dest = path_join(dest, namelist[0])
        except (RuntimeError, KeyboardInterrupt) as e:
            remove = False
            if os.path.exists(dest):
                if os.path.isfile(dest):
                    os.remove(dest)
                else:
                    shutil.rmtree(dest)
            raise e
    if remove:
        remove_file(path)
    return dest


def split_if_compressed(path: str, compressed_ext=('.zip', '.tgz', '.gz', 'bz2', '.xz')) -> Tuple[str, Optional[str]]:
    root, ext = os.path.splitext(path)
    if ext in compressed_ext:
        return root, ext
    return path, None


def get_resource(path: str, save_dir=None, extract=True, prefix=HANLP_URL, append_location=True):
    """
    Fetch real path for a resource (model, corpus, whatever)
    :param path: the general path (can be a url or a real path)
    :param extract: whether to unzip it if it's a zip file
    :param save_dir:
    :return: the real path to the resource
    """
    anchor: str = None
    compressed = None
    if os.path.isdir(path):
        return path
    elif os.path.isfile(path):
        pass
    elif path.startswith('http:') or path.startswith('https:'):
        url = path
        if '#' in url:
            url, anchor = url.split('#', maxsplit=1)
        realpath = path_from_url(path, save_dir, prefix, append_location)
        realpath, compressed = split_if_compressed(realpath)
        # check if resource is there
        if anchor:
            if anchor.startswith('/'):
                # indicates the folder name has to be polished
                anchor = anchor.lstrip('/')
                parts = anchor.split('/')
                realpath = str(Path(realpath).parent.joinpath(parts[0]))
                anchor = '/'.join(parts[1:])
            child = path_join(realpath, anchor)
            if os.path.exists(child):
                return child
        elif os.path.isdir(realpath) or (os.path.isfile(realpath) and (compressed and extract)):
            return realpath
        else:
            pattern = realpath + '*'
            files = glob.glob(pattern)
            zip_path = realpath + compressed
            if extract and zip_path in files:
                files.remove(zip_path)
            if files:
                if len(files) > 1:
                    logger.debug(f'Found multiple files with {pattern}, will use the first one.')
                return files[0]
        # realpath is where its path after exaction
        if compressed:
            realpath += compressed
        if not os.path.isfile(realpath):
            path = download(url=path, save_path=realpath)
        else:
            path = realpath
    if extract and compressed:
        path = uncompress(path)
        if anchor:
            path = path_join(path, anchor)

    return path


def path_from_url(url, save_dir=hanlp_home(), prefix=HANLP_URL, append_location=True):
    if not save_dir:
        save_dir = hanlp_home()
    domain, relative_path = parse_url_path(url)
    if append_location:
        if not url.startswith(prefix):
            save_dir = os.path.join(save_dir, 'thirdparty', domain)
        else:
            # remove the relative path in prefix
            middle = prefix.split(domain)[-1].lstrip('/')
            if relative_path.startswith(middle):
                relative_path = relative_path[len(middle):]
        realpath = os.path.join(save_dir, relative_path)
    else:
        realpath = os.path.join(save_dir, os.path.basename(relative_path))
    return realpath


def human_bytes(file_size: int) -> str:
    file_size /= 1024  # KB
    if file_size > 1024:
        file_size /= 1024  # MB
        if file_size > 1024:
            file_size /= 1024  # GB
            return '%.1f GB' % file_size
        return '%.1f MB' % file_size
    return '%d KB' % file_size


def read_cells(filepath: str, delimiter='auto', strip=True, skip_header=False):
    filepath = get_resource(filepath)
    if delimiter == 'auto':
        if filepath.endswith('.tsv'):
            delimiter = '\t'
        elif filepath.endswith('.csv'):
            delimiter = ','
        else:
            delimiter = None
    with open(filepath, encoding='utf-8') as src:
        if skip_header:
            next(src)
        for line in src:
            line = line.strip()
            if not line:
                continue
            cells = line.split(delimiter)
            if strip:
                cells = [c.strip() for c in cells]
                yield cells


def replace_ext(filepath, ext) -> str:
    """
    Replace the extension of filepath to ext
    Parameters
    ----------
    filepath
    ext

    Returns
    -------

    """
    file_prefix, _ = os.path.splitext(filepath)
    return file_prefix + ext


def load_word2vec(path, delimiter=' ', cache=True) -> Tuple[Dict[str, np.ndarray], int]:
    realpath = get_resource(path)
    binpath = replace_ext(realpath, '.pkl')
    if cache:
        try:
            word2vec, dim = load_pickle(binpath)
            logger.debug(f'Loaded {binpath}')
            return word2vec, dim
        except IOError:
            pass

    dim = None
    word2vec = dict()
    with open(realpath, encoding='utf-8', errors='ignore') as f:
        for idx, line in enumerate(f):
            line = line.rstrip().split(delimiter)
            if len(line) > 2:
                if dim is None:
                    dim = len(line)
                else:
                    if len(line) != dim:
                        logger.warning('{}#{} length mismatches with {}'.format(path, idx + 1, dim))
                        continue
                word, vec = line[0], line[1:]
                word2vec[word] = np.array(vec, dtype=np.float32)
    dim -= 1
    if cache:
        save_pickle((word2vec, dim), binpath)
        logger.debug(f'Cached {binpath}')
    return word2vec, dim


def save_word2vec(word2vec: dict, filepath, delimiter=' '):
    with open(filepath, 'w', encoding='utf-8') as out:
        for w, v in word2vec.items():
            out.write(f'{w}{delimiter}')
            out.write(f'{delimiter.join(str(x) for x in v)}\n')


def read_tsv(tsv_file_path):
    sent = []
    tsv_file_path = get_resource(tsv_file_path)
    with open(tsv_file_path, encoding='utf-8') as tsv_file:
        for line in tsv_file:
            cells = line.strip().split()
            if cells:
                # if len(cells) != 2:
                #     print(line)
                sent.append(cells)
            else:
                yield sent
                sent = []
    if sent:
        yield sent


def generator_words_tags(tsv_file_path, lower=True, gold=True, max_seq_length=None):
    for sent in read_tsv(tsv_file_path):
        words = [cells[0] for cells in sent]
        if max_seq_length and len(words) > max_seq_length:
            offset = 0
            # try to split the sequence to make it fit into max_seq_length
            for shorter_words in split_long_sentence_into(words, max_seq_length):
                if gold:
                    shorter_tags = [cells[1] for cells in sent[offset:offset + len(shorter_words)]]
                    offset += len(shorter_words)
                else:
                    shorter_tags = None
                if lower:
                    shorter_words = [word.lower() for word in shorter_words]
                yield shorter_words, shorter_tags
        else:
            if gold:
                tags = [cells[1] for cells in sent]
            else:
                tags = None
            if lower:
                words = [word.lower() for word in words]
            yield words, tags


def split_file(filepath, train=0.8, valid=0.1, test=0.1, names=None, shuffle=False):
    num_lines = 0
    with open(filepath, encoding='utf-8') as src:
        for line in src:
            num_lines += 1
    splits = {'train': train, 'valid': valid, 'test': test}
    splits = dict((k, v) for k, v in splits.items() if v)
    splits = dict((k, v / sum(splits.values())) for k, v in splits.items())
    accumulated = 0
    r = []
    for k, v in splits.items():
        r.append(accumulated)
        accumulated += v
        r.append(accumulated)
        splits[k] = accumulated
    if names is None:
        names = {}
    name, ext = os.path.splitext(filepath)
    outs = [open(names.get(split, name + '.' + split + ext), 'w', encoding='utf-8') for split in splits.keys()]
    if shuffle:
        shuffle = list(range(num_lines))
        random.shuffle(shuffle)
    with open(filepath, encoding='utf-8') as src:
        for idx, line in enumerate(src):
            if shuffle:
                idx = shuffle[idx]
            ratio = idx / num_lines
            for sid, out in enumerate(outs):
                if r[2 * sid] <= ratio < r[2 * sid + 1]:
                    out.write(line)
                    break
    for out in outs:
        out.close()


def fileno(file_or_fd):
    try:
        fd = getattr(file_or_fd, 'fileno', lambda: file_or_fd)()
    except:
        return None
    if not isinstance(fd, int):
        raise ValueError("Expected a file (`.fileno()`) or a file descriptor")
    return fd


@contextmanager
def stdout_redirected(to=os.devnull, stdout=None):
    """
    Redirect stdout to else where
    Copied from https://stackoverflow.com/questions/4675728/redirect-stdout-to-a-file-in-python/22434262#22434262
    Parameters
    ----------
    to
    stdout
    """
    if windows():  # This doesn't play well with windows
        yield None
        return
    if stdout is None:
        stdout = sys.stdout
    stdout_fd = fileno(stdout)
    if not stdout_fd:
        yield None
        return
        # copy stdout_fd before it is overwritten
    # NOTE: `copied` is inheritable on Windows when duplicating a standard stream
    with os.fdopen(os.dup(stdout_fd), 'wb') as copied:
        stdout.flush()  # flush library buffers that dup2 knows nothing about
        try:
            os.dup2(fileno(to), stdout_fd)  # $ exec >&to
        except ValueError:  # filename
            with open(to, 'wb') as to_file:
                os.dup2(to_file.fileno(), stdout_fd)  # $ exec > to
        try:
            yield stdout  # allow code to be run with the redirected stdout
        finally:
            # restore stdout to its previous value
            # NOTE: dup2 makes stdout_fd inheritable unconditionally
            try:
                stdout.flush()
                os.dup2(copied.fileno(), stdout_fd)  # $ exec >&copied
            except:
                # This is the best we can do
                pass


def check_outdated(package='hanlp', version=__version__, repository_url='https://pypi.python.org/pypi/%s/json'):
    """
    Given the name of a package on PyPI and a version (both strings), checks
    if the given version is the latest version of the package available.
    Returns a 2-tuple (installed_version, latest_version)
    `repository_url` is a `%` style format string
    to use a different repository PyPI repository URL,
    e.g. test.pypi.org or a private repository.
    The string is formatted with the package name.
    Adopted from https://github.com/alexmojaki/outdated/blob/master/outdated/__init__.py
    """

    installed_version = parse_version(version)
    latest_version = get_latest_info_from_pypi(package, repository_url)
    return installed_version, latest_version


def get_latest_info_from_pypi(package='hanlp', repository_url='https://pypi.python.org/pypi/%s/json'):
    url = repository_url % package
    response = urllib.request.urlopen(url).read()
    return parse_version(json.loads(response)['info']['version'])
