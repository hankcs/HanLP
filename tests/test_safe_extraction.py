import io
import os
import shutil
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from hanlp.utils.io_util import uncompress


class TestSafeExtraction(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def archive(self, members):
        path = self.root / 'model.tar.gz'
        with tarfile.open(str(path), 'w:gz') as archive:
            for member in members:
                if isinstance(member, str):
                    info = tarfile.TarInfo(member)
                    info.size = 7
                    archive.addfile(info, io.BytesIO(b'weights'))
                else:
                    archive.addfile(member)
        return str(path)

    def test_tar_layouts(self):
        for names, expected in ((['old/', 'old/weights'], 'model/weights'),
                                (['weights'], 'weights'),
                                (['a', 'b'], 'model/a')):
            with self.subTest(names=names):
                members = []
                for name in names:
                    if name.endswith('/'):
                        member = tarfile.TarInfo(name)
                        member.type = tarfile.DIRTYPE
                        member.mode = 0o755
                        members.append(member)
                    else:
                        members.append(name)
                path = self.archive(members)
                result = uncompress(path, verbose=False)
                self.assertTrue(os.path.exists(result))
                self.assertEqual((self.root / expected).read_bytes(), b'weights')
                # Keep each layout independent of earlier extracted files.
                for child in self.root.iterdir():
                    if child.is_dir():
                        shutil.rmtree(str(child))
                    else:
                        child.unlink()

    def test_tar_rejects_outside_paths_before_writing(self):
        dest = self.root / 'dest'
        for name in ('../escaped', str(self.root / 'absolute')):
            with self.subTest(name=name):
                path = self.archive(['safe', name])
                with self.assertRaises(ValueError):
                    uncompress(path, dest=str(dest), verbose=False)
                self.assertFalse((dest / 'safe').exists())
                self.assertFalse((self.root / 'escaped').exists())
                self.assertFalse((self.root / 'absolute').exists())

    def test_tar_preserves_internal_links(self):
        for kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
            with self.subTest(kind=kind):
                link = tarfile.TarInfo('alias')
                link.type = kind
                link.linkname = 'weights'
                dest = self.root / kind.decode('ascii')
                uncompress(self.archive(['weights', link]), dest=str(dest), verbose=False)
                self.assertEqual((dest / 'alias').read_bytes(), b'weights')

    def test_tar_rejects_external_links_and_special_files(self):
        for kind in (tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE, tarfile.CHRTYPE):
            member = tarfile.TarInfo('unsafe')
            member.type = kind
            member.linkname = '../outside'
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                uncompress(self.archive([member]), dest=str(self.root / 'dest'), verbose=False)

    def test_tar_rechecks_paths_after_symlink_creation(self):
        link = tarfile.TarInfo('link')
        link.type = tarfile.SYMTYPE
        link.linkname = '.'
        path = self.archive([link, 'link/../escaped'])
        if os.name == 'nt':
            # Windows normalizes '..' before following the symlink, so this
            # member safely resolves to dest/escaped rather than escaping.
            uncompress(path, dest=str(self.root / 'dest'), verbose=False)
        else:
            with self.assertRaises(ValueError):
                uncompress(path, dest=str(self.root / 'dest'), verbose=False)
        self.assertFalse((self.root / 'escaped').exists())

    def test_tar_rejects_existing_external_symlink(self):
        dest = self.root / 'dest'
        dest.mkdir()
        outside = self.root / 'outside'
        outside.mkdir()
        (dest / 'link').symlink_to(outside, target_is_directory=True)
        with self.assertRaises(ValueError):
            uncompress(self.archive(['link/escaped']), dest=str(dest), verbose=False)
        self.assertFalse((outside / 'escaped').exists())

    def test_tar_without_builtin_filter(self):
        # Exercise the compatibility path independently of tarfile's protection.
        original = tarfile.TarFile.extractall

        def old_extractall(archive, path, members):
            return original(archive, path, members=members)

        with patch('hanlp.utils.io_util.tarfile', SimpleNamespace(open=tarfile.open)):
            with patch.object(tarfile.TarFile, 'extractall', old_extractall):
                with self.assertRaises(ValueError):
                    uncompress(self.archive(['../escaped']), dest=str(self.root / 'dest'), verbose=False)
                uncompress(self.archive(['weights']), dest=str(self.root / 'valid'), verbose=False)
        self.assertEqual((self.root / 'valid/weights').read_bytes(), b'weights')

    def test_zip_still_extracts(self):
        path = self.root / 'model.zip'
        with zipfile.ZipFile(str(path), 'w') as archive:
            archive.writestr('weights', b'weights')
        self.assertEqual(Path(uncompress(str(path), verbose=False)).read_bytes(), b'weights')

if __name__ == '__main__':
    unittest.main()
