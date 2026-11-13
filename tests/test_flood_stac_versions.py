"""Updating a catalog record must not corrupt earlier observations sharing it."""
import hashlib
import tempfile
import unittest
from pathlib import Path

from scripts.download_flood_m2 import save_stac, saved_stac


class StacVersionTests(unittest.TestCase):
    def test_catalog_revisions_coexist_and_rebuild_reads_exact_revision(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination=Path(tmp)
            folder=destination/'scenes'/'same-scene'
            folder.mkdir(parents=True)
            first={'id':'same-scene','properties':{'revision':1}}
            second={'id':'same-scene','properties':{'revision':2}}
            old=save_stac(first,folder)
            old_bytes=old.read_bytes()
            new=save_stac(second,folder)
            self.assertNotEqual(old,new)
            self.assertEqual(old.read_bytes(),old_bytes)
            self.assertEqual((folder/'stac.json').read_bytes(),old_bytes)
            for path,expected in [(old,first),(new,second),(folder/'stac.json',first)]:
                scene={'files':{path.relative_to(destination).as_posix():hashlib.sha256(path.read_bytes()).hexdigest()}}
                self.assertEqual(saved_stac(destination,scene),expected)
            self.assertEqual(save_stac(first,folder),old)

    def test_corrupt_revision_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            path=save_stac({'id':'scene'},folder)
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            path.write_text('{}')
            with self.assertRaises(ValueError):
                saved_stac(folder,{'files':{path.name:digest}})
            with self.assertRaises(ValueError):
                save_stac({'id':'scene'},folder)


if __name__=='__main__':
    unittest.main()
