"""Native overlay inventory and fail-closed hash checks without changing a runtime."""
import json
from pathlib import Path
import runpy
import shutil
import pytest
ROOT=Path(__file__).resolve().parents[1]
VERIFY=runpy.run_path(str(ROOT/'scripts/install_native.py'))['verify_tree']
SOURCE=ROOT/'vendor/native_tetris_031/tetris_gymnasium'
HASHES=json.loads((SOURCE.parent/'manifest.json').read_text())['source_hashes']

def test_preserved_native_inventory():
    assert len(HASHES)==19
    VERIFY(SOURCE,HASHES)

@pytest.mark.parametrize('fault',['changed_bytes','extra_source'])
def test_unknown_native_is_rejected_before_install(tmp_path,fault):
    dst=tmp_path/'native'
    shutil.copytree(SOURCE,dst)
    if fault=='changed_bytes':
        with (dst/'envs/tetris.py').open('ab') as stream: stream.write(b'\n# altered\n')
    else:
        (dst/'unexpected.py').write_text('')
    with pytest.raises(RuntimeError,match='mismatch'):
        VERIFY(dst,HASHES)
