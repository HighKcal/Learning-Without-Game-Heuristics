"""Release Asset 9개를 검증하고 새 Registry artifact root에 배치한다."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from tetris_agent.experiments import Registry, relative_path


def digest(path):
    result = sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset-dir', type=Path, required=True,
                        help='신뢰 출처에서 받은 9개 .pt Asset 디렉터리')
    parser.add_argument('--artifact-root', type=Path, required=True,
                        help='아직 존재하지 않는 새 배치 디렉터리')
    parser.add_argument('--manifest', type=Path, default=ROOT / 'MODEL_RELEASE_MANIFEST.json')
    args = parser.parse_args(argv)
    if args.artifact_root.exists():
        parser.error('기존 artifact root에는 덮어쓰지 않는다. 새 경로를 지정한다.')
    if not args.asset_dir.is_dir() or args.asset_dir.is_symlink() or args.asset_dir.is_junction():
        parser.error('asset-dir은 실제 디렉터리여야 한다.')
    if args.artifact_root.is_symlink() or args.artifact_root.is_junction():
        parser.error('연결된 출력 경로를 사용하지 않는다.')
    data = json.loads(args.manifest.read_text(encoding='utf-8'))
    rows = data.get('checkpoints_metadata', [])
    if len(rows) != 9 or data.get('release_asset_count') != 9:
        parser.error('지원된 9개 Checkpoint manifest가 필요하다.')
    registry = Registry(ROOT, artifact_root=args.artifact_root)
    expected_supported = {
        (entry['id'], item['artifact_id'])
        for entry in registry.entries.values()
        for item in entry.get('checkpoints', [])
        if item.get('staging_compatibility') == 'supported'
    }
    keys = [(r['experiment_id'], r['artifact_id']) for r in rows]
    names = [r['release_asset_name'] for r in rows]
    if (len(set(keys)) != 9 or set(keys) != expected_supported
            or len(set(names)) != 9
            or {p.name for p in args.asset_dir.glob('*.pt')} != set(names)):
        parser.error('Asset 목록과 지원된 Registry 9개가 정확히 일치해야 한다.')
    plans = []
    # 모든 manifest/Registry/size/hash를 검사한 후 처음으로 파일을 쓴다.
    for row in rows:
        name = row['release_asset_name']
        if '/' in name or '\\' in name:
            parser.error('Asset 이름은 단일 파일명이어야 한다.')
        entry, item = registry.select(row['experiment_id'],
                                      step=row['checkpoint_decisions'], role=row['role'])
        rel = entry['current_path'] + '/' + item['filename']
        if (item['artifact_id'] != row['artifact_id'] or rel != row['relative_path']
                or item['sha256'] != row['sha256'] or item['bytes'] != row['bytes']):
            parser.error('Manifest와 Registry의 경로/역할/hash/size가 다르다.')
        source = relative_path(args.asset_dir, name)
        target = relative_path(args.artifact_root, rel)
        if (not source.is_file() or source.stat().st_size != item['bytes']
                or digest(source) != item['sha256']):
            parser.error('Asset 파일 size 또는 SHA256이 다르다: ' + name)
        plans.append((source, target, item['sha256']))
    args.artifact_root.mkdir(parents=True, exist_ok=False)
    for source, target, expected in plans:
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open('rb') as src, target.open('xb') as dst:
            shutil.copyfileobj(src, dst)
        if digest(target) != expected:
            raise RuntimeError('복사 후 SHA256 불일치: ' + target.name)
    print(json.dumps({'installed': len(plans), 'sha256_verified': True,
                      'registry_contract_changed': False}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
