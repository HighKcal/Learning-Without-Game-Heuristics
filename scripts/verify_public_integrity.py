"""공개 allowlist·Git 목록·manifest·provenance를 읽기 전용으로 검증한다."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess

ROOT = Path(__file__).resolve().parents[1]
CONTROL = {"PUBLIC_MANIFEST.json", "PUBLIC_ALLOWLIST.txt", "CODE_PROVENANCE.json"}


def indexed(rows):
    result = {row["path"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError("중복 manifest/provenance 경로")
    return result


def verify():
    paths = (ROOT / "PUBLIC_ALLOWLIST.txt").read_text(encoding="utf-8").splitlines()
    allowed = set(paths)
    if len(allowed) != len(paths):
        raise ValueError("중복 allowlist 경로")
    for name in paths:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name or not name:
            raise ValueError(f"잘못된 공개 상대 경로: {name}")
        if not (ROOT / name).resolve().is_relative_to(ROOT):
            raise ValueError(f"checkout 밖의 경로: {name}")
    tracked = set(subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=ROOT
    ).decode("utf-8").rstrip("\0").split("\0"))
    if tracked != allowed:
        raise ValueError(f"Git/allowlist 불일치: {sorted(tracked ^ allowed)}")
    manifest = json.loads((ROOT / "PUBLIC_MANIFEST.json").read_text(encoding="utf-8"))
    entries = indexed(manifest["entries"])
    if manifest["total_files"] != len(allowed) or set(entries) != allowed - {"PUBLIC_MANIFEST.json"}:
        raise ValueError("manifest 공개 목록 불일치")
    if manifest["self_hash_excluded"] is not True:
        raise ValueError("manifest 자기참조 제외 계약 불일치")
    digests = {}
    for name, entry in entries.items():
        raw = (ROOT / name).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if entry["bytes"] != len(raw) or entry["sha256"] != digest:
            raise ValueError(f"파일 bytes/SHA256 불일치: {name}")
        if "candidate_sha256" in entry and entry["byte_identical_to_candidate"] != (digest == entry["candidate_sha256"]):
            raise ValueError(f"candidate 동일성 표시 불일치: {name}")
        digests[name] = digest
    provenance = indexed(json.loads((ROOT / "CODE_PROVENANCE.json").read_text(encoding="utf-8"))["records"])
    if set(provenance) != allowed - CONTROL:
        raise ValueError("provenance 공개 목록 불일치")
    for name, entry in provenance.items():
        if entry["sha256"] != digests[name]:
            raise ValueError(f"provenance SHA256 불일치: {name}")
    print(json.dumps({"status": "PASS", "public_files": len(allowed),
                      "hashed_files": len(entries), "provenance_records": len(provenance)}))


if __name__ == "__main__":
    verify()
