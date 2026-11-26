"""Create a pre-training identity record for the source development phase."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    target=ROOT/'manifests/source_freeze.json'
    if target.exists():raise SystemExit('Freeze exists; do not silently replace it')
    paths=[ROOT/'protocol_v1.md',ROOT/'config/study.json',ROOT/'config/experiment_v1.json',
           ROOT/'requirements-lock.txt',ROOT/'manifests/participant_splits.csv',
           ROOT/'src/model_utils.py',ROOT/'src/train_source.py',ROOT/'src/audit_data.py']
    study=json.loads((ROOT/'config/study.json').read_text())
    for cycle in study['cycles']:
        if cycle['role'] in {'source_train','source_validation'}:
            paths.append(ROOT/f"data/processed/eligibility_{cycle['name']}.csv")
    data={'phase':'source development only','frozen_utc':datetime.now(timezone.utc).isoformat(),
          'external_preregistration':False,'novelty_clearance':'provisional application/replication',
          'target_test_predictions_at_freeze':False,
          'files':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    target.write_text(json.dumps(data,indent=2))
    print(target)


if __name__=='__main__':main()
