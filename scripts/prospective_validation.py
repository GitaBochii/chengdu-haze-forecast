"""Isolated research ledger CLI. Never changes the website or starts a scheduler."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3

from haze_forecast.prospective_baseline import baseline_manifest
from haze_forecast.prospective_ledger import freeze_candidate, score_prediction

ROOT=Path(__file__).resolve().parents[1]


def status(root):
    result={'candidates':[], 'predictions':0, 'scores':0,
            'enhanced_candidate':'not_ready', 'pollution_input':'not_connected',
            'deployment_authorized':False, 'scheduler_started':False,
            'limitations':['No compatible GEFS onset/recovery classifier frozen',
                           'No verified current station-median 24-hour pollution input connected',
                           'Scores are retrospective after prospective predictions mature, not dissipation-time MAE']}
    path=Path(root)/'ledger.sqlite3'
    if not path.exists():
        return result
    with sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True) as db:
        for name,frozen,payload,expected in db.execute('SELECT name,frozen_at,manifest,sha256 FROM candidates'):
            if hashlib.sha256(payload.encode('utf-8')).hexdigest()!=expected:
                raise ValueError('Candidate manifest checksum mismatch')
            manifest=json.loads(payload)
            valid=all(Path(p).is_file() and hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
                      for p,h in manifest.get('files',{}).items())
            result['candidates'].append({'name':name,'frozen_at':frozen,'sha256':expected,
                                         'frozen_files_verified':valid,'kind':manifest.get('kind')})
        result['predictions']=db.execute('SELECT COUNT(*) FROM predictions').fetchone()[0]
        result['scores']=db.execute('SELECT COUNT(*) FROM scores').fetchone()[0]
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=ROOT/'artifacts/prospective_validation')
    sub=parser.add_subparsers(dest='command',required=True)
    freeze=sub.add_parser('freeze-baseline')
    freeze.add_argument('--name',default='pollution-only-v1')
    sub.add_parser('status')
    score=sub.add_parser('score')
    score.add_argument('prediction_id',type=int)
    score.add_argument('observations',type=Path,help='JSON list of 24 timestamp/AQI records; not an automatic truth collector')
    args=parser.parse_args()
    if args.command=='freeze-baseline':
        result=freeze_candidate(args.directory,args.name,baseline_manifest(ROOT))
    elif args.command=='score':
        result=score_prediction(args.directory,args.prediction_id,json.loads(args.observations.read_text('utf-8')))
    else:
        result=status(args.directory)
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
