"""Run one explicit HTTPS receipt, never a scheduler or historical backfill."""

import argparse
import json
from pathlib import Path

from haze_forecast.weather_receipts import receive_once


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('url')
    parser.add_argument('--directory', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'artifacts/weather_receipts')
    parser.add_argument('--max-bytes', type=int, default=10_000_000)
    args = parser.parse_args()
    print(json.dumps(receive_once(args.directory, args.url, max_bytes=args.max_bytes), indent=2))


if __name__ == '__main__':
    main()
