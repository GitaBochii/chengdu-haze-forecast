"""Split the inventory into two independently readable CSV files."""
import csv
import hashlib
import io
from pathlib import Path


def main():
    directory = Path(__file__).resolve().parents[1] / 'docs' / 'project-overview-20261002'
    source = directory / 'project-files.csv'
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    with source.open(encoding='utf-8-sig', newline='') as handle:
        records = list(csv.reader(handle))
    header, rows = records[0], records[1:]

    def encode(row):
        buffer = io.StringIO(newline='')
        csv.writer(buffer, quoting=csv.QUOTE_ALL, lineterminator='\r\n').writerow(row)
        return buffer.getvalue().encode('utf-8')

    prefix = b'\xef\xbb\xbf' + encode(header)
    encoded = [encode(row) for row in rows]
    target = sum(map(len, encoded)) // 2
    size, boundary = 0, 0
    for row in encoded:
        if size + len(row) > target:
            break
        size += len(row)
        boundary += 1
    groups = (encoded[:boundary], encoded[boundary:])
    assert all(len(prefix) + sum(map(len, group)) < 20_000_000 for group in groups)
    destinations = [directory / f'project-files-part-{index}.csv' for index in (1, 2)]
    if any(path.exists() for path in destinations):
        raise FileExistsError('Split files already exist; original and outputs left unchanged.')
    restored = []
    for path, group in zip(destinations, groups):
        with path.open('xb') as handle:
            handle.write(prefix)
            handle.writelines(group)
        with path.open(encoding='utf-8-sig', newline='') as handle:
            reader = csv.reader(handle)
            assert next(reader) == header
            restored.extend(reader)
        print(f'{path.name}: {path.stat().st_size} bytes, {len(group)} records')
    assert restored == rows, 'Record order or content mismatch'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    print(f'Verified all {len(rows)} records; original file unchanged.')


if __name__ == '__main__':
    main()
