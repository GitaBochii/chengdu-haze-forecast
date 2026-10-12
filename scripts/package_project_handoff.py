"""Create and verify a portable project snapshot without environments or caches."""

import hashlib
import argparse
import json
import os
from pathlib import Path
import subprocess
from datetime import datetime
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {
    '.git', '.venv', '.venv_air_models', '.venv_web', '.air_pip_cache', '.ruff_cache',
    '__pycache__', '.workbuddy', '_lo_profile', 'node_modules', 'packages',
    '.agents', '.codex',
}


def excluded_directory(name):
    return name in SKIP_DIRS or name.startswith(('.pytest', 'test_tmp'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compression-level', type=int, choices=range(10), default=6)
    args = parser.parse_args()
    output = ROOT / 'packages'
    output.mkdir(exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    destination = output / f'chengdu-haze-project-{stamp}.zip'
    files = []
    excluded = []
    for directory, dirs, names in os.walk(ROOT, followlinks=False):
        base = Path(directory)
        for name in list(dirs):
            path = base / name
            if excluded_directory(name) or path.is_symlink():
                dirs.remove(name)
                excluded.append(path.relative_to(ROOT).as_posix() + '/')
        for name in sorted(names):
            path = base / name
            relative = path.relative_to(ROOT).as_posix()
            if (path.is_symlink() or name.endswith(('.pyc', '.pyo', '.pem', '.key'))
                    or name == '.env' or name.startswith('.env.')
                    or name.lower() in {'secrets.toml', 'credentials.json'}):
                excluded.append(relative)
                continue
            files.append((relative, path))
    files.sort()
    print(f'Packing {len(files)} files into {destination}', flush=True)
    manifest = []
    prefix = 'chengdu-haze-project/'
    with zipfile.ZipFile(destination, 'x', zipfile.ZIP_DEFLATED,
                         compresslevel=args.compression_level, allowZip64=True) as archive:
        for index, (relative, path) in enumerate(files):
            before = path.stat()
            digest = hashlib.sha256()
            with path.open('rb') as source, archive.open(prefix + relative, 'w', force_zip64=True) as target:
                while chunk := source.read(1024 * 1024):
                    digest.update(chunk)
                    target.write(chunk)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError(f'Source changed during packaging: {relative}')
            manifest.append({'path': relative, 'bytes': before.st_size, 'sha256': digest.hexdigest()})
            if index % 250 == 0:
                print(f'Packed {index + 1}/{len(files)}', flush=True)
        for environment in ('.venv', '.venv_air_models'):
            python = ROOT / environment / 'Scripts' / 'python.exe'
            result = subprocess.run([str(python), '-m', 'pip', 'freeze'],
                                    capture_output=True, text=True, check=True)
            archive.writestr(prefix + f'PACKAGE_ENV_{environment.strip(".")}.txt', result.stdout)
        metadata = {'created_local': stamp, 'files': manifest, 'excluded': sorted(excluded)}
        archive.writestr(prefix + 'PACKAGE_MANIFEST.json', json.dumps(metadata, ensure_ascii=False, indent=2))
        archive.writestr(prefix + 'PACKAGE_README.txt',
            '成都雾霾恢复时间预测项目交接包\n\n'
            '包含源码、历史数据、模型检查点、实验结果、测试及任务文档。\n'
            '不含虚拟环境、缓存、测试临时目录和常见独立凭据文件；不是脱敏公开发布包。\n'
            '打开网页：先解压，双击根目录的 启动网页.cmd。\n'
            '新电脑先安装 Python 并运行 安装网页环境.cmd，详见 START_HERE.md。\n'
            '请先阅读 artifacts/structure_weather_results_20260905.md，\n'
            '再阅读 docs/tasks/ 下的任务二、任务三文档。\n'
            '主负责人负责恢复概率输出头，该结构尚未实现；当前部署未替换。\n'
            '恢复环境：使用 Python 3.11 至 3.13，在解压目录运行 pip install -e ".[dev]"。\n'
            '神经模型还需 PyTorch 等依赖，两个 PACKAGE_ENV 文件记录原机版本，\n'
            '其中本地路径及 CUDA 版本须按目标机器调整，不保证直接安装成功。\n'
            '历史报告含原机器绝对路径，换目录后须调整配置，不要覆盖既有实验输出。\n'
            'PACKAGE_MANIFEST.json 记录源文件 SHA256 与排除清单。\n')
    print('Verifying every archived file against SHA256 manifest...', flush=True)
    with zipfile.ZipFile(destination) as archive:
        for entry in manifest:
            digest = hashlib.sha256()
            with archive.open(prefix + entry['path']) as source:
                while chunk := source.read(1024 * 1024):
                    digest.update(chunk)
            if digest.hexdigest() != entry['sha256']:
                raise RuntimeError(f'Archive mismatch: {entry["path"]}')
        for name in archive.namelist():
            if name.startswith(prefix + 'PACKAGE_'):
                archive.read(name)
    digest = hashlib.sha256()
    with destination.open('rb') as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    print(json.dumps({'archive': str(destination), 'files': len(manifest),
                      'bytes': destination.stat().st_size, 'sha256': digest.hexdigest(),
                      'verification': 'PASS'}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
