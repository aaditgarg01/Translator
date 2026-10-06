"""Download pinned OpenCV assets with TLS, integrity checks and atomic replacement."""

import argparse
import hashlib
from pathlib import Path
import ssl
import sys
import urllib.request

import certifi

ROOT = Path(__file__).resolve().parents[1]
LBF_REV = '7523caa8539ef58c5f4132bf99b13c617fbb58df'
ASSETS = {
    'speaker': [(
        'lbfmodel.yaml',
        f'https://raw.githubusercontent.com/kurnianggoro/GSOC2017/{LBF_REV}/data/lbfmodel.yaml',
        '70dd8b1657c42d1595d6bd13d97d932877b3bed54a95d3c4733a0f740d1fd66b',
    )],
}


def download(name, url, digest):
    target = ROOT / 'models' / 'vision' / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == digest:
        print(f'Already installed: {name}', flush=True)
        return
    part = target.with_suffix(target.suffix + '.part')
    try:
        print(f'Downloading {name}...', flush=True)
        context = ssl.create_default_context(cafile=certifi.where())
        request = urllib.request.Request(url, headers={'User-Agent': 'ESA-Translator'})
        sha = hashlib.sha256()
        with urllib.request.urlopen(request, context=context, timeout=60) as response, part.open('wb') as output:
            while block := response.read(1024 * 1024):
                sha.update(block)
                output.write(block)
        if sha.hexdigest() != digest:
            raise RuntimeError(f'Checksum mismatch for {name}; original file kept')
        part.replace(target)
        print(f'Installed {target}', flush=True)
    finally:
        part.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('feature', choices=[*ASSETS, 'all'])
    args = parser.parse_args()
    groups = ASSETS.values() if args.feature == 'all' else [ASSETS[args.feature]]
    for group in groups:
        for asset in group:
            download(*asset)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'Vision setup failed: {exc}', file=sys.stderr)
        raise SystemExit(1)
