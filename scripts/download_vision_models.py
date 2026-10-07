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
ZOO_REV = '47534e27c9851bb1128ccc0102f1145e27f23f98'
ASSETS = {
    'fonts': [['NotoSans-Regular.ttf',
  'https://raw.githubusercontent.com/notofonts/noto-fonts/ffebf8c1ee449e544955a7e813c54f9b73848eac/hinted/ttf/NotoSans/NotoSans-Regular.ttf',
  'b85c38ecea8a7cfb39c24e395a4007474fa5a4fc864f6ee33309eb4948d232d5'],
 ['NotoSansArabic-Regular.ttf',
  'https://raw.githubusercontent.com/notofonts/noto-fonts/ffebf8c1ee449e544955a7e813c54f9b73848eac/hinted/ttf/NotoSansArabic/NotoSansArabic-Regular.ttf',
  'ceea25b464a656dc3b26849bab9356740401af62aedf1bfa8b7f0d9b75925b1b'],
 ['NotoSansDevanagari-Regular.ttf',
  'https://raw.githubusercontent.com/notofonts/noto-fonts/ffebf8c1ee449e544955a7e813c54f9b73848eac/hinted/ttf/NotoSansDevanagari/NotoSansDevanagari-Regular.ttf',
  '385e78e6359a9d88a0f243d53b1209d7548361ba2194e2b9ec779bcaa7e8949d'],
 ['noto-fonts-LICENSE.txt',
  'https://raw.githubusercontent.com/notofonts/noto-fonts/ffebf8c1ee449e544955a7e813c54f9b73848eac/LICENSE',
  '0dab92d0544f7b233403f14b84a663bdbfa746982eda629e7f4f9ffe1b036feb'],
 ['NotoSansCJKjp-Regular.otf',
  'https://raw.githubusercontent.com/notofonts/noto-cjk/f8d157532fbfaeda587e826d4cd5b21a49186f7c/Sans/OTF/Japanese/NotoSansCJKjp-Regular.otf',
  '68a3fc98800b2a27b371f2fb79991daf3633bd89309d4ffaa6946fd587f375b5'],
 ['noto-cjk-LICENSE.txt',
  'https://raw.githubusercontent.com/notofonts/noto-cjk/f8d157532fbfaeda587e826d4cd5b21a49186f7c/Sans/LICENSE',
  '6a73f9541c2de74158c0e7cf6b0a58ef774f5a780bf191f2d7ec9cc53efe2bf2']],
    'ocr': [
        ('text_detection_en_ppocrv3_2023may.onnx',
         f'https://media.githubusercontent.com/media/opencv/opencv_zoo/{ZOO_REV}/models/text_detection_ppocr/text_detection_en_ppocrv3_2023may.onnx',
         '03f550c6b406fda8bf54bd8327815f6c7e2edd98cea02348c93d879254366587'),
        ('text_recognition_CRNN_CH_2021sep.onnx',
         f'https://media.githubusercontent.com/media/opencv/opencv_zoo/{ZOO_REV}/models/text_recognition_crnn/text_recognition_CRNN_CH_2021sep.onnx',
         '2dc566fd01ac2118b25c6960508ebd758b64c421a2bfa78dc05401ada6737e0b'),
    ],
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
