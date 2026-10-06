import sys
sys.path.append('.')
from tts.piper_backend import PiperBackend
pb = PiperBackend()
text = "こんにちは"
pcm, rate = pb.synth(text, "Japanese")
print(f"Synthesized {len(pcm)} bytes at {rate} Hz")
