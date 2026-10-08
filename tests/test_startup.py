import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import startup


class StartupTests(unittest.TestCase):
    def test_placeholder_is_rejected_without_reading_it(self):
        with patch.object(Path, 'stat', return_value=SimpleNamespace(st_flags=0x40000000)), \
                patch.object(Path, 'open', side_effect=AssertionError('must not read cloud file')):
            with self.assertRaisesRegex(RuntimeError, 'iCloud has offloaded'):
                startup.require_local('cv2/config.py')

    def test_nested_placeholder_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            child = root / 'config.py'
            child.write_text('')
            original = Path.stat

            def fake_stat(path, *args, **kwargs):
                if path == child:
                    return SimpleNamespace(st_flags=0x40000000)
                return original(path, *args, **kwargs)

            with patch.object(Path, 'stat', fake_stat):
                with self.assertRaisesRegex(RuntimeError, 'config.py'):
                    startup.require_local(root, recursive=True)

    def test_missing_files_are_left_to_normal_installation_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            startup.require_local(Path(directory) / 'missing')
            startup.require_local(directory, recursive=True)

    def test_platform_without_darwin_flags(self):
        with patch.object(Path, 'stat', return_value=SimpleNamespace(st_size=3)):
            startup.require_local('local-file')

    def test_windows_skips_icloud_runtime_scan(self):
        with patch.object(startup.sys, 'platform', 'win32'), \
                patch.object(startup, 'require_local') as check:
            startup.check_runtime()
            check.assert_not_called()

    def test_registration_help_does_not_load_native_or_subtitle_libraries(self):
        root = Path(__file__).resolve().parents[1]
        code = """
import runpy, sys
class BlockNative:
    def find_spec(self, name, *args):
        if name.split('.')[0] in {'cv2', 'PIL', 'uharfbuzz', 'freetype', 'numpy'}:
            raise AssertionError('Unexpected import: ' + name)
sys.meta_path.insert(0, BlockNative())
sys.argv = ['register.py', '--help']
runpy.run_path('register.py', run_name='__main__')
"""
        result = subprocess.run([sys.executable, '-c', code], cwd=root,
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--name', result.stdout)


if __name__ == '__main__':
    unittest.main()
