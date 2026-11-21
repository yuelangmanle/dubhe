import os
from pathlib import Path
import subprocess
import tempfile
import textwrap
import unittest

TOOL = Path(__file__).resolve().parents[1] / "tools/quality_android.sh"


class AndroidQualityToolTests(unittest.TestCase):
    def test_android_test_failures_are_not_treated_as_successful_adb_commands(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            adb = root / "fake-adb"
            log = root / "calls"
            adb.write_text(textwrap.dedent("""\
                #!/bin/bash
                test "$1" = -s && test "$2" = emulator-5554 || exit 9
                shift 2
                printf '%s\n' "$*" >> "$ACTIONS"
                case "$*" in
                  get-state) echo device ;;
                  "shell getprop ro.product.cpu.abilist") echo arm64-v8a,armeabi-v7a ;;
                  "install -r "*) echo Success ;;
                  "shell am force-stop "*) ;;
                  "shell am start "*) echo "Status: ok" ;;
                  "shell am instrument "*)
                    if [[ "$FAIL_TESTS" = yes ]]; then echo 'FAILURES!!!'; else echo 'OK (3 tests)'; fi
                    ;;
                  *) exit 8 ;;
                esac
                """), encoding="utf-8")
            adb.chmod(0o700)
            app, tests = root / "app.apk", root / "tests.apk"
            for path in (app, tests):
                path.touch()
            command = ["bash", str(TOOL), "emulator-5554", "test", str(app), str(tests)]
            env = {**os.environ, "ADB": str(adb), "ACTIONS": str(log)}
            for failed in ("no", "yes"):
                run = subprocess.run(command, env={**env, "FAIL_TESTS": failed}, text=True, capture_output=True)
                self.assertEqual(run.returncode, int(failed == "yes"), run.stdout + run.stderr)
            self.assertIn(f"install -r {app}", log.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
