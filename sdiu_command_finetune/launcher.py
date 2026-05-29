from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys
from typing import Any


def _bundle_root() -> Path:
    return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1])).resolve()


def _app_path() -> Path:
    bundled_app = _bundle_root() / "sdiu_command_finetune" / "app.py"
    if bundled_app.exists():
        return bundled_app
    return Path(__file__).resolve().with_name("app.py")


def _run_module_main(module_name: str, args: list[str]) -> None:
    sys.argv = [sys.argv[0], *args]
    module = importlib.import_module(module_name)
    module.main()


def _dispatch_subcommand(argv: list[str]) -> bool:
    if not argv:
        return False

    commands = {
        "--build-dataset": "sdiu_command_finetune.build_dataset",
        "--infer": "sdiu_command_finetune.infer",
        "--train-lora": "sdiu_command_finetune.train_lora",
    }
    module_name = commands.get(argv[0])
    if not module_name:
        return False

    _run_module_main(module_name, argv[1:])
    return True


def _coerce_flag_value(value: str) -> Any:
    lowered = value.strip().lower()
    if lowered in {"true", "1", "yes", "on"}:
        return True
    if lowered in {"false", "0", "no", "off"}:
        return False
    try:
        return int(value)
    except ValueError:
        return value


def _parse_streamlit_args(argv: list[str]) -> tuple[dict[str, Any], list[str]]:
    flag_options: dict[str, Any] = {
        "global.developmentMode": False,
        "server.headless": False,
        "server.fileWatcherType": "none",
        "browser.gatherUsageStats": False,
    }
    script_args: list[str] = []
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg == "--":
            script_args.extend(argv[index + 1 :])
            break
        if arg.startswith("--") and "." in arg[2:]:
            key_value = arg[2:]
            if "=" in key_value:
                key, value = key_value.split("=", 1)
            elif index + 1 < len(argv) and not argv[index + 1].startswith("--"):
                key = key_value
                value = argv[index + 1]
                index += 1
            else:
                key = key_value
                value = "true"
            flag_options[key] = _coerce_flag_value(value)
        else:
            script_args.append(arg)
        index += 1
    return flag_options, script_args


def main() -> None:
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")

    if _dispatch_subcommand(sys.argv[1:]):
        return

    from streamlit import config as streamlit_config
    from streamlit.web import bootstrap

    flag_options, script_args = _parse_streamlit_args(sys.argv[1:])
    main_script_path = str(_app_path().resolve())
    streamlit_config._main_script_path = main_script_path
    bootstrap.load_config_options(flag_options=flag_options)
    bootstrap.run(main_script_path, False, script_args, flag_options)


if __name__ == "__main__":
    main()
