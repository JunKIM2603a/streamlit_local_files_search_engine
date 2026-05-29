# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata


datas = [
    ("sdiu_command_finetune/app.py", "sdiu_command_finetune"),
    ("sdiu_command_finetune/data/commands.json", "sdiu_command_finetune/data"),
    ("sdiu_command_finetune/data/sdiu_sft.jsonl", "sdiu_command_finetune/data"),
    ("sdiu_command_finetune/data/records.json", "sdiu_command_finetune/data"),
]
datas += collect_data_files("streamlit")

for package_name in (
    "streamlit",
    "altair",
    "blinker",
    "cachetools",
    "click",
    "gitpython",
    "jinja2",
    "jsonschema",
    "numpy",
    "packaging",
    "pandas",
    "pillow",
    "protobuf",
    "pyarrow",
    "pydeck",
    "requests",
    "rich",
    "tenacity",
    "toml",
    "tornado",
    "typing_extensions",
    "watchdog",
):
    try:
        datas += copy_metadata(package_name)
    except Exception:
        pass

excluded_modules = [
    "accelerate",
    "datasets",
    "huggingface_hub",
    "peft",
    "safetensors",
    "sentencepiece",
    "torch",
    "torchaudio",
    "torchvision",
    "transformers",
    "trl",
]

a = Analysis(
    ["sdiu_command_finetune/launcher.py"],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=(
        collect_submodules("streamlit", on_error="ignore")
        + collect_submodules("sdiu_command_finetune", on_error="ignore")
    ),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excluded_modules,
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="sdiu-command-finder",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
