#!/usr/bin/env python3
"""enrich_parser.py — 转发垫片。真身已搬到 bots/enrich/enrich_parser.py。

为什么留这个: 根目录还有 20 个回测/研究脚本 (bt_*.py / backtest_enrich.py /
mirror_* / proxy_* / build_interp_audit.py / signal_history.py) 写着
`from enrich_parser import parse_signal`。它们是一次性研究脚本, 挨个改 import
风险大于收益, 所以这里做转发。

⚠ 不能写成 `from enrich_parser import *` —— 本文件自己就叫 enrich_parser,
   那样会 import 自己造成空模块。必须按【文件路径】显式加载真身。

新代码请直接用 bots/enrich/ 下的真身, 不要依赖这个垫片。
"""
import importlib.util as _ilu
import sys as _sys
from pathlib import Path as _Path

_impl_path = _Path(__file__).resolve().parent / "bots" / "enrich" / "enrich_parser.py"
if not _impl_path.exists():
    raise ImportError(f"enrich_parser 真身不存在: {_impl_path} (是不是又搬了目录?)")

_spec = _ilu.spec_from_file_location("_enrich_parser_impl", _impl_path)
_impl = _ilu.module_from_spec(_spec)
_sys.modules["_enrich_parser_impl"] = _impl
_spec.loader.exec_module(_impl)

globals().update({k: v for k, v in vars(_impl).items() if not k.startswith("_")})
