#!/bin/bash
cd /Users/xin/Documents/Claude/Projects/money/quant_system
python3 trade_report.py > /dev/null 2>&1
open output/trade_report.html
