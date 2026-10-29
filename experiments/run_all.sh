#!/usr/bin/env bash
# Runs every experiment of Chapter 5 (needs OpenFHE).
cd "$(dirname "$0")"
${PYTHON:-python3} -u experiments.py all > run_all.log 2>&1
