#!/usr/bin/env bash
# Prints the modulus chain for the (depth, slots) pairs used in the dissertation.
cd "$(dirname "$0")"
for c in '10 64' '9 4096' '11 64'; do ${PYTHON:-python3} inspect_moduli.py $c; done
