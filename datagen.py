#!/usr/bin/env python3

"""
Synthetic ML Dataset Generator Entry Point

Usage:
    python datagen.py
    python datagen.py --task classification --size medium --accept
"""

from datagen.main import app

if __name__ == "__main__":
    app()