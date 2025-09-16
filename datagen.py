#!/usr/bin/env python3

"""
Synthetic ML Dataset Generator Entry Point

Usage:
    python datagen.py
    python datagen.py --task classification --size medium --features 15 --domain ecommerce --accept
"""

from cli import app

if __name__ == "__main__":
    app()