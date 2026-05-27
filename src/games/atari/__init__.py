# src/games/atari/__init__.py
from .atari_adapter import AtariAdapter, AtariFallbackSim, make_atari_adapter, ATARI_SPECS

__all__ = ["AtariAdapter", "AtariFallbackSim", "make_atari_adapter", "ATARI_SPECS"]
