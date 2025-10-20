__version__ = 'v0.4.1'
from .flash_hlo import register_custom_calls
from .flash import flash_mha
from .varlen import flash_mha_varlen

register_custom_calls()

__all__ = ['flash_mha', 'flash_mha_varlen']
