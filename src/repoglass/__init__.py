from .config import Paths, Settings
from .index import Index
from .models import Chunk, Hit, RefreshReport, SearchMode, Symbol

# The only copy. pyproject reads this attribute rather than restating it, so a
# release cannot ship a number that disagrees with what `rpg --version` prints.
__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Index",
    "Settings",
    "Paths",
    "Symbol",
    "Chunk",
    "Hit",
    "RefreshReport",
    "SearchMode",
]
