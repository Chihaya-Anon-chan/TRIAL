"""Small text utilities shared by TRIAL's rollout and training paths."""

from typing import Any, Optional

import numpy as np


def normalize_optional_text(value: Any) -> Optional[str]:
    """Return a stripped textual value, or ``None`` when no text is present."""
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        value = value.tolist() if value.ndim > 0 else value.item()
    if isinstance(value, (list, tuple)):
        value = "\n".join(str(item) for item in value if item is not None)
    else:
        value = str(value)
    value = value.strip()
    return value or None
