import pytest
import imaging.mask_feather_edge as mod

CASES = [
    ("radius_by_resolution", (1,)),
    ("guided_feather", (1,)),
    ("cache_version", (1,)),
    ("wait_task", (1,)),
    ("normalize_overlay", (1,)),
    ("clamp_edge", (1,)),
    ("load", (1,)),
    ("feather", (1,)),
    ("composite", (1,)),
    ("save", (1,)),
    ("validate_mask", (1,)),
    ("metrics", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("determinism", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
