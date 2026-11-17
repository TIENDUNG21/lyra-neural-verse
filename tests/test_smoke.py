import lyra


def test_version_is_semver() -> None:
    parts = lyra.__version__.split(".")
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)
