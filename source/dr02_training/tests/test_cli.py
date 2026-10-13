from types import SimpleNamespace

from dr02_training.cli import _physics_preset, _requires_early_kit


def test_explicit_kit_physx_preset_requires_early_kit():
    request = SimpleNamespace(hydra_args=("physics=isaacsim_physx",))

    assert _requires_early_kit(request)


def test_newton_and_unrelated_overrides_do_not_require_early_kit():
    request = SimpleNamespace(hydra_args=("physics=newton_mjwarp", "terrain=flat"))

    assert not _requires_early_kit(request)


def test_default_physics_is_not_treated_as_an_explicit_preset():
    request = SimpleNamespace(hydra_args=("terrain=flat",))

    assert _physics_preset(request) is None
