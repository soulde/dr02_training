"""Unit tests for AMP recovery sampling and assistance curriculum."""

from types import SimpleNamespace

import torch

from dr02_training.tasks.amp.mdp.recovery import (
    apply_recovery_assist_force,
    recovery_assist_scale,
    recovery_failed_after_window,
    recovery_root_height_below_minimum,
    sample_recovery_mask,
)


def _tensor(value: torch.Tensor) -> SimpleNamespace:
    return SimpleNamespace(torch=value)


def test_recovery_sampling_uses_configured_fraction():
    torch.manual_seed(7)
    recovery_mask = sample_recovery_mask(1000, probability=0.15, device="cpu")

    assert recovery_mask.dtype == torch.bool
    assert 120 <= recovery_mask.sum().item() <= 180


def test_assist_force_scale_anneals_linearly_to_zero():
    assert recovery_assist_scale(0, anneal_steps=100) == 1.0
    assert recovery_assist_scale(50, anneal_steps=100) == 0.5
    assert recovery_assist_scale(100, anneal_steps=100) == 0.0
    assert recovery_assist_scale(150, anneal_steps=100) == 0.0


def test_recovery_termination_waits_three_seconds_and_checks_standing():
    env = SimpleNamespace(
        _amp_recovery_mask=torch.tensor([True, True, True, False]),
        episode_length_buf=torch.tensor([31, 31, 29, 100]),
        step_dt=0.1,
        scene={
            "robot": SimpleNamespace(
                data=SimpleNamespace(
                    root_pos_w=_tensor(
                        torch.tensor([[0.0, 0.0, 0.6], [0.0, 0.0, 0.2], [0.0, 0.0, 0.6], [0.0, 0.0, 0.1]])
                    ),
                    projected_gravity_b=_tensor(
                        torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
                    ),
                )
            )
        },
    )

    failed = recovery_failed_after_window(env, window_s=3.0, minimum_height=0.48, max_tilt_angle=0.7)

    assert torch.equal(failed, torch.tensor([False, True, False, False]))


def test_assist_is_only_applied_to_recovery_envs():
    class WrenchRecorder:
        def set_forces_and_torques_index(self, **kwargs):
            self.kwargs = kwargs

    wrench_recorder = WrenchRecorder()
    env = SimpleNamespace(
        _amp_recovery_mask=torch.tensor([True, False, True]),
        common_step_counter=0,
        scene={
            "robot": SimpleNamespace(
                device="cpu",
                permanent_wrench_composer=wrench_recorder,
            )
        },
    )
    asset_cfg = SimpleNamespace(name="robot", body_ids=[0])

    apply_recovery_assist_force(env, torch.tensor([0, 1, 2]), asset_cfg, max_force=200.0, anneal_steps=100)

    upward_force = wrench_recorder.kwargs["forces"][:, 0, 2]
    assert torch.all(upward_force[[0, 2]] >= 0.0)
    assert torch.all(upward_force[[0, 2]] <= 200.0)
    assert upward_force[1].item() == 0.0
    assert wrench_recorder.kwargs["is_global"] is True


def test_low_root_termination_is_suppressed_only_during_recovery_window():
    robot = SimpleNamespace(data=SimpleNamespace(root_pos_w=_tensor(torch.tensor([[0.0, 0.0, 0.1], [0.0, 0.0, 0.1]]))))
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        _amp_recovery_mask=torch.tensor([True, False]),
        episode_length_buf=torch.tensor([10, 10]),
        step_dt=0.1,
        scene={"robot": robot},
    )

    during_grace = recovery_root_height_below_minimum(env, minimum_height=0.48)
    env.episode_length_buf[:] = 31
    after_grace = recovery_root_height_below_minimum(env, minimum_height=0.48)

    assert torch.equal(during_grace, torch.tensor([False, True]))
    assert torch.equal(after_grace, torch.tensor([True, True]))
