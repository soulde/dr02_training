import pytest
import torch

from dr02_training.amp_motion_compat import add_virtual_toes


def _legacy_motion(body_count=30):
    pos = torch.zeros((2, body_count, 3))
    quat_xyzw = torch.zeros((2, body_count, 4))
    quat_xyzw[..., 3] = 1.0
    quat_w = quat_xyzw[..., [3, 0, 1, 2]]
    return {
        "body_pos_w": pos,
        "body_quat_xyzw": quat_xyzw,
        "body_quat_w": quat_w,
        "body_lin_vel_w": torch.zeros_like(pos),
        "body_ang_vel_w": torch.zeros_like(pos),
    }


def test_add_virtual_toes_extends_legacy_dr02_motion_contract():
    body_names = [f"body_{index}" for index in range(30)]
    body_names[4] = "left_ankle_x_link"
    body_names[9] = "right_ankle_x_link"
    body_names += ["left_toe_link", "right_toe_link"]
    motion = _legacy_motion()
    motion["body_pos_w"][:, 4] = torch.tensor([1.0, 2.0, 3.0])
    motion["body_pos_w"][:, 9] = torch.tensor([-1.0, 0.0, 0.5])

    augmented = add_virtual_toes(motion, body_names)

    assert augmented["body_pos_w"].shape == (2, 32, 3)
    torch.testing.assert_close(augmented["body_pos_w"][:, 30], torch.tensor([[1.11, 2.0, 2.985]] * 2))
    torch.testing.assert_close(augmented["body_pos_w"][:, 31], torch.tensor([[-0.89, 0.0, 0.485]] * 2))
    assert augmented["body_quat_xyzw"].shape == (2, 32, 4)


def test_add_virtual_toes_rejects_unrecognized_partial_contract():
    body_names = [f"body_{index}" for index in range(31)] + ["left_toe_link"]

    with pytest.raises(ValueError, match="body contract mismatch"):
        add_virtual_toes(_legacy_motion(31), body_names)
