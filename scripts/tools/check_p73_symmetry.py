#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from types import SimpleNamespace

import torch


REPO_ROOT = Path("/home/piene/p73/isaaclab_walker")
SYMMETRY_PY = REPO_ROOT / "source/isaaclab_walker/isaaclab_walker/tasks/manager_based/isaaclab_walker/mdp/symmetry.py"
ROUGH_CFG_PY = REPO_ROOT / "source/isaaclab_walker/isaaclab_walker/tasks/manager_based/isaaclab_walker/rough_env_cfg.py"


def _load_symmetry_module():
    spec = importlib.util.spec_from_file_location("p73_symmetry_debug", SYMMETRY_PY)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {SYMMETRY_PY}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _extract_list_block(src: str, key: str) -> list[str]:
    m = re.search(rf"{re.escape(key)}\s*=\s*\[(.*?)\]", src, flags=re.DOTALL)
    if not m:
        raise RuntimeError(f"Could not find list block for key={key}")
    return re.findall(r'"([^"]+)"', m.group(1))


def _extract_measured_joint_names(src: str) -> list[str]:
    # Use the first measured_joint_pos block from PolicyCfg.
    m = re.search(
        r"measured_joint_pos\s*=\s*ObsTerm\((.*?)\)\s*measured_joint_vel",
        src,
        flags=re.DOTALL,
    )
    if not m:
        raise RuntimeError("Could not find measured_joint_pos block.")
    return _extract_list_block(m.group(1), "joint_names")


def _make_fake_env(lower_names: list[str]):
    def joint_pos_ordered_rel():
        return None

    def joint_vel_ordered():
        return None

    term_cfg_pos = SimpleNamespace(
        params={"asset_cfg": SimpleNamespace(name="robot", joint_ids=slice(None), joint_names=lower_names)},
        func=joint_pos_ordered_rel,
    )
    term_cfg_vel = SimpleNamespace(
        params={"asset_cfg": SimpleNamespace(name="robot", joint_ids=slice(None), joint_names=lower_names)},
        func=joint_vel_ordered,
    )
    policy_cfg = SimpleNamespace(motor_joint_pos=term_cfg_pos, motor_joint_vel=term_cfg_vel)
    obs_cfg = SimpleNamespace(policy=policy_cfg)
    cfg = SimpleNamespace(observations=obs_cfg)

    class ActionManager:
        def __init__(self, names: list[str]):
            self._term = SimpleNamespace(
                _joint_names=names,
                _joint_ids=list(range(len(names))),
                _asset=SimpleNamespace(data=SimpleNamespace(joint_names=names)),
            )

        def get_term(self, name: str):
            if name != "joint_pos":
                raise KeyError(name)
            return self._term

    scene = {"robot": SimpleNamespace(joint_names=lower_names)}
    env = SimpleNamespace(action_manager=ActionManager(lower_names), scene=scene, cfg=cfg, __dict__={})
    env.unwrapped = env
    return env


def main():
    torch.manual_seed(0)
    mod = _load_symmetry_module()

    lower_names = list(mod._P73_LOWER_JOINT_NAMES_ORDERED)
    print(f"[INFO] Loaded symmetry module: {SYMMETRY_PY}")

    # 1) Config list consistency checks
    src_cfg = ROUGH_CFG_PY.read_text(encoding="utf-8")
    cfg_lower_names = _extract_list_block(src_cfg, "lower_joint_names")
    cfg_measured_names = _extract_measured_joint_names(src_cfg)
    expected_measured = [
        "L_KneePitch_Joint",
        "L_AnklePitch_Joint",
        "L_AnkleRoll_Joint",
        "R_KneePitch_Joint",
        "R_AnklePitch_Joint",
        "R_AnkleRoll_Joint",
    ]
    assert cfg_lower_names == lower_names, "lower_joint_names in rough_env_cfg.py mismatch symmetry order."
    assert cfg_measured_names == expected_measured, "measured_joint_pos order mismatch expected L3+R3 order."
    print("[PASS] Config consistency: lower_joint_names and measured_joint_pos order")

    # 2) Lower-body axis-aware exact mapping check on synthetic input
    x = torch.arange(12, dtype=torch.float32).view(1, 12) + 1.0
    y = mod._flip_lowerbody_12_axis_aware(x)
    expected = torch.tensor([[7.0, -8.0, -9.0, -10.0, -11.0, -12.0, 1.0, -2.0, -3.0, -4.0, -5.0, -6.0]])
    assert torch.allclose(y, expected), f"Axis-aware mapping mismatch.\nGot: {y}\nExp: {expected}"
    print("[PASS] Lower-body mapping: HipRoll swap-only, others swap+negate")

    # 3) Involution checks: f(f(x)) == x
    x_rand = torch.randn(64, 12)
    xx = mod._flip_lowerbody_12_axis_aware(mod._flip_lowerbody_12_axis_aware(x_rand))
    assert torch.allclose(x_rand, xx, atol=1e-6), "Lower-body involution failed."

    m_rand = torch.randn(64, 6)
    mm = mod._flip_measured_6_swap_negate(mod._flip_measured_6_swap_negate(m_rand))
    assert torch.allclose(m_rand, mm, atol=1e-6), "Measured-joint involution failed."
    print("[PASS] Involution: lower-body(12D), measured(6D)")

    # 4) Runtime order assert with fake env
    fake_env = _make_fake_env(lower_names)
    mod._assert_p73_joint_order(fake_env)
    mod._assert_p73_joint_order(fake_env)  # cache path
    print("[PASS] Runtime order assert path (+cache hit)")

    # 5) Full augmentation function checks (shape + double flip recover)
    batch = 8
    hist = 5
    obs = torch.randn(batch, 59 * hist)
    actions = torch.randn(batch, 12)
    obs_aug, act_aug = mod.p73_data_augmentation_lowerbody_mirror(obs, actions, fake_env, obs_type="policy")
    assert obs_aug is not None and act_aug is not None
    assert obs_aug.shape == (batch * 2, 59 * hist), f"Unexpected obs_aug shape: {obs_aug.shape}"
    assert act_aug.shape == (batch * 2, 12), f"Unexpected act_aug shape: {act_aug.shape}"
    assert torch.allclose(obs_aug[:batch], obs), "First half of obs_aug must be original obs."
    assert torch.allclose(act_aug[:batch], actions), "First half of act_aug must be original actions."

    flipped_obs = obs_aug[batch:]
    obs_aug_2, _ = mod.p73_data_augmentation_lowerbody_mirror(flipped_obs, None, fake_env, obs_type="policy")
    assert obs_aug_2 is not None
    recovered_obs = obs_aug_2[batch:]
    assert torch.allclose(recovered_obs, obs, atol=1e-6), "Double-flip recover failed for observations."

    flipped_act = act_aug[batch:]
    _, act_aug_2 = mod.p73_data_augmentation_lowerbody_mirror(None, flipped_act, fake_env, obs_type="policy")
    assert act_aug_2 is not None
    recovered_act = act_aug_2[batch:]
    assert torch.allclose(recovered_act, actions, atol=1e-6), "Double-flip recover failed for actions."
    print("[PASS] Augmentation function: shape/original-half/double-flip recover")

    print("\n[OK] All P73 symmetry checks passed.")


if __name__ == "__main__":
    main()
