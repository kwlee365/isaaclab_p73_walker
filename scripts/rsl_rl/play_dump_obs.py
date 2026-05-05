# Copyright (c) 2025, Jaeyong Shin (jasonshin0537@snu.ac.kr).
# Walker sim-to-sim debug: dump per-term obs with zero command.
#
# Usage:
#   cd /home/piene/isaaclab5.2/isaaclab_walker
#   OMNI_KIT_ACCEPT_EULA=YES python scripts/rsl_rl/play_dump_obs.py \
#       --task=P73-Rough-Play --num_envs=1 \
#       --checkpoint=<path_to_model.pt> \
#       --dump_steps 25

"""Launch Isaac Sim Simulator first."""

import argparse
from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Walker sim2sim debug: dump obs with zero cmd.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--dump_steps", type=int, default=25, help="Number of policy steps to dump.")
parser.add_argument("--dump_file", type=str, default="/tmp/walker_isaac_obs.jsonl")
parser.add_argument("--disable_fabric", action="store_true", default=False)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.num_envs = 1  # force single env

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json
import os

import gymnasium as gym
import isaaclab_walker.tasks  # noqa: F401
import isaaclab_tasks  # noqa: F401
import torch
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from rsl_rl.runners import OnPolicyRunner

try:
    from isaaclab_walker.algorithms.rsl_rl import P73OnPolicyRunner as OnPolicyRunner
    print("[INFO] Using custom P73OnPolicyRunner")
except ImportError:
    pass


def extract_newest_frame(obs_flat, H=5):
    """Extract newest 47D frame from 235D frame-major observation.

    Frame-major layout: [frame0(47D), frame1(47D), ..., frame4(47D)]
    Newest frame is the LAST 47 elements.
    """
    F = 47  # single frame size
    newest = obs_flat[(H - 1) * F : H * F]

    dims = [3, 3, 3, 1, 1, 12, 12, 12]
    names = ["ang_vel", "gravity", "cmd", "gait_sin", "gait_cos",
             "joint_pos", "joint_vel", "last_action"]
    frame = {}
    offset = 0
    for name, d in zip(names, dims):
        vals = newest[offset:offset + d].tolist()
        frame[name] = vals if d > 1 else vals[0]
        offset += d
    return frame


def main():
    task_name = args_cli.task.split(":")[-1]
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=1,
        use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(task_name, args_cli)

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)

    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Load policy
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # Force zero velocity command
    cmd_mgr = env.unwrapped.command_manager
    print(f"[DEBUG] Command manager terms: {list(cmd_mgr._terms.keys())}")

    # Reset environment
    obs_raw = env.get_observations()

    # Debug: understand obs structure
    print(f"[DEBUG] obs type={type(obs_raw)}")
    if isinstance(obs_raw, torch.Tensor):
        print(f"[DEBUG] obs tensor shape={obs_raw.shape}")
    elif isinstance(obs_raw, dict):
        print(f"[DEBUG] obs dict keys={list(obs_raw.keys())}")
        for k, v in obs_raw.items():
            if isinstance(v, torch.Tensor):
                print(f"[DEBUG]   {k}: tensor shape={v.shape}")
            else:
                print(f"[DEBUG]   {k}: type={type(v)}")
    else:
        # TensorDict or similar
        print(f"[DEBUG] obs repr={repr(obs_raw)[:500]}")
        if hasattr(obs_raw, 'keys'):
            print(f"[DEBUG] obs keys={list(obs_raw.keys())}")

    # Extract policy obs as a plain tensor (num_envs, obs_dim)
    def get_policy_tensor(o):
        """Recursively extract the policy observation tensor."""
        if isinstance(o, torch.Tensor) and o.dim() >= 2:
            return o
        if isinstance(o, torch.Tensor) and o.dim() == 1:
            return o.unsqueeze(0)
        if isinstance(o, dict):
            if "policy" in o:
                return get_policy_tensor(o["policy"])
            # try first value
            first = next(iter(o.values()))
            return get_policy_tensor(first)
        # TensorDict
        if hasattr(o, 'keys') and hasattr(o, 'to_dict'):
            return get_policy_tensor(o.to_dict())
        if hasattr(o, 'keys'):
            if "policy" in list(o.keys()):
                return get_policy_tensor(o["policy"])
        # Last resort: try converting
        if hasattr(o, 'contiguous'):
            t = o.contiguous()
            if isinstance(t, torch.Tensor):
                return t
        raise ValueError(f"Cannot extract policy tensor from {type(o)}: {repr(o)[:200]}")

    policy_obs = get_policy_tensor(obs_raw)
    obs_dim = policy_obs.shape[-1]
    single_frame = 47
    H = obs_dim // single_frame
    print(f"[DEBUG] policy_obs shape={policy_obs.shape}, obs_dim={obs_dim}, H={H}")

    dump_path = args_cli.dump_file
    print(f"[INFO] Dumping {args_cli.dump_steps} steps to {dump_path}")

    with open(dump_path, "w") as f:
        for step in range(args_cli.dump_steps):
            with torch.inference_mode():
                # Force zero command BEFORE observation/action
                for term_name in cmd_mgr._terms:
                    term = cmd_mgr._terms[term_name]
                    if hasattr(term, 'command'):
                        term.command[:] = 0.0

                actions = policy(obs_raw)
                obs_raw, _, _, _ = env.step(actions)

                # Force zero command AFTER step (for next obs)
                for term_name in cmd_mgr._terms:
                    term = cmd_mgr._terms[term_name]
                    if hasattr(term, 'command'):
                        term.command[:] = 0.0

            policy_obs = get_policy_tensor(obs_raw)
            obs_flat = policy_obs[0].cpu().float()
            act_flat = actions[0].cpu().float()
            frame = extract_newest_frame(obs_flat, H=H)

            # Get raw robot state
            robot = env.unwrapped.scene["robot"]
            quat = robot.data.root_quat_w[0].cpu().tolist()  # (w,x,y,z) isaac convention
            ang_vel = robot.data.root_ang_vel_b[0].cpu().tolist()
            joint_pos = robot.data.joint_pos[0].cpu().tolist()
            joint_vel = robot.data.joint_vel[0].cpu().tolist()

            record = {
                "step": step,
                "obs_235": obs_flat.tolist(),
                "actions": act_flat.tolist(),
                "frame_47": frame,
                "raw": {
                    "quat_wxyz": quat,
                    "ang_vel_body": ang_vel,
                    "joint_pos": joint_pos,
                    "joint_vel": joint_vel,
                },
            }
            f.write(json.dumps(record) + "\n")

            if step < 5:
                print(f"\n=== IsaacLab STEP {step} ===")
                print(f"  ang_vel:    {frame['ang_vel']}")
                print(f"  gravity:    {frame['gravity']}")
                print(f"  cmd:        {frame['cmd']}")
                print(f"  gait_sin:   {frame['gait_sin']}")
                print(f"  gait_cos:   {frame['gait_cos']}")
                print(f"  joint_pos:  {[f'{v:.4f}' for v in frame['joint_pos']]}")
                print(f"  joint_vel:  {[f'{v:.4f}' for v in frame['joint_vel']]}")
                print(f"  last_act:   {[f'{v:.4f}' for v in frame['last_action']]}")
                print(f"  actions:    {[f'{v:.4f}' for v in act_flat.tolist()]}")

    print(f"\n[DONE] Dumped {args_cli.dump_steps} steps to {dump_path}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
