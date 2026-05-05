# P73 Teleop Command Control (IsaacLab 5.1)

P73 policy를 **playback(재생)** 하면서 Omniverse UI로 다음을 실시간 제어합니다:

- **Base velocity command**(기저 속도 명령): $(v_x, v_y, \\, \\omega_z)$
- **Disturbance/Push**(외란/푸시): `push_by_setting_velocity`로 root velocity delta를 1회 적용
- **Foot external force**(발 외력): `set_external_force_and_torque`로 발 링크에 임펄스(짧은 시간 힘) 적용

> 적용 범위: **env0 only**(env 0만). 멀티 env에서도 안전하게 사용 가능.

## 실행 예시

### 1) GUI + plotting 파이프

```bash
cd /home/piene/p73/isaaclab_p73 && \
TERM=xterm-256color OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 \
python scripts/tools/p73_command_control/play_with_teleop_p73.py \
  --task P73-Flat-Play \
  --checkpoint logs/rsl_rl/p73_flat/2026-02-11_10-29-56/model_9000.pt \
  --num_envs 1 \
  --control_mode gui \
  --disable_auto_push \
  2>&1 | python tools/plot_contact_force_loco.py
```

### 2) GUI만 (plotting 없음)

```bash
cd /home/piene/p73/isaaclab_p73 && \
TERM=xterm-256color OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 \
python scripts/tools/p73_command_control/play_with_teleop_p73.py \
  --task P73-Flat-Play \
  --checkpoint <PATH_TO_MODEL.pt> \
  --num_envs 1 \
  --control_mode gui \
  --disable_auto_push
```

## 중요한 동작/제약

- **heading_command 강제 OFF**: IsaacLab 기본 locomotion velocity task는 `heading_command=True`일 수 있고, 이 경우 **`command[:, 2] (wz)`가 매 step heading controller에 의해 덮어써질 수 있습니다.**  
  `play_with_teleop_p73.py`는 teleop에서 yaw-rate 슬라이더가 확실히 먹도록 `heading_command=False`로 강제합니다.
- **auto push 비활성 권장**: 학습 cfg의 `events.push_robot`는 interval로 자동 push가 들어옵니다. teleop에서는 혼동을 줄이기 위해 `--disable_auto_push`를 권장합니다(수동 push 버튼은 그대로 동작).

## GUI 구성

- **Base Velocity Control**: 슬라이더로 `vx, vy, wz` 업데이트 (command manager term buffer 직접 수정 + `time_left` 연장)
- **Disturbance / Push (env0)**:
  - `Random Push`: env cfg의 `push_robot.velocity_range`를 best-effort로 읽어 1회 적용(없으면 fallback range)
  - `Custom Push`: 슬라이더 값으로 1회 적용
- **Foot External Force (env0, impulse)**:
  - `Force Mag (N)`와 `Duration (s)` 설정 후 Left/Right 버튼
  - 발 링크는 `L_Foot_Link`, `R_Foot_Link`를 사용

