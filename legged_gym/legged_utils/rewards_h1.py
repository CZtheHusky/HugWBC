import torch
from typing import Dict, Tuple


def _get_tensor(d: Dict, key: str, alt_keys=None, required=True):
    if key in d:
        return d[key]
    if alt_keys:
        for k in alt_keys:
            if k in d:
                return d[k]
    if required:
        raise KeyError(f"Missing required key: {key}")
    return None


def _maybe_compute_torques(s: Dict, a: torch.Tensor) -> torch.Tensor:
    # Prefer directly provided torques
    if 'torques' in s:
        return s['torques']

    # Optional PD reconstruction: torques = p_gains*(actions_scaled + default_dof_pos - dof_pos + motor_offsets) - d_gains*dof_vel
    p_gains = s.get('p_gains', None)
    d_gains = s.get('d_gains', None)
    default_dof_pos = s.get('default_dof_pos', None)
    motor_offsets = s.get('motor_offsets', None)
    dof_pos = s.get('dof_pos', None)
    dof_vel = s.get('dof_vel', None)
    action_scale = s.get('action_scale', None)
    motor_strength = s.get('motor_strength', None)
    torque_limits = s.get('torque_limits', None)

    if p_gains is None or d_gains is None or default_dof_pos is None or dof_pos is None or dof_vel is None or action_scale is None:
        raise KeyError("Cannot reconstruct torques: provide 'torques' or PD fields: p_gains, d_gains, default_dof_pos, dof_pos, dof_vel, action_scale")

    # Broadcast constants to batch if needed
    batch = dof_pos.shape[0]
    if p_gains.dim() == 1:
        p_gains = p_gains.unsqueeze(0).expand(batch, -1)
    if d_gains.dim() == 1:
        d_gains = d_gains.unsqueeze(0).expand(batch, -1)
    if default_dof_pos.dim() == 1:
        default_dof_pos = default_dof_pos.unsqueeze(0).expand(batch, -1)
    if motor_offsets is None:
        motor_offsets = torch.zeros_like(dof_pos)
    if motor_strength is None:
        motor_strength = torch.ones_like(dof_pos)

    actions_scaled = a * action_scale
    torques = p_gains * (actions_scaled + default_dof_pos - dof_pos + motor_offsets) - d_gains * dof_vel
    torques = torques * motor_strength
    if torque_limits is not None:
        if torque_limits.dim() == 1:
            torque_limits = torque_limits.unsqueeze(0).expand_as(torques)
        torques = torch.clip(torques, -torque_limits, torque_limits)
    return torques


def compute_reward_sas1(
    s: Dict[str, torch.Tensor],
    a: torch.Tensor,
    s1: Dict[str, torch.Tensor],
    config: Dict,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Pure reward computation r(s, a, s1) for H1-style legged env.

    Required inputs (batched):
      s: {
        base_lin_vel: [N,3] (body frame),
        base_ang_vel: [N,3] (body frame),
        dof_vel: [N,D],
        commands: [N,3] (vx, vy, yaw)
        ... (optional PD fields or torques),
        collision_states: [N,K] (bool) optional
      }
      a: [N,D] actions
      s1: {
        dof_vel: [N,D] (next),
      }
      config: {
        dt: float,
        reward_scales: dict[str,float],
        tracking_sigma: float,
        max_contact_force: float (if using force-based collisions),
      }

    Returns:
      total_reward: [N]
      terms: dict of per-term rewards [N]
    """
    dt = torch.as_tensor(config.get('dt', 0.02), dtype=a.dtype, device=a.device)
    reward_scales = config.get('reward_scales', {})
    tracking_sigma = torch.as_tensor(config.get('tracking_sigma', 0.25), dtype=a.dtype, device=a.device)

    # Extract common tensors
    base_lin_vel = _get_tensor(s, 'base_lin_vel')
    base_ang_vel = _get_tensor(s, 'base_ang_vel')
    dof_vel = _get_tensor(s, 'dof_vel')
    commands = _get_tensor(s, 'commands')  # [:,0:3] -> vx, vy, yaw_rate_cmd
    dof_vel_next = _get_tensor(s1, 'dof_vel')

    # Optional
    collision_states = s.get('collision_states', None)

    # Optionally compute torques from PD or use provided
    torques = None
    if ('torques' in s) or all(k in s for k in ['p_gains', 'd_gains', 'default_dof_pos', 'dof_pos', 'dof_vel', 'action_scale']):
        torques = _maybe_compute_torques(s, a)

    terms: Dict[str, torch.Tensor] = {}

    # tracking_lin_vel
    if 'tracking_lin_vel' in reward_scales:
        lin_err = torch.sum((commands[:, :2] - base_lin_vel[:, :2]) ** 2, dim=1)
        terms['tracking_lin_vel'] = torch.exp(-lin_err / tracking_sigma)

    # tracking_ang_vel
    if 'tracking_ang_vel' in reward_scales:
        ang_err = (commands[:, 2] - base_ang_vel[:, 2]) ** 2
        terms['tracking_ang_vel'] = torch.exp(-ang_err / tracking_sigma)

    # lin_vel_z (penalty)
    if 'lin_vel_z' in reward_scales:
        terms['lin_vel_z'] = (base_lin_vel[:, 2] ** 2)

    # ang_vel_xy (penalty)
    if 'ang_vel_xy' in reward_scales:
        terms['ang_vel_xy'] = torch.sum(base_ang_vel[:, :2] ** 2, dim=1)

    # torques (penalty)
    if 'torques' in reward_scales and torques is not None:
        terms['torques'] = torch.sum(torques ** 2, dim=1)

    # dof_vel (penalty)
    if 'dof_vel' in reward_scales:
        terms['dof_vel'] = torch.sum(dof_vel ** 2, dim=1)

    # dof_acc (penalty)
    if 'dof_acc' in reward_scales and dof_vel_next is not None:
        terms['dof_acc'] = torch.sum(((dof_vel_next - dof_vel) / dt) ** 2, dim=1)

    # collision (penalty) – if provided as boolean states
    if 'collision' in reward_scales and collision_states is not None:
        terms['collision'] = torch.sum((collision_states.to(a.dtype) > 0), dim=1)

    # Aggregate total with scales (scaled by dt like in env)
    total = torch.zeros(a.shape[0], dtype=a.dtype, device=a.device)
    for name, scale in reward_scales.items():
        if name not in terms:
            continue
        total = total + terms[name] * (scale * dt)

    return total, terms


