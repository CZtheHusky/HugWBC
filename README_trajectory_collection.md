# HugWBC 轨迹采集系统

这个系统用于收集HugWBC的轨迹数据，支持两种类型的轨迹：命令不变的轨迹和命令变化的轨迹。

## 功能特点

### 1. 轨迹类型
- **命令不变轨迹**: 整个10秒轨迹使用相同的命令
- **命令变化轨迹**: 前5秒使用一个命令，后5秒使用另一个命令

### 2. 数据格式
- **轨迹数据**: 保存为 `.npz` 格式，包含完整的观察、动作、奖励等数据
- **元数据**: 保存为 `JSON` 格式，包含轨迹ID、类型、命令等信息

### 3. 命令采样
- 从训练时使用的可行域中均匀随机采样
- 应用物理约束确保命令的合理性
- 支持跳跃和行走两种步态

## 使用方法

### 基本用法

```bash
# 收集10条命令不变轨迹和10条命令变化轨迹
python legged_gym/scripts/collect_trajectories.py --task=h1int --headless

# 指定轨迹数量
python legged_gym/scripts/collect_trajectories.py --task=h1int --num_constant=20 --num_changing=15 --headless

# 使用训练好的策略生成动作
python legged_gym/scripts/collect_trajectories.py --task=h1int --load_checkpoint --headless
```

### 参数说明

- `--task`: 任务名称，默认为 "h1int"
- `--num_constant`: 命令不变轨迹的数量，默认为10
- `--num_changing`: 命令变化轨迹的数量，默认为10
- `--load_checkpoint`: 是否加载训练好的策略
- `--headless`: 是否以无头模式运行

## 输出结构

### 目录结构
```
collected_trajectories/
├── trajectory_metadata.json          # 轨迹元数据
├── uuid1.npz                        # 轨迹1数据
├── uuid2.npz                        # 轨迹2数据
└── ...
```

### 轨迹数据格式 (.npz)
每个 `.npz` 文件包含以下数据：

- `obs`: 观察数据，形状为 `(timesteps, 1, 5, 76)`
- `critic_obs`: 评论者观察数据
- `actions`: 动作数据，形状为 `(timesteps, 1, 19)`
- `rewards`: 奖励数据
- `dones`: 完成状态
- `root_states`: 根状态数据
- `commands`: 命令数据，形状为 `(timesteps, 11)`
- `dt`: 时间步长

### 元数据格式 (JSON)
```json
{
  "trajectory_id": "uuid-string",
  "trajectory_type": "constant" | "changing",
  "commands": [
    [vx, vy, yaw, freq, phase, duration, swing_h, body_h, body_pitch, waist_roll, interrupt_flag]
  ],
  "length_s": 10.0,
  "length_timesteps": 5000,
  "collection_time": "2024-01-01T12:00:00",
  "file_path": "collected_trajectories/uuid.npz"
}
```

## 命令约束

系统会自动应用以下约束确保物理合理性：

### 1. 高速环境约束
- 当速度水平 > 1.8 时：
  - 最小步态频率: 2.0 Hz
  - 最大摆动高度: 0.20 m
  - 最大身体俯仰角: 0.3 rad
  - 腰部侧倾角限制: [-0.15, 0.15] rad

### 2. 跳跃环境约束
- 最大摆动高度: 0.2 m
- 最大身体俯仰角: 0.3 rad

### 3. 低身体高度约束
- 当身体高度 < -0.15 m 时，限制摆动高度
- 当身体高度 < -0.2 m 时，限制俯仰角

## 命令范围

系统使用训练时的命令范围：

```python
command_ranges = {
    'lin_vel_x': [-0.6, 0.6],        # 前进速度 [m/s]
    'lin_vel_y': [-0.6, 0.6],        # 侧向速度 [m/s]
    'ang_vel_yaw': [-0.6, 0.6],      # 偏航角速度 [rad/s]
    'gait_frequency': [1.5, 3.5],    # 步态频率 [Hz]
    'foot_swing_height': [0.1, 0.35], # 摆动高度 [m]
    'body_height': [-0.3, 0.0],      # 身体高度 [m]
    'body_pitch': [0.0, 0.4],        # 身体俯仰角 [rad]
    'waist_roll': [-1.0, 1.0]        # 腰部侧倾角 [rad]
}
```

## 使用示例

### 1. 收集基础轨迹
```bash
cd /root/workspace/HugWBC
python legged_gym/scripts/collect_trajectories.py --task=h1int --headless
```

### 2. 使用训练好的策略
```bash
python legged_gym/scripts/collect_trajectories.py --task=h1int --load_checkpoint --headless
```

### 3. 自定义轨迹数量
```bash
python legged_gym/scripts/collect_trajectories.py --task=h1int --num_constant=50 --num_changing=30 --headless
```

## 注意事项

1. **环境要求**: 确保Isaac Gym环境已正确安装和配置
2. **策略加载**: 使用 `--load_checkpoint` 时需要确保有可用的训练检查点
3. **存储空间**: 每条轨迹约占用几MB存储空间，请确保有足够的磁盘空间
4. **中断恢复**: 系统支持中断后继续采集，会自动加载已有的元数据

## 故障排除

### 常见问题

1. **环境创建失败**: 检查Isaac Gym安装和配置
2. **策略加载失败**: 检查检查点文件路径和格式
3. **内存不足**: 减少同时运行的环境数量
4. **存储空间不足**: 检查输出目录的可用空间

### 调试模式

可以修改脚本添加更多调试信息：

```python
# 在 _collect_single_trajectory 中添加
print(f"Command: {current_command}")
print(f"Observation shape: {obs.shape}")
print(f"Action shape: {actions.shape}")
```

## 扩展功能

系统设计为可扩展的，可以轻松添加：

1. **新的轨迹类型**: 在 `_collect_single_trajectory` 中添加新的类型
2. **新的命令约束**: 在 `_apply_command_constraints` 中添加新的约束
3. **数据后处理**: 在保存前添加数据清洗和验证
4. **批量处理**: 支持并行采集多条轨迹

## 联系信息

如有问题或建议，请通过以下方式联系：
- 提交Issue到项目仓库
- 发送邮件到项目维护者


