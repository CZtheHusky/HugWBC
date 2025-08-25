# 🎯 训练时指令采集机制详解

## 📋 概述

HugWBC在训练过程中使用一套复杂的指令采集和重采样系统，确保机器人能够学习在各种运动指令下执行任务。指令系统设计考虑了课程学习、多模态运动和安全性约束。

## 🔧 指令系统架构

### 1. **指令维度定义**
```python
CMD_DIM = 3 + 4 + 1 + 2  # 总共10维指令
# 3: 基础运动指令 (lin_vel_x, lin_vel_y, ang_vel_yaw)
# 4: 步态指令 (gait_frequency, phase, duration, foot_swing_height)
# 1: 身体高度指令 (body_height)
# 2: 身体姿态指令 (body_pitch, waist_roll)
```

### 2. **指令范围配置**
```python
class ranges:
    # 基础运动指令
    lin_vel_x = [-0.6, 0.6]      # 前进/后退速度 [m/s]
    lin_vel_y = [-0.6, 0.6]      # 侧向速度 [m/s]
    ang_vel_yaw = [-0.6, 0.6]    # 偏航角速度 [rad/s]
    
    # 步态指令
    gait_frequency = [1.5, 3.5]  # 步态频率 [Hz]
    foot_swing_height = [0.1, 0.35]  # 摆动腿高度 [m]
    
    # 身体姿态指令
    body_height = [-0.3, 0.0]    # 身体高度偏移 [m]
    body_pitch = [0.0, 0.4]      # 身体俯仰角 [rad]
    waist_roll = [-1.0, 1.0]     # 腰部侧倾角 [rad]
```

## 🎲 指令重采样机制

### 1. **重采样触发时机**
- **环境重置时**: `reset_idx()` 函数中调用 `_resample_commands()`
- **课程学习更新时**: 根据性能动态调整指令范围
- **特定条件下**: 如高速环境、跳跃环境等

### 2. **重采样函数实现**
```python
def _resample_commands(self, env_ids):
    """随机选择一些环境的指令"""
    
    # 1. 基础运动指令采样
    self.commands[env_ids, 0] = torch_rand_float(
        self.command_ranges["lin_vel_x"][0], 
        self.command_ranges["lin_vel_x"][1], 
        (len(env_ids), 1), device=self.device).squeeze(1)
    
    # 2. 步态指令采样
    self.commands[env_ids, 3] = torch_rand_float(
        self.command_ranges["gait_frequency"][0], 
        self.command_ranges["gait_frequency"][1], 
        (len(env_ids), 1), device=self.device).squeeze(1)
    
    # 3. 身体姿态指令采样
    if self.cfg.env.observe_body_height:
        self.commands[env_ids, 7] = torch_rand_float(
            self.command_ranges["body_height"][0], 
            self.command_ranges["body_height"][1], 
            (len(env_ids), 1), device=self.device).squeeze(1)
```

## 🎓 课程学习机制

### 1. **指令课程学习**
```python
class commands:
    curriculum = True           # 启用课程学习
    max_curriculum = 1.        # 最大课程值
    min_vel = 0.15            # 最小速度阈值
    num_bins_vel_x = 12       # 速度x方向分箱数
    num_bins_vel_yaw = 10     # 偏航角速度分箱数
```

### 2. **课程更新逻辑**
```python
def update_command_curriculum_grid(self, env_ids):
    """更新指令课程网格"""
    # 根据性能动态调整指令范围
    self.command_ranges["lin_vel_x"][0] = np.clip(
        self.command_ranges["lin_vel_x"][0] - 0.5, 
        -self.cfg.commands.max_curriculum, 0.)
    self.command_ranges["lin_vel_x"][1] = np.clip(
        self.command_ranges["lin_vel_x"][1] + 0.5, 
        0., self.cfg.commands.max_curriculum)
```

## 🚀 多模态运动支持

### 1. **步态类型**
- **行走模式**: `phase = 0.5` (标准行走步态)
- **跳跃模式**: `phase = 0.0` (跳跃步态)
- **混合模式**: 支持多种步态切换

### 2. **环境分类**
```python
# 高速环境
high_speed_env_mask = self.velocity_level[env_ids] > 1.8
self.commands[high_speed_env_mask, 3] = self.commands[high_speed_env_mask, 3].clip(min=2.0)

# 站立环境
standing_env_floats = torch.rand(len(env_ids), device=self.device)
probability_standing = 1. / 10
standing_env_ids = env_ids[torch.logical_and(0 <= standing_env_floats, standing_env_floats < probability_standing)]
self.commands[standing_env_ids, :3] = 0  # 停止运动
```

## 🔒 安全约束机制

### 1. **速度约束**
```python
# 设置最小速度阈值
self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > self.cfg.commands.min_vel).unsqueeze(1)
self.commands[env_ids, 2] *= (torch.abs(self.commands[env_ids, 2]) > self.cfg.commands.min_vel)
```

### 2. **姿态约束**
```python
# 高速环境下的姿态约束
self.commands[env_ids[high_speed_env_mask], 8] = self.commands[env_ids[high_speed_env_mask], 8].clip(max=0.3)

# 低高度环境下的摆动高度约束
low_height_env_mask = self.commands[env_ids, 7] < -0.15
self.commands[env_ids[low_height_env_mask], 6] = self.commands[env_ids[low_height_env_mask], 6].clip(max=0.20)
```

## 📊 指令分布策略

### 1. **均匀分布**
- **基础运动指令**: 在指定范围内均匀随机采样
- **步态参数**: 频率、相位、摆动高度等均匀分布

### 2. **离散分布**
- **步态相位**: 从预定义集合 `[0, 0.5]` 中随机选择
- **步态类型**: 行走/跳跃模式随机切换

### 3. **固定值**
- **步态持续时间**: `duration = 0.5` (固定值)
- **某些约束参数**: 根据环境条件固定设置

## 🎯 指令使用流程

### 1. **训练循环中的指令流**
```
环境重置 → 指令重采样 → 策略执行 → 奖励计算 → 课程更新
    ↓           ↓           ↓         ↓         ↓
reset_idx() → _resample_commands() → step() → compute_reward() → update_curriculum()
```

### 2. **指令生命周期**
- **生成**: 环境重置时随机生成
- **执行**: 策略根据指令生成动作
- **评估**: 奖励函数评估指令执行质量
- **更新**: 课程学习系统动态调整指令范围

## 🔍 指令监控和分析

### 1. **指令统计**
```python
# 记录指令执行统计
self.command_sums[name][env_ids] += rew

# 记录最大指令值
self.extras["episode"]["max_command_x"] = torch.max(self.commands[:, 0])
self.extras["episode"]["max_command_yaw"] = torch.max(self.commands[:, 2])
```

### 2. **性能关联**
- **指令难度**: 与机器人性能水平关联
- **课程进度**: 根据成功率动态调整
- **失败分析**: 识别困难指令模式

## 🚀 高级特性

### 1. **地形适应**
- **地形课程**: 指令与地形难度协同
- **环境感知**: 根据地形类型调整指令

### 2. **多机器人协调**
- **环境并行**: 多个环境同时执行不同指令
- **负载均衡**: 指令分布优化计算资源

### 3. **实时调整**
- **动态重采样**: 训练过程中实时调整指令
- **性能反馈**: 根据奖励信号优化指令分布

## 📝 总结

HugWBC的训练指令系统是一个高度复杂和智能的系统：

1. **多维度指令**: 涵盖运动、步态、姿态等多个方面
2. **智能采样**: 使用课程学习和约束机制确保指令质量
3. **安全约束**: 多层次的安全检查防止危险指令
4. **动态适应**: 根据训练进度和性能动态调整
5. **多模态支持**: 支持行走、跳跃等多种运动模式

这个系统确保了机器人能够在各种复杂指令下学习稳定的运动技能，为实际应用奠定了坚实的基础！

