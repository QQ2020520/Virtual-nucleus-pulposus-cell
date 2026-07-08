"""
ABM 微环境建模 — Virtual NP Cell
========================================
基于 Agent-Based Model (ABM) 的椎间盘微环境仿真:
NP 细胞、免疫细胞(巨噬细胞)、成纤维细胞、基质成分在 2D 网格上的交互

核心设计:
- NP 细胞具有状态机: 正常 → 应激 → (凋亡/衰老)
- 网格微环境: ECM 浓度、炎症因子浓度、氧浓度
- 细胞行为受局部微环境(炎症、氧、力学负载)调控
"""

import numpy as np
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from typing import Optional, Dict, List, Tuple, Any
from collections import defaultdict

plt.rcParams['font.family'] = ['HarmonyHeiTi', 'Droid Sans', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. 椎间盘微环境网格
# ============================================================

class DiscMicroenvGrid:
    """
    椎间盘微环境 2D 网格。

    每个格点存储:
    - ecm: ECM 浓度 (0~1)
    - inflammation: 炎症因子浓度 (0~1)
    - oxygen: 氧浓度 (0~1)
    - mechanical_load: 力学负载 (0~1)
    - nutrient: 营养浓度 (0~1)
    """

    def __init__(self, width: int = 30, height: int = 30, seed: int = 42):
        self.width = width
        self.height = height
        self.rng = np.random.RandomState(seed)

        # 初始化环境
        self.ecm = np.ones((height, width)) * 0.8  # 富 ECM
        self.inflammation = np.zeros((height, width))
        self.oxygen = np.ones((height, width)) * 0.7  # 正常氧
        self.mechanical_load = np.ones((height, width)) * 0.3  # 正常负载
        self.nutrient = np.ones((height, width)) * 0.6  # 营养

        # 添加区域差异: 中心区域 ECM 更丰富, 边缘氧供更好
        center_y, center_x = height // 2, width // 2
        for y in range(height):
            for x in range(width):
                dist = np.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)
                max_dist = np.sqrt(center_y ** 2 + center_x ** 2)
                # 中心 → ECM 高, 氧低; 边缘 → ECM 低, 氧高
                norm_dist = dist / max_dist
                self.ecm[y, x] = 0.5 + 0.5 * (1 - norm_dist)
                self.oxygen[y, x] = 0.3 + 0.7 * norm_dist
                # 初始炎症: 随机微小波动
                self.inflammation[y, x] = self.rng.uniform(0, 0.05)

    def update_environment(self, agents: List['NPAgent'],
                           dt: float = 1.0,
                           inflam_diffusion: float = 0.1,
                           ecm_degradation: float = 0.02,
                           oxygen_consumption: float = 0.01):
        """
        根据 agent 活动更新微环境。

        Parameters
        ----------
        agents : list of NPAgent
        dt : float
            时间步
        inflam_diffusion : float
            炎症扩散系数
        ecm_degradation : float
            ECM 降解速率 (由炎症等驱动)
        oxygen_consumption : float
            氧消耗速率
        """
        # 扩散 (简易扩散: 相邻格平均)
        inflam_new = self.inflammation.copy()
        for y in range(1, self.height - 1):
            for x in range(1, self.width - 1):
                neighbors = self.inflammation[y-1:y+2, x-1:x+2]
                inflam_new[y, x] += inflam_diffusion * (neighbors.mean() - self.inflammation[y, x])
        self.inflammation = np.clip(inflam_new, 0, 1)

        # 炎症自然衰减
        self.inflammation *= (1 - 0.03 * dt)

        # ECM 降解 (炎症区域)
        self.ecm -= ecm_degradation * self.inflammation * dt
        self.ecm = np.clip(self.ecm, 0, 1)

        # ECM 基础合成 (NP 细胞维持)
        for agent in agents:
            if agent.state in ('normal', 'stressed') and agent.alive:
                y, x = int(agent.y), int(agent.x)
                if 0 <= y < self.height and 0 <= x < self.width:
                    self.ecm[y, x] = min(1.0, self.ecm[y, x] + 0.01 * dt)

        # 力学负载随时间变化 (模拟日常活动)
        load_change = self.rng.normal(0, 0.02)
        self.mechanical_load = np.clip(self.mechanical_load + load_change, 0, 1)

        # 氧消耗 (细胞呼吸)
        self.oxygen -= oxygen_consumption * dt
        # 氧从边缘补充
        self.oxygen[0, :] = 0.9
        self.oxygen[-1, :] = 0.9
        self.oxygen[:, 0] = 0.9
        self.oxygen[:, -1] = 0.9
        # 内扩散
        for _ in range(2):
            oxy_new = self.oxygen.copy()
            for y in range(1, self.height - 1):
                for x in range(1, self.width - 1):
                    neighbors = self.oxygen[y-1:y+2, x-1:x+2]
                    oxy_new[y, x] += 0.1 * (neighbors.mean() - self.oxygen[y, x])
            self.oxygen = np.clip(oxy_new, 0, 1)


# ============================================================
# 2. NP 微环境 Agent 类
# ============================================================

class NPAgent:
    """
    NP 微环境 Agent。

    属性:
    - cell_type: 'NP_cell', 'macrophage', 'fibroblast'
    - state: 'normal', 'stressed', 'apoptotic', 'senescent'
    - x, y: 网格位置
    - age: 年龄 (时间步)
    - lifespan: 最大寿命
    - alive: 存活标志
    """

    # 细胞类型颜色映射
    TYPE_COLORS = {
        'NP_cell': '#3498DB',
        'macrophage': '#E74C3C',
        'fibroblast': '#27AE60',
    }

    # 状态颜色
    STATE_COLORS = {
        'normal': '#2ECC71',
        'stressed': '#F39C12',
        'apoptotic': '#E74C3C',
        'senescent': '#8E44AD',
        'dead': '#7F8C8D',
    }

    def __init__(self, agent_id: int, cell_type: str, x: float, y: float,
                 seed: int = 42):
        self.id = agent_id
        self.cell_type = cell_type
        self.x = x
        self.y = y
        self.rng = np.random.RandomState(seed + agent_id)

        # 细胞状态
        self.alive = True
        self.state = 'normal'
        self.age = 0

        # 寿命 (不同细胞类型不同)
        if cell_type == 'NP_cell':
            self.max_age = self.rng.randint(80, 150)
            self.stress_threshold = 0.6
        elif cell_type == 'macrophage':
            self.max_age = self.rng.randint(15, 40)
            self.stress_threshold = 0.4
        elif cell_type == 'fibroblast':
            self.max_age = self.rng.randint(50, 100)
            self.stress_threshold = 0.5
        else:
            self.max_age = 50
            self.stress_threshold = 0.5

        # 累积应激
        self.stress_level = 0.0
        self.proliferation_rate = 0.0

    def step(self, grid: DiscMicroenvGrid, dt: float = 1.0):
        """
        单步更新 agent 行为。

        Parameters
        ----------
        grid : DiscMicroenvGrid
            微环境网格
        dt : float
            时间步
        """
        if not self.alive:
            return

        self.age += dt

        # 获取局部环境
        y, x = int(self.y), int(self.x)
        y = np.clip(y, 0, grid.height - 1)
        x = np.clip(x, 0, grid.width - 1)

        local_inflam = grid.inflammation[y, x]
        local_oxygen = grid.oxygen[y, x]
        local_load = grid.mechanical_load[y, x]
        local_nutrient = grid.nutrient[y, x]
        local_ecm = grid.ecm[y, x]

        # 计算应激
        stress_factors = {
            'inflammation': local_inflam * 0.4,
            'hypoxia': max(0, 0.3 - local_oxygen) * 0.3,
            'mechanical': local_load * 0.2,
            'nutrient_deficit': max(0, 0.3 - local_nutrient) * 0.3,
            'ecm_loss': max(0, 0.3 - local_ecm) * 0.2,
        }
        total_stress = sum(stress_factors.values())

        # 状态转换
        if self.state == 'normal':
            self.stress_level = total_stress

            if total_stress > self.stress_threshold * 1.5:
                self.state = 'stressed'
            elif total_stress > self.stress_threshold * 2.5:
                self.state = 'stressed'

            # 正常细胞增殖 (极低速率)
            if total_stress < 0.2 and self.rng.random() < 0.005 * dt:
                self.proliferation_rate = 1.0

        elif self.state == 'stressed':
            self.stress_level = total_stress

            # 应激 → 凋亡/衰老
            if total_stress > 0.8 or self.age > self.max_age * 0.9:
                if self.rng.random() < 0.1 * dt:
                    self.state = 'apoptotic'
                elif self.rng.random() < 0.05 * dt:
                    self.state = 'senescent'
            # 应激缓解则恢复
            elif total_stress < 0.2 and self.rng.random() < 0.03 * dt:
                self.state = 'normal'

        elif self.state == 'apoptotic':
            # 凋亡细胞不久后死亡
            if self.rng.random() < 0.3 * dt:
                self.alive = False
                self.state = 'dead'

        elif self.state == 'senescent':
            # 衰老细胞存活但失去功能
            pass

        # 迁移 (仅 NP 细胞和巨噬细胞有微迁移能力)
        if self.cell_type in ('NP_cell', 'macrophage') and self.state in ('normal', 'stressed'):
            # 向低炎症或营养丰富的方向迁移
            dx = self.rng.uniform(-0.3, 0.3)
            dy = self.rng.uniform(-0.3, 0.3)
            self.x = np.clip(self.x + dx, 0, grid.width - 1)
            self.y = np.clip(self.y + dy, 0, grid.height - 1)

    def get_color(self) -> str:
        """获取 Agent 显示颜色。"""
        if not self.alive:
            return self.STATE_COLORS['dead']
        if self.state in self.STATE_COLORS:
            return self.STATE_COLORS[self.state]
        return self.TYPE_COLORS.get(self.cell_type, '#95A5A6')

    def get_marker(self) -> str:
        """获取 Agent 显示标记。"""
        markers = {
            'NP_cell': 'o',
            'macrophage': '^',
            'fibroblast': 's',
        }
        return markers.get(self.cell_type, 'o')


# ============================================================
# 3. ABM 仿真运行
# ============================================================

def run_abm_simulation(
    grid_size: int = 25,
    n_np_cells: int = 40,
    n_macrophages: int = 8,
    n_fibroblasts: int = 10,
    n_steps: int = 100,
    seed: int = 42,
    degenerative: bool = False,
    inflam_seed: float = 0.0,
    mechanical_overload: float = 0.0,
) -> Dict[str, Any]:
    """
    运行 NP 微环境 ABM 仿真。

    Parameters
    ----------
    grid_size : int
        网格大小 (grid_size × grid_size)
    n_np_cells : int
        NP 细胞初始数量
    n_macrophages : int
        巨噬细胞初始数量
    n_fibroblasts : int
        成纤维细胞初始数量
    n_steps : int
        仿真步数
    seed : int
        随机种子
    degenerative : bool
        是否退变模式 (初始炎症更高)
    inflam_seed : float
        初始炎症种子强度 (0-1)
    mechanical_overload : float
        力学过载程度

    Returns
    -------
    dict
        {
            'agents': list of NPAgent (最终状态)
            'grid_history': list of grid snapshots (每10步)
            'population_history': dict — 各类细胞数量变化
            'state_history': dict — NP 细胞状态比例变化
            'ecm_history': list — ECM 均值变化
            'inflam_history': list — 炎症均值变化
        }
    """
    rng = np.random.RandomState(seed)

    # 创建网格
    grid = DiscMicroenvGrid(width=grid_size, height=grid_size, seed=seed)

    # 退变模式下增加初始炎症和 ECM 降解
    if degenerative:
        grid.inflammation += 0.2 + inflam_seed * 0.3
        grid.ecm -= 0.2
        grid.oxygen -= 0.15
        grid.mechanical_load += mechanical_overload * 0.3
        grid.inflammation = np.clip(grid.inflammation, 0, 1)
        grid.ecm = np.clip(grid.ecm, 0, 1)
        grid.oxygen = np.clip(grid.oxygen, 0, 1)
        grid.mechanical_load = np.clip(grid.mechanical_load, 0.3, 1)

    # 创建 agents
    agents: List[NPAgent] = []
    agent_counter = 0

    for _ in range(n_np_cells):
        x = rng.uniform(grid_size * 0.2, grid_size * 0.8)  # NP 细胞趋向中心
        y = rng.uniform(grid_size * 0.2, grid_size * 0.8)
        agents.append(NPAgent(agent_counter, 'NP_cell', x, y, seed + agent_counter))
        agent_counter += 1

    for _ in range(n_macrophages):
        x = rng.uniform(0, grid_size)
        y = rng.uniform(0, grid_size)
        agents.append(NPAgent(agent_counter, 'macrophage', x, y, seed + agent_counter))
        agent_counter += 1

    for _ in range(n_fibroblasts):
        x = rng.uniform(0, grid_size)
        y = rng.uniform(0, grid_size)
        agents.append(NPAgent(agent_counter, 'fibroblast', x, y, seed + agent_counter))
        agent_counter += 1

    # 记录历史
    grid_history = []
    pop_history = defaultdict(list)
    state_history = defaultdict(list)
    ecm_history = []
    inflam_history = []
    oxy_history = []

    # 仿真主循环
    for step in range(n_steps):
        # 更新环境 (每步)
        grid.update_environment(agents)

        # 更新每个 agent
        for agent in agents:
            agent.step(grid)

        # 移除死亡的 agent (只取前几步)
        agents = [a for a in agents if a.alive]

        # 记录历史 (每 5 步)
        if step % 5 == 0:
            # 种群统计
            for ct in ['NP_cell', 'macrophage', 'fibroblast']:
                count = sum(1 for a in agents if a.cell_type == ct)
                pop_history[ct].append((step, count))

            # NP 细胞状态统计
            np_cells = [a for a in agents if a.cell_type == 'NP_cell']
            for state in ['normal', 'stressed', 'apoptotic', 'senescent']:
                count = sum(1 for a in np_cells if a.state == state)
                state_history[state].append((step, count))

            # 微环境均值
            ecm_history.append((step, grid.ecm.mean()))
            inflam_history.append((step, grid.inflammation.mean()))
            oxy_history.append((step, grid.oxygen.mean()))

        # 保存网格快照 (每 20 步)
        if step % 20 == 0 or step == n_steps - 1:
            grid_history.append({
                'step': step,
                'ecm': grid.ecm.copy(),
                'inflammation': grid.inflammation.copy(),
                'oxygen': grid.oxygen.copy(),
                'agents': [{
                    'x': a.x, 'y': a.y,
                    'type': a.cell_type,
                    'state': a.state,
                    'alive': a.alive,
                } for a in agents],
            })

    return {
        'agents': agents,
        'grid_history': grid_history,
        'population_history': dict(pop_history),
        'state_history': dict(state_history),
        'ecm_history': ecm_history,
        'inflam_history': inflam_history,
        'oxy_history': oxy_history,
        'grid': grid,
        'n_steps': n_steps,
    }


# ============================================================
# 4. 可视化
# ============================================================

def plot_abm_grid(
    snapshot: Dict[str, Any],
    grid_size: int,
    figsize: Tuple[int, int] = (16, 5),
    output_path: Optional[str] = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    绘制网格状态快照:
    ECM 热图 | 炎症热图 | Agent 分布

    Parameters
    ----------
    snapshot : dict
        网格快照 (来自 run_abm_simulation 的 grid_history)
    grid_size : int
    figsize : tuple
    output_path : str or None
    dpi : int

    Returns
    -------
    plt.Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # 1. ECM 热图
    ax = axes[0]
    im1 = ax.imshow(snapshot['ecm'], cmap='YlGn', vmin=0, vmax=1, interpolation='bilinear')
    plt.colorbar(im1, ax=ax, shrink=0.8, label='ECM 浓度')
    ax.set_title(f'ECM 分布 (Step {snapshot["step"]})', fontsize=11, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # 2. 炎症热图
    ax = axes[1]
    im2 = ax.imshow(snapshot['inflammation'], cmap='Reds', vmin=0, vmax=1, interpolation='bilinear')
    plt.colorbar(im2, ax=ax, shrink=0.8, label='炎症因子')
    ax.set_title(f'炎症分布 (Step {snapshot["step"]})', fontsize=11, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # 3. Agent 分布
    ax = axes[2]
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_title(f'细胞分布 (Step {snapshot["step"]})', fontsize=11, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # Agent 类型颜色
    type_colors = {
        'NP_cell': '#3498DB',
        'macrophage': '#E74C3C',
        'fibroblast': '#27AE60',
    }
    state_markers = {
        'normal': 'o',
        'stressed': 's',
        'apoptotic': 'x',
        'senescent': 'D',
        'dead': '.',
    }

    for agent_data in snapshot['agents']:
        if not agent_data['alive']:
            continue
        color = type_colors.get(agent_data['type'], '#95A5A6')
        marker = state_markers.get(agent_data['state'], 'o')
        alpha = 0.5 if agent_data['state'] == 'senescent' else 0.8
        size = 80 if agent_data['type'] == 'NP_cell' else 50
        ax.scatter(agent_data['x'], agent_data['y'],
                   c=color, marker=marker, s=size,
                   alpha=alpha, edgecolors='black', linewidth=0.3)

    # 图例
    legend_elements = []
    for t, c in type_colors.items():
        legend_elements.append(mpatches.Patch(color=c, label=t, alpha=0.7))
    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                       markerfacecolor='grey', markersize=6, label='normal'))
    legend_elements.append(plt.Line2D([0], [0], marker='s', color='w',
                                       markerfacecolor='grey', markersize=6, label='stressed'))
    legend_elements.append(plt.Line2D([0], [0], marker='x', color='grey',
                                       markersize=6, label='apoptotic'))
    legend_elements.append(plt.Line2D([0], [0], marker='D', color='w',
                                       markerfacecolor='grey', markersize=6, label='senescent'))
    ax.legend(handles=legend_elements, fontsize=6, loc='upper right')

    fig.suptitle('NP 微环境 ABM 仿真 — 网格状态',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"[✓] ABM 网格快照: {output_path}")

    return fig


def animate_abm(
    results: Dict[str, Any],
    grid_size: int,
    figsize: Tuple[int, int] = (16, 5),
    output_path: Optional[str] = None,
    dpi: int = 100,
) -> List[plt.Figure]:
    """
    生成多步时序热图 (各时间点快照)。

    Parameters
    ----------
    results : dict
        run_abm_simulation() 的输出
    grid_size : int
    figsize : tuple
    output_path : str or None
    dpi : int

    Returns
    -------
    list of plt.Figure
    """
    figures = []
    for snapshot in results['grid_history']:
        fig = plot_abm_grid(
            snapshot, grid_size,
            figsize=figsize,
            output_path=(
                f"{output_path.replace('.png', '')}_step{snapshot['step']:04d}.png"
                if output_path else None
            ),
            dpi=dpi,
        )
        figures.append(fig)
        plt.close(fig)

    print(f"  [✓] 生成了 {len(figures)} 帧时序快照")
    return figures


def plot_abm_timeseries(
    results: Dict[str, Any],
    figsize: Tuple[int, int] = (14, 10),
    output_path: Optional[str] = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    绘制各类细胞数量和微环境参数的时间变化曲线。

    Parameters
    ----------
    results : dict
        run_abm_simulation() 的输出
    figsize : tuple
    output_path : str or None
    dpi : int

    Returns
    -------
    plt.Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    # 1. 种群数量变化
    ax = axes[0]
    colors = {'NP_cell': '#3498DB', 'macrophage': '#E74C3C', 'fibroblast': '#27AE60'}
    for ct, hist in results['population_history'].items():
        steps = [h[0] for h in hist]
        counts = [h[1] for h in hist]
        ax.plot(steps, counts, color=colors.get(ct, '#95A5A6'),
                label=ct, linewidth=2, marker='o', markersize=4)
    ax.set_xlabel('时间步', fontsize=10)
    ax.set_ylabel('细胞数量', fontsize=10)
    ax.set_title('细胞种群数量变化', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 2. NP 细胞状态比例
    ax = axes[1]
    state_colors = {'normal': '#2ECC71', 'stressed': '#F39C12',
                    'apoptotic': '#E74C3C', 'senescent': '#8E44AD'}
    for state, hist in results['state_history'].items():
        steps = [h[0] for h in hist]
        counts = [h[1] for h in hist]
        ax.plot(steps, counts, color=state_colors.get(state, '#95A5A6'),
                label=state, linewidth=2, marker='s', markersize=4)
    ax.set_xlabel('时间步', fontsize=10)
    ax.set_ylabel('NP 细胞数量', fontsize=10)
    ax.set_title('NP 细胞状态变化', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 3. ECM 和炎症变化
    ax = axes[2]
    for hist, label, color in [
        (results['ecm_history'], 'ECM 均值', '#27AE60'),
        (results['inflam_history'], '炎症均值', '#E74C3C'),
    ]:
        steps = [h[0] for h in hist]
        vals = [h[1] for h in hist]
        ax.plot(steps, vals, color=color, label=label, linewidth=2)
    ax.set_xlabel('时间步', fontsize=10)
    ax.set_ylabel('浓度', fontsize=10)
    ax.set_title('微环境参数变化', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 4. 氧浓度
    ax = axes[3]
    steps = [h[0] for h in results['oxy_history']]
    vals = [h[1] for h in results['oxy_history']]
    ax.plot(steps, vals, color='#1ABC9C', linewidth=2, label='氧浓度')
    ax.axhline(0.3, color='red', linestyle='--', alpha=0.5, label='缺氧阈值 (0.3)')
    ax.set_xlabel('时间步', fontsize=10)
    ax.set_ylabel('氧浓度', fontsize=10)
    ax.set_title('氧浓度变化', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.suptitle('ABM 仿真定量输出', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"[✓] ABM 时序曲线: {output_path}")

    return fig


# ============================================================
# 5. 一键运行全部
# ============================================================

def run_full_abm_pipeline(
    grid_size: int = 20,
    n_steps: int = 80,
    output_dir: str = './output',
    dpi: int = 150,
) -> Dict[str, Any]:
    """
    运行完整 ABM 仿真管道:
    1. 正常状态仿真
    2. 退变状态仿真
    3. 网格快照
    4. 时序分析

    Parameters
    ----------
    grid_size : int
    n_steps : int
    output_dir : str
    dpi : int

    Returns
    -------
    dict
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 50)
    print("🦴 ABM 微环境仿真管道")
    print("=" * 50)

    # === 正常仿真 ===
    print(f"\n[1/4] 正常 NP 微环境仿真 (grid={grid_size}×{grid_size}, {n_steps}步)...")
    normal_result = run_abm_simulation(
        grid_size=grid_size, n_steps=n_steps, degenerative=False, seed=42
    )

    # 网格快照 (最后一步)
    fig1 = plot_abm_grid(
        normal_result['grid_history'][-1], grid_size,
        output_path=os.path.join(output_dir, 'abm_normal_grid.png'),
        dpi=dpi
    )
    print(f"  → 正常网格快照已保存")

    # 时序曲线
    fig2 = plot_abm_timeseries(
        normal_result,
        output_path=os.path.join(output_dir, 'abm_normal_timeseries.png'),
        dpi=dpi
    )
    print(f"  → 正常时序曲线已保存")

    # === 退变仿真 ===
    print(f"\n[2/4] 退变 NP 微环境仿真...")
    degen_result = run_abm_simulation(
        grid_size=grid_size, n_steps=n_steps,
        degenerative=True, inflam_seed=0.5, mechanical_overload=0.6,
        seed=43
    )

    fig3 = plot_abm_grid(
        degen_result['grid_history'][-1], grid_size,
        output_path=os.path.join(output_dir, 'abm_degen_grid.png'),
        dpi=dpi
    )
    print(f"  → 退变网格快照已保存")

    fig4 = plot_abm_timeseries(
        degen_result,
        output_path=os.path.join(output_dir, 'abm_degen_timeseries.png'),
        dpi=dpi
    )
    print(f"  → 退变时序曲线已保存")

    # === 正常 vs 退变对比 ===
    print(f"\n[3/4] 正常 vs 退变对比...")
    final_np_normal = sum(1 for a in normal_result['agents']
                          if a.cell_type == 'NP_cell' and a.state in ('normal', 'stressed'))
    final_np_degen = sum(1 for a in degen_result['agents']
                          if a.cell_type == 'NP_cell' and a.state in ('normal', 'stressed'))
    print(f"  · 正常仿真: NP 存活 = {final_np_normal}")
    print(f"  · 退变仿真: NP 存活 = {final_np_degen}")

    ecm_normal_end = normal_result['ecm_history'][-1][1] if normal_result['ecm_history'] else 0
    ecm_degen_end = degen_result['ecm_history'][-1][1] if degen_result['ecm_history'] else 0
    print(f"  · 正常 ECM = {ecm_normal_end:.3f} | 退变 ECM = {ecm_degen_end:.3f}")

    # === 多步动画快照 ===
    print(f"\n[4/4] 生成退变仿真时序快照...")
    animate_abm(
        degen_result, grid_size,
        output_path=os.path.join(output_dir, 'abm_degen_anim.png'),
        dpi=100
    )

    print(f"\n✅ ABM 管道完成")
    return {
        'normal_result': normal_result,
        'degen_result': degen_result,
        'output_dir': output_dir,
    }


if __name__ == '__main__':
    run_full_abm_pipeline(grid_size=16, n_steps=40)
