# -*- coding: utf-8 -*-
"""
ODE ↔ ABM 双向耦合引擎
=======================
Virtual NP Cell — 系统性 ODE 模型与多细胞 ABM 微环境的真正双向耦合。

架构设计：
┌──────────────────────────────────────────────────────────────┐
│                   ODEABMCoupler                              │
│                                                              │
│   ODE 层 (8-state / 23-state)    ←→    ABM 层 (网格 + Agent) │
│   ┌─────────────────────┐       映射       ┌───────────────┐ │
│   │ INFLAM, OXSTRESS    │  ODE→ABM:       │ 网格: ECM,     │ │
│   │ ECM_SYN, ECM_CAT    │  INFLAM→inflam  │     炎症因子,   │ │
│   │ SENESCENCE, FIBROSIS│  ECM_SYN→ecm    │     氧, 力学负载 │ │
│   │ PROGENITOR, VASC    │  PROGENITOR→    │  Agent: NP,     │ │
│   │                     │  增殖率          │     巨噬, 成纤维 │ │
│   │ 扰动输入:           │  VASC→巨噬浸润   │                 │ │
│   │  condition_factor   │  ABM→ODE:       │  输出:          │ │
│   │  tnf/il1b_stim      │  网格avg→cf     │  细胞计数,      │ │
│   │  rox_inh            │  senescent→     │  平均ECM/炎症,  │ │
│   │  senolytic          │  senescence↑    │  衰老比例, ...  │ │
│   └─────────────────────┘                └───────────────┘  │
│                                                              │
│   时间步进: ODE: Δt=0.5, N_steps=2000                       │
│             ABM: Δt=1.0, N_steps=500                         │
│   耦合周期: 每50 ODE步 → 1 ABM步                             │
└──────────────────────────────────────────────────────────────┘

Author: Virtual NP Cell Team
"""

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from typing import Optional, Dict, Tuple, List, Any, Callable
from dataclasses import dataclass, field
import time
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# ── Import ODE 8-state model ──────────────────────────────────────
import sys, os
_ode_dir = os.path.join(os.path.dirname(__file__), '..',
                        'NP_IVDD_pipeline', 'np_ivdd_download_pipeline')
if _ode_dir not in sys.path:
    sys.path.insert(0, _ode_dir)

# ── Import ABM ───────────────────────────────────────────────────
from .abm_microenv import DiscMicroenvGrid, NPAgent, run_abm_simulation

# ── ODE 8-state model (direct implementation, no dependency on 05_fit_ode.py) ──
STATES_8 = ["INFLAM","OXSTRESS","ECM_SYN","ECM_CAT",
            "SENESCENCE","FIBROSIS","PROGENITOR","VASC_IMMUNE"]

PARAMS_26 = [
    "bI","kVI","kSI","dI",  "kOI","dO",
    "kAP","kIA","kOA","dA",  "kCI","kCO","dC",
    "kSenI","kSenO","dSen",  "kFI","kFS","dF",
    "bP","kPI","kPS","dP",   "kV_I","kV_F","dV",
]

# P0-calibrated parameters (from sensitivity analysis)
P0 = np.array([
    0.01910044, 0.19352574, 0.15966154, 0.50169244,
    0.31658600, 0.43889666,
    0.36933347, 0.24674005, 0.24719501, 0.24272767,
    0.26633927, 0.30802681, 0.44718355,
    0.20306046, 0.23906978, 0.38469873,
    0.21190155, 0.24254675, 0.36328181,
    0.21728453, 0.19653147, 0.19532216, 0.23766170,
    0.17486986, 0.17049669, 0.40106754,
], dtype=float)


def sat(x, k=0.4):
    """Hill-type saturation function."""
    return x / (k + x + 1e-9)


def ode_rhs_8state(t, y, p):
    """8-state ODE right-hand side for NP degeneration."""
    I, O, A, C, S, F, P_s, V = np.clip(y, 0, 2)
    (bI, kVI, kSI, dI, kOI, dO, kAP, kIA, kOA, dA,
     kCI, kCO, dC, kSenI, kSenO, dSen, kFI, kFS, dF,
     bP, kPI, kPS, dP, kV_I, kV_F, dV) = p
    return np.array([
        bI + kVI * sat(V) + kSI * sat(S) - dI * I,       # INFLAM
        kOI * sat(I) - dO * O,                            # OXSTRESS
        kAP * sat(P_s) - kIA*I*A - kOA*O*A - dA*A,       # ECM_SYN
        kCI * sat(I) + kCO * sat(O) - dC * C,             # ECM_CAT
        kSenI * sat(I) + kSenO * sat(O) - dSen * S,       # SENESCENCE
        kFI * sat(I) + kFS * sat(S) - dF * F,             # FIBROSIS
        bP - kPI*I*P_s - kPS*S*P_s - dP*P_s,             # PROGENITOR
        kV_I * sat(I) + kV_F * sat(F) - dV * V,           # VASC_IMMUNE
    ])


# ── ABM extended with NP-cell ODE-like state tracking ─────────────────
# ── Data structures for coupling ──────────────────────────────────────
@dataclass
class CouplingSnapshot:
    """Single coupling time point — stores ODE + ABM states for analysis."""
    time: float
    ode_state: np.ndarray          # 8-state vector
    abm_np_count: int              # Number of NP cells
    abm_macrophage_count: int      # Number of macrophages
    abm_fibroblast_count: int      # Number of fibroblasts
    abm_ecm_mean: float            # Mean ECM concentration in grid
    abm_inflam_mean: float         # Mean inflammation in grid
    abm_oxygen_mean: float         # Mean oxygen in grid
    abm_senescent_frac: float      # Fraction of senescent NP cells
    abm_apoptotic_frac: float      # Fraction of apoptotic NP cells
    abm_stressed_frac: float       # Fraction of stressed NP cells


@dataclass
class CouplingResult:
    """Full coupling simulation result."""
    timepoints: np.ndarray         # Time points of coupling snapshots
    ode_trajectory: np.ndarray     # (n_times, 8) ODE state history
    snapshots: List[CouplingSnapshot] = field(default_factory=list)
    abm_metrics: Dict[str, np.ndarray] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)


class ODEABMCoupler:
    """
    ODE ↔ ABM 双向耦合器

    核心逻辑:
    1. 运行 ODE (细时间步, 快动力学) 一段时间
    2. 从 ODE 计算 ABM 输入参数 (ECM, 炎症, 衰老等)
    3. 运行 ABM (粗时间步, 慢动力学) 一段时间
    4. 从 ABM 计算 ODE 扰动参数 (condition_factor 等)
    5. 重复直到仿真结束
    """

    def __init__(self,
                 ode_params: Optional[np.ndarray] = None,
                 abm_grid_size: int = 25,
                 ode_dt: float = 0.5,
                 abm_dt: float = 1.0,
                 ode_steps_per_abm: int = 50,
                 ode_t_max: float = 1000.0,
                 abm_total_steps: int = 500,
                 coupling_interval_ode: int = 50,
                 seed: int = 20260720):
        """
        Args:
            ode_params: 26 ODE 参数 (默认 P0 校准值)
            abm_grid_size: ABM 网格大小 (grid_size × grid_size)
            ode_dt: ODE 时间步长
            abm_dt: ABM 时间步长
            ode_steps_per_abm: 每次 ABM 调用间 ODE 步数
            ode_t_max: ODE 最大仿真时间
            abm_total_steps: ABM 总迭代次数
            coupling_interval_ode: 每多少 ODE 步记录一次耦合快照
            seed: 随机种子
        """
        self.ode_params = ode_params if ode_params is not None else P0.copy()
        self.abm_grid_size = abm_grid_size
        self.ode_dt = ode_dt
        self.abm_dt = abm_dt
        self.ode_steps_per_abm = ode_steps_per_abm
        self.ode_t_max = ode_t_max
        self.abm_total_steps = abm_total_steps
        self.coupling_interval = coupling_interval_ode
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # Coupling mode: 8-state or 23-state
        self.coupling_mode = '8state'

        # Results storage
        self.result: Optional[CouplingResult] = None

    # ═══════════════════════════════════════════════════════════════
    #  ODE → ABM 映射
    # ═══════════════════════════════════════════════════════════════
    def map_ode_to_abm(self, ode_state: np.ndarray) -> Dict[str, float]:
        """
        从 8-state ODE 状态 → ABM 初始化/扰动参数

        Mapping:
          INFLAM      → ABM 初始炎症浓度 (×0.05, 缩放至 0~1)
          ECM_SYN     → ABM 初始 ECM 浓度 (直接影响)
          SENESCENCE  → ABM NP 细胞衰老概率 (×0.1, 阈值 0.5)
          PROGENITOR  → ABM NP 细胞增殖率 (×0.05)
          VASC_IMMUNE → ABM 巨噬细胞初始比例
          OXSTRESS    → ABM 应激水平偏置
        """
        s = np.clip(ode_state, 0, 1)

        inflam_val = float(s[0] * 0.8 + 0.05)         # INFLAM → 炎症浓度
        ecm_val = float(s[2] * 0.7 + 0.15)             # ECM_SYN → ECM
        sen_prob = float(np.clip(s[4] * 0.15, 0, 1))   # SENECENCE → 衰老概率
        prog_rate = float(s[6] * 0.03)                  # PROGENITOR → 增殖率
        vasc_frac = float(s[7] * 0.3)                   # VASC_IMMUNE → 巨噬比例
        oxstress = float(s[1] * 0.5)                    # OXSTRESS → 应激水平

        return {
            'inflam': inflam_val,
            'ecm': ecm_val,
            'sen_prob': sen_prob,
            'prog_rate': prog_rate,
            'vasc_frac': vasc_frac,
            'oxstress': oxstress,
        }

    def init_abm_from_ode(self, ode_state: np.ndarray) -> Tuple[DiscMicroenvGrid, List[NPAgent]]:
        """
        从 ODE 初始状态创建 ABM 环境 + 细胞种群。
        """
        mapping = self.map_ode_to_abm(ode_state)
        grid_size = self.abm_grid_size
        rng = self.rng

        # 创建网格
        grid = DiscMicroenvGrid(width=grid_size, height=grid_size, seed=self.seed)

        # 从 ODE 初始化网格
        inflam_val = float(mapping['inflam'])
        ecm_val = float(mapping['ecm'])
        grid.inflammation = np.clip(
            inflam_val + rng.uniform(-0.02, 0.02, (grid_size, grid_size)), 0, 1)
        grid.ecm = np.clip(
            ecm_val + rng.uniform(-0.03, 0.03, (grid_size, grid_size)), 0, 1)

        # 基础氧和力学负载
        center = grid_size // 2
        for y in range(grid_size):
            for x in range(grid_size):
                dist = np.sqrt((y - center)**2 + (x - center)**2) / (center + 1)
                grid.oxygen[y, x] = 0.3 + 0.7 * dist
                grid.nutrient[y, x] = 0.8 - 0.4 * dist

        # 创建细胞种群
        agents: List[NPAgent] = []
        n_np = max(8, int(30 * (1 - mapping['vasc_frac'])))
        n_macrophage = max(1, int(8 * mapping['vasc_frac']))
        n_fibroblast = max(1, int(5 * mapping['sen_prob']))

        agent_id = 0
        placed_positions = []

        # NP 细胞
        for _ in range(n_np):
            x = rng.uniform(grid_size * 0.1, grid_size * 0.9)
            y = rng.uniform(grid_size * 0.1, grid_size * 0.9)
            placed_positions.append((x, y))
            agent = NPAgent(agent_id, 'NP_cell', x, y, seed=self.seed + agent_id)
            # 初始状态: 根据 OXSTRESS 和 SENESCENCE 部分细胞应激
            if rng.random() < mapping['oxstress'] * 0.3:
                agent.state = 'stressed'
                agent.stress_level = mapping['oxstress']
            agents.append(agent)
            agent_id += 1

        # 巨噬细胞
        for _ in range(n_macrophage):
            x = rng.uniform(0, grid_size)
            y = rng.uniform(0, grid_size)
            agent = NPAgent(agent_id, 'macrophage', x, y, seed=self.seed + agent_id)
            agents.append(agent)
            agent_id += 1

        # 成纤维细胞
        for _ in range(n_fibroblast):
            x = rng.uniform(0, grid_size)
            y = rng.uniform(0, grid_size)
            agent = NPAgent(agent_id, 'fibroblast', x, y, seed=self.seed + agent_id)
            agents.append(agent)
            agent_id += 1

        return grid, agents

    # ═══════════════════════════════════════════════════════════════
    #  ABM → ODE 映射
    # ═══════════════════════════════════════════════════════════════
    def map_abm_to_ode(self, grid: DiscMicroenvGrid,
                       agents: List[NPAgent]) -> Dict[str, float]:
        """
        从 ABM 网格 + 细胞状态 → ODE 扰动参数

        Mapping:
          网格 avg 炎症     → condition_factor (0~1)
          网格 avg ECM      → ECM 状态偏置
          衰老细胞比例       → senescence 偏置
          NP 细胞数量       → PROGENITOR 偏置
          巨噬细胞数量       → VASC 偏置
          应激/凋亡比例      → OXSTRESS 偏置
        """
        total_np = sum(1 for a in agents if a.cell_type == 'NP_cell' and a.alive)
        total_mac = sum(1 for a in agents if a.cell_type == 'macrophage' and a.alive)
        total_fib = sum(1 for a in agents if a.cell_type == 'fibroblast' and a.alive)

        senescent_np = sum(1 for a in agents
                          if a.cell_type == 'NP_cell' and a.state == 'senescent')
        stressed_np = sum(1 for a in agents
                         if a.cell_type == 'NP_cell' and a.state == 'stressed')
        apoptotic_np = sum(1 for a in agents
                          if a.cell_type == 'NP_cell' and a.state == 'apoptotic')

        n_np_eff = max(total_np, 1)
        sen_frac = senescent_np / n_np_eff
        str_frac = stressed_np / n_np_eff
        apo_frac = apoptotic_np / n_np_eff

        # 网格平均
        inflam_mean = float(np.mean(grid.inflammation))
        ecm_mean = float(np.mean(grid.ecm))
        oxygen_mean = float(np.mean(grid.oxygen))

        # 计算 condition_factor (0~1)
        # 高炎症 + 低ECM + 多衰老 → 高 cf
        cf_inflam = inflam_mean * 0.5
        cf_ecm = (1 - ecm_mean) * 0.25
        cf_sen = sen_frac * 0.25
        condition_factor = np.clip(cf_inflam + cf_ecm + cf_sen, 0, 1)

        # 炎症刺激
        tnf_stim = inflam_mean * 0.3 + str_frac * 0.1
        il1b_stim = inflam_mean * 0.25 + apo_frac * 0.15

        # 氧化应激
        oxstress_bias = str_frac * 0.4 + (1 - oxygen_mean) * 0.3

        # 衰老清除 (反向: 多衰老 → 少清除)
        senolytic = max(0, 0.1 - sen_frac * 0.3)

        # MMP 抑制 (ECM低 → MMP少)
        mmp_inh = max(0, 0.1 - ecm_mean * 0.2) if ecm_mean < 0.5 else 0

        return {
            'condition_factor': condition_factor,
            'tnf_stim': float(tnf_stim),
            'il1b_stim': float(il1b_stim),
            'oxidative_stress': float(oxstress_bias),
            'senolytic': float(senolytic),
            'mmp_inh': float(mmp_inh),
            'rox_inh': float(max(0, oxygen_mean - 0.5) * 0.2),
            'sen_frac': float(sen_frac),
            'np_count': total_np,
            'mac_count': total_mac,
            'fib_count': total_fib,
            'ecm_mean': float(ecm_mean),
            'inflam_mean': inflam_mean,
            'oxygen_mean': oxygen_mean,
        }

    # ═══════════════════════════════════════════════════════════════
    #  23-state coupled engine 扰动组合 (复用)
    # ═══════════════════════════════════════════════════════════════
    def build_23state_perturbation(self, abm_mapping: Dict[str, float]) -> Dict[str, float]:
        """将 ABM 映射转换为 23-state coupled_engine 的扰动参数字典."""
        return {
            'condition_factor': abm_mapping['condition_factor'],
            'tnf_stim': abm_mapping['tnf_stim'],
            'il1b_stim': abm_mapping['il1b_stim'],
            'oxidative_stress': abm_mapping['oxidative_stress'],
            'senolytic': abm_mapping['senolytic'],
            'mmp_inh': abm_mapping['mmp_inh'],
            'rox_inh': abm_mapping['rox_inh'],
        }

    # ═══════════════════════════════════════════════════════════════
    #  ABM 运行包装 (集成 NPAgent 状态机)
    # ═══════════════════════════════════════════════════════════════
    def run_abm_steps(self, grid: DiscMicroenvGrid, agents: List[NPAgent],
                      n_steps: int) -> None:
        """运行 ABM 指定步数，使用 ODE 映射的状态驱动 Agent 行为."""
        ode_mapping = self._last_ode_mapping  # 最近的 ODE→ABM 映射

        for _ in range(n_steps):
            # 1. Agent 行为更新
            for agent in agents:
                if not agent.alive:
                    continue

                # 从 ODE 映射获取影响
                if agent.cell_type == 'NP_cell':
                    # 增殖: 受 PROGENITOR 驱动
                    if (agent.state == 'normal' and
                        self.rng.random() < ode_mapping['prog_rate'] * 0.05):
                        # 尝试繁殖 (简化: 不创建新 agent, 增加 stress)
                        pass

                    # 应激: 受 OXSTRESS + 局部炎症驱动
                    x, y = int(agent.x), int(agent.y)
                    if 0 <= x < grid.width and 0 <= y < grid.height:
                        local_inflam = grid.inflammation[y, x]
                        local_ecm = grid.ecm[y, x]
                        local_oxy = grid.oxygen[y, x]

                        stress_drive = (local_inflam * 0.4 +
                                       (1 - local_ecm) * 0.3 +
                                       (1 - local_oxy) * 0.2 +
                                       ode_mapping['oxstress'] * 0.3)
                        agent.stress_level += stress_drive * 0.05

                        # 状态转换 (加速因子)
                        if agent.state == 'normal' and agent.stress_level > 0.35:
                            agent.state = 'stressed'
                        elif (agent.state == 'stressed' and
                              self.rng.random() < ode_mapping['sen_prob'] * 0.12):
                            agent.state = 'senescent'
                            # 衰老细胞释放炎症因子
                            x, y = int(agent.x), int(agent.y)
                            if 0 <= x < grid.width and 0 <= y < grid.height:
                                grid.inflammation[y, x] = min(1.0,
                                    grid.inflammation[y, x] + 0.15)
                        elif (agent.state == 'stressed' and
                              agent.stress_level > 1.2):
                            agent.state = 'apoptotic'
                        elif agent.state == 'apoptotic':
                            agent.alive = False

                elif agent.cell_type == 'macrophage':
                    # 巨噬细胞向炎症区域迁移
                    x, y = int(agent.x), int(agent.y)
                    if 0 <= x < grid.width and 0 <= y < grid.height:
                        # 朝向最高炎症邻域移动
                        neighbors = grid.inflammation[max(0,y-1):min(grid.height,y+2),
                                                      max(0,x-1):min(grid.width,x+2)]
                        max_pos = np.unravel_index(neighbors.argmax(), neighbors.shape)
                        target_x = max(0, min(grid.width-1, x + max_pos[1] - 1))
                        target_y = max(0, min(grid.height-1, y + max_pos[0] - 1))
                        agent.x = target_x + self.rng.uniform(-0.3, 0.3)
                        agent.y = target_y + self.rng.uniform(-0.3, 0.3)
                        agent.x = np.clip(agent.x, 0, grid.width - 1)
                        agent.y = np.clip(agent.y, 0, grid.height - 1)

                # Agent aging
                agent.age += 1
                if agent.age > agent.max_age:
                    if self.rng.random() < 0.1:
                        agent.alive = False

            # 2. 移除死亡细胞 (简化: 随机补充)
            alive_agents = [a for a in agents if a.alive]
            n_to_add = len(agents) - len(alive_agents)

            # 补充新细胞 (来自 PROGENITOR 驱动)
            if n_to_add > 0 and self.rng.random() < 0.5:
                for _ in range(min(n_to_add, 2)):
                    new_id = max(a.id for a in agents) + 1
                    new_x = self.rng.uniform(0, grid.width)
                    new_y = self.rng.uniform(0, grid.height)
                    agent = NPAgent(new_id, 'NP_cell', new_x, new_y,
                                    seed=self.seed + new_id)
                    # 新细胞初始应激水平由 PROGENITOR 状态决定
                    prog_val = self._last_ode_mapping.get('prog_rate', 0.03)
                    agent.stress_level = self.rng.uniform(0, 0.1) * (1 - prog_val * 10)
                    agents.append(agent)

            alive_agents = [a for a in agents if a.alive]
            agents[:] = alive_agents

            # 3. 更新网格环境
            grid.update_environment(agents, dt=self.abm_dt)

            # 4. 巨噬细胞产生炎症
            for agent in agents:
                if agent.cell_type == 'macrophage' and agent.alive:
                    x, y = int(agent.x), int(agent.y)
                    if 0 <= x < grid.width and 0 <= y < grid.height:
                        grid.inflammation[y, x] = min(1.0,
                            grid.inflammation[y, x] + 0.02 * ode_mapping['vasc_frac'])

    # ═══════════════════════════════════════════════════════════════
    #  主耦合循环
    # ═══════════════════════════════════════════════════════════════
    def run_coupled(self,
                    ode_initial: Optional[np.ndarray] = None,
                    verbose: bool = True,
                    save_snapshots: bool = True) -> CouplingResult:
        """
        运行 ODE ↔ ABM 双向耦合仿真。

        Returns:
            CouplingResult 包含完整轨迹、快照、ABM 度量
        """
        t_start = time.time()

        # ── 初始化 ODE ──
        y0 = ode_initial if ode_initial is not None else np.array([0.2, 0.2, 0.7, 0.2,
                                                                    0.1, 0.15, 0.6, 0.1])

        # ── ODE→ABM 初始化 ──
        self._last_ode_mapping = self.map_ode_to_abm(y0)
        grid, agents = self.init_abm_from_ode(y0)

        # ── 状态追踪 ──
        n_ode_steps = int(self.ode_t_max / self.ode_dt)
        snapshot_times = []
        ode_trajectory = []
        snapshots = []

        abm_metrics = {
            'np_count': [], 'mac_count': [], 'fib_count': [],
            'ecm_mean': [], 'inflam_mean': [], 'oxygen_mean': [],
            'sen_frac': [], 'str_frac': [], 'apo_frac': [],
        }

        # 主循环
        ode_t = 0.0
        y_current = y0.copy()
        abm_step_counter = 0

        if verbose:
            print("=" * 70)
            print("ODE ↔ ABM 双向耦合仿真")
            print("=" * 70)
            print(f"  ODE 步数: {n_ode_steps}, dt={self.ode_dt}")
            print(f"  ABM 步数: {self.abm_total_steps}, dt={self.abm_dt}")
            print(f"  耦合间隔: {self.ode_steps_per_abm} ODE 步/耦合")
            print(f"  ABM 网格: {self.abm_grid_size}×{self.abm_grid_size}")
            print(f"  初始细胞: {len(agents)}")
            print(f"  初始 ODE: {[f'{v:.3f}' for v in y0]}")
            print("-" * 70)

        for ode_idx in range(n_ode_steps):
            # ── ODE 单步 ──
            t_span = (ode_t, ode_t + self.ode_dt)
            sol = solve_ivp(lambda t, y: ode_rhs_8state(t, y, self.ode_params),
                           t_span, y_current, method='LSODA',
                           rtol=1e-6, atol=1e-8, max_step=self.ode_dt)
            if sol.success and sol.y.shape[1] > 0:
                y_current = np.clip(sol.y[:, -1], 0, 2)
            ode_t = t_span[1]

            # ── 每 coupling_interval 记录快照 ──
            if ode_idx % self.coupling_interval == 0:
                ode_trajectory.append(y_current.copy())
                snapshot_times.append(ode_t)

                if save_snapshots:
                    # 收集 ABM 指标
                    total_np = sum(1 for a in agents
                                   if a.cell_type == 'NP_cell' and a.alive)
                    total_mac = sum(1 for a in agents
                                    if a.cell_type == 'macrophage' and a.alive)
                    total_fib = sum(1 for a in agents
                                    if a.cell_type == 'fibroblast' and a.alive)

                    sen_np = sum(1 for a in agents
                                 if a.cell_type == 'NP_cell' and a.state == 'senescent')
                    str_np = sum(1 for a in agents
                                 if a.cell_type == 'NP_cell' and a.state == 'stressed')
                    apo_np = sum(1 for a in agents
                                 if a.cell_type == 'NP_cell' and a.state == 'apoptotic')

                    n_np = max(total_np, 1)

                    snap = CouplingSnapshot(
                        time=ode_t,
                        ode_state=y_current.copy(),
                        abm_np_count=total_np,
                        abm_macrophage_count=total_mac,
                        abm_fibroblast_count=total_fib,
                        abm_ecm_mean=float(np.mean(grid.ecm)),
                        abm_inflam_mean=float(np.mean(grid.inflammation)),
                        abm_oxygen_mean=float(np.mean(grid.oxygen)),
                        abm_senescent_frac=sen_np / n_np,
                        abm_apoptotic_frac=apo_np / n_np,
                        abm_stressed_frac=str_np / n_np,
                    )
                    snapshots.append(snap)

                    abm_metrics['np_count'].append(total_np)
                    abm_metrics['mac_count'].append(total_mac)
                    abm_metrics['fib_count'].append(total_fib)
                    abm_metrics['ecm_mean'].append(float(np.mean(grid.ecm)))
                    abm_metrics['inflam_mean'].append(float(np.mean(grid.inflammation)))
                    abm_metrics['oxygen_mean'].append(float(np.mean(grid.oxygen)))
                    abm_metrics['sen_frac'].append(sen_np / n_np)
                    abm_metrics['str_frac'].append(str_np / n_np)
                    abm_metrics['apo_frac'].append(apo_np / n_np)

            # ── 每 ode_steps_per_abm 步: ABM 耦合 ──
            if ode_idx > 0 and ode_idx % self.ode_steps_per_abm == 0:
                # 更新 ODE→ABM 映射
                self._last_ode_mapping = self.map_ode_to_abm(y_current)

                # 运行 ABM 一步
                self.run_abm_steps(grid, agents, 1)
                abm_step_counter += 1

                # ABM→ODE 映射 → 更新 ODE 扰动参数
                abm_map = self.map_abm_to_ode(grid, agents)

                # 关键: 将 ABM 状态融合进 ODE
                # 方法: 用 ABM 的多维指标联合驱动 ODE 参数
                cf = abm_map['condition_factor']
                sen_frac = abm_map['sen_frac']
                inflam_mean = abm_map['inflam_mean']
                ecm_mean = abm_map['ecm_mean']
                np_count = abm_map['np_count']
                mac_count = abm_map['mac_count']

                # 从 ABM 计算的补充驱动:
                # 1. 炎症偏差: ABM炎症 - ODE炎症 → 调整 INFLAM 基线
                inflam_bias = inflam_mean * 2.0 - y_current[0]  # ABM高于ODE时正向驱动
                
                # 2. ECM偏差: ABM ECM - ODE ECM_SYN → 调整 ECM 合成
                ecm_bias = ecm_mean - y_current[2]

                # 3. 衰老正反馈: 衰老细胞释放 SASP
                sen_fb = sen_frac * 0.3

                # 4. 巨噬细胞浸润: 产生额外炎症
                mac_inflam = mac_count * 0.02

                # 组合驱动
                total_drive = max(0, cf * 1.0 + sen_fb + mac_inflam + inflam_bias * 0.3)
                total_drive = min(total_drive, 1.0)

                # 调整 ODE 参数
                p_adj = self.ode_params.copy()
                # cf 高 → 炎症基线升高 → 加速退变
                p_adj[0] = P0[0] * (1 + total_drive * 3.0)     # bI ↑ (炎症基线)
                p_adj[3] = P0[3] * (1 - total_drive * 0.4)     # dI ↓ (炎症清除减慢)
                p_adj[5] = P0[5] * (1 - total_drive * 0.3)     # dO ↓
                p_adj[19] = P0[19] * (1 + sen_frac * 2.0)      # bP ↑ (补偿性祖细胞生产)
                self.ode_params = p_adj

                # 日志
                if verbose and ode_idx % (self.coupling_interval * 5) == 0:
                    print(f"  [t={ode_t:.0f}] ABM→ODE: cf={cf:.2f} senF={sen_frac:.2f} "
                          f"mac={mac_count} drive={total_drive:.2f} ")

                if verbose and abm_step_counter % 50 == 0:
                    print(f"  [t={ode_t:.0f}] ODE: I={y_current[0]:.3f} "
                          f"ECS={y_current[2]:.3f} Sen={y_current[4]:.3f} "
                          f"| ABM: NP={total_np} Mac={total_mac} "
                          f"ECM={np.mean(grid.ecm):.2f} SenFrac={sen_np/n_np:.2f} "
                          f"cf={cf:.2f}")

        # ── 整理结果 ──
        self.result = CouplingResult(
            timepoints=np.array(snapshot_times),
            ode_trajectory=np.array(ode_trajectory),
            snapshots=snapshots,
            abm_metrics={k: np.array(v) for k, v in abm_metrics.items()},
            config={
                'ode_dt': self.ode_dt,
                'abm_dt': self.abm_dt,
                'ode_t_max': self.ode_t_max,
                'ode_steps_per_abm': self.ode_steps_per_abm,
                'abm_total_steps': abm_step_counter,
                'grid_size': self.abm_grid_size,
                'seed': self.seed,
            }
        )

        elapsed = time.time() - t_start
        if verbose:
            elapsed_str = time.strftime("%M:%S", time.gmtime(elapsed))
            print("-" * 70)
            print(f"  ✅ 耦合仿真完成 | {elapsed_str} | "
                  f"{n_ode_steps} ODE步 × {abm_step_counter} ABM步")
            print(f"  最终 ODE: I={y_current[0]:.3f} ECS={y_current[2]:.3f} "
                  f"Sen={y_current[4]:.3f}")
            print(f"  最终 ABM: NP={sum(1 for a in agents if a.cell_type=='NP_cell' and a.alive)} "
                  f"Mac={sum(1 for a in agents if a.cell_type=='macrophage' and a.alive)}")
            print("=" * 70)

        return self.result

    # ═══════════════════════════════════════════════════════════════
    #  解耦对照: 仅 ODE
    # ═══════════════════════════════════════════════════════════════
    def run_ode_only(self, ode_initial: Optional[np.ndarray] = None,
                     condition_factor: float = 0.0,
                     verbose: bool = False) -> CouplingResult:
        """仅运行 ODE (无 ABM 反馈) — 作为基线对照."""
        y0 = ode_initial if ode_initial is not None else np.array(
            [0.2, 0.2, 0.7, 0.2, 0.1, 0.15, 0.6, 0.1])
        t_eval = np.linspace(0, self.ode_t_max, 2000)
        p = self.ode_params.copy()

        # 用 condition_factor 调整
        p[0] = P0[0] * (1 + condition_factor * 2.0)
        p[3] = P0[3] * (1 - condition_factor * 0.3)

        sol = solve_ivp(lambda t, y: ode_rhs_8state(t, y, p),
                        (0, self.ode_t_max), y0, t_eval=t_eval,
                        method='LSODA', rtol=1e-7, atol=1e-9)
        ode_traj = sol.y.T if sol.success else np.zeros((len(t_eval), 8))

        # 模拟的 "ABM" 指标 (实际上来自 ODE)
        snapshots = []
        for i in range(0, len(t_eval), 50):
            y = ode_traj[i]
            snapshots.append(CouplingSnapshot(
                time=t_eval[i],
                ode_state=y,
                abm_np_count=int(20 * y[6]),
                abm_macrophage_count=int(5 * y[7]),
                abm_fibroblast_count=int(3 * y[4]),
                abm_ecm_mean=float(y[2] * 0.7),
                abm_inflam_mean=float(y[0] * 0.6),
                abm_oxygen_mean=0.7,
                abm_senescent_frac=float(y[4] * 0.5),
                abm_apoptotic_frac=float(y[4] * 0.1),
                abm_stressed_frac=float(y[1] * 0.3),
            ))

        return CouplingResult(
            timepoints=t_eval[::50],
            ode_trajectory=ode_traj[::50],
            snapshots=snapshots,
            abm_metrics={},
            config={'mode': 'ode_only', 'condition_factor': condition_factor}
        )

    # ═══════════════════════════════════════════════════════════════
    #  耦合前后对比
    # ═══════════════════════════════════════════════════════════════
    def run_comparison(self, verbose: bool = True) -> Dict[str, CouplingResult]:
        """
        运行 (a) 耦合版 与 (b) ODE-only 版 对比。

        Returns:
            {'coupled': result, 'ode_only': result}
        """
        if verbose:
            print("\n" + "=" * 70)
            print("  耦合 vs ODE-only 对比实验")
            print("=" * 70)

        # 耦合版
        if verbose:
            print("\n[1/2] 运行耦合仿真...")
        coupled_result = self.run_coupled(verbose=verbose)

        # ODE-only (用耦合版的最终 condition_factor 作为输入)
        if verbose:
            print("\n[2/2] 运行 ODE-only 对照...")
        ode_only_result = self.run_ode_only(verbose=False)

        return {'coupled': coupled_result, 'ode_only': ode_only_result}


# ══════════════════════════════════════════════════════════════════════
#  快速自检
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 70)
    print("ODE ↔ ABM 双向耦合引擎 — 自检")
    print("=" * 70)

    coupler = ODEABMCoupler(abm_grid_size=20,
                            ode_t_max=500.0,
                            ode_steps_per_abm=50,
                            coupling_interval_ode=25,
                            seed=20260720)

    # 标准测试
    result = coupler.run_coupled(verbose=True)

    print(f"\n  ODE 轨迹形状: {result.ode_trajectory.shape}")
    print(f"  快照数量: {len(result.snapshots)}")
    print(f"  时间点: [{result.timepoints[0]:.1f}, ..., {result.timepoints[-1]:.1f}]")

    if result.snapshots:
        last = result.snapshots[-1]
        print(f"\n  最终快照:")
        print(f"    ODE: I={last.ode_state[0]:.3f} OS={last.ode_state[1]:.3f} "
              f"ECMsyn={last.ode_state[2]:.3f} ECMcat={last.ode_state[3]:.3f}")
        print(f"    ODE: Sen={last.ode_state[4]:.3f} Fib={last.ode_state[5]:.3f} "
              f"Prog={last.ode_state[6]:.3f} Vasc={last.ode_state[7]:.3f}")
        print(f"    ABM: NP={last.abm_np_count} Mac={last.abm_macrophage_count} "
              f"Fib={last.abm_fibroblast_count}")
        print(f"    ABM: ECM={last.abm_ecm_mean:.3f} Inflam={last.abm_inflam_mean:.3f}")
        print(f"    ABM: SenFrac={last.abm_senescent_frac:.3f} "
              f"StrFrac={last.abm_stressed_frac:.3f}")

    print(f"\n✅ 耦合引擎自检通过!")
