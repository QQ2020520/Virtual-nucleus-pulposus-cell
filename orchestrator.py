"""
Virtual NP Cell — 虚拟髓核细胞主系统 v3.0
Orchestrator: 智能调度 + 功能路由 + 多模块编排
==================================================
升级内容:
  - 延迟加载所有 v2.0 新模块 (mechanotransduction, metabolism, 
    senescence, spatial, epigenetics)
  - 尝试导入 coupled_engine 和 drug_screening (向后兼容)
  - run_module() 动态路由
  - run_pipeline() 多模块顺序执行
  - quick_report() 快速Markdown报告
"""

import sys
import os
import textwrap
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ==================== 核心模块导入 (try/except 确保向后兼容) ====================

from core.np_knowledge_base import NP_KNOWLEDGE_BASE
import numpy as np

# v1.0 模块
try:
    from simulation.signaling import NPSignalingModel
    _HAS_SIGNALING = True
except ImportError:
    NPSignalingModel = None
    _HAS_SIGNALING = False

try:
    from simulation.ecm_model import ECMDegradationModel
    _HAS_ECM = True
except ImportError:
    ECMDegradationModel = None
    _HAS_ECM = False

# v2.0 新模块 — MRTF-A 力传导
try:
    from simulation.mechanotransduction import MRTFAMechanotransductionModel
    _HAS_MECHANO = True
except ImportError:
    MRTFAMechanotransductionModel = None
    _HAS_MECHANO = False

# v2.0 新模块 — 代谢
try:
    from simulation.metabolism_model import NPMetabolismModel
    _HAS_METABOLISM = True
except ImportError:
    NPMetabolismModel = None
    _HAS_METABOLISM = False

# v2.0 新模块 — 衰老
try:
    from simulation.senescence_model import NPSenescenceModel
    _HAS_SENESCENCE = True
except ImportError:
    NPSenescenceModel = None
    _HAS_SENESCENCE = False

# v2.0 新模块 — 空间转录组
try:
    from analysis.spatial_transcriptomics import SpatialTranscriptomics
    _HAS_SPATIAL = True
except ImportError:
    SpatialTranscriptomics = None
    _HAS_SPATIAL = False

# v2.0 新模块 — m6A 表观
try:
    from regulation.epigenetics_m6a_model import M6AEpigeneticModel
    _HAS_M6A = True
except ImportError:
    M6AEpigeneticModel = None
    _HAS_M6A = False

# v2.0 可选模块 — 耦合引擎
try:
    from simulation.coupled_engine import NPCoupledModel
    _HAS_COUPLED = True
except (ImportError, ModuleNotFoundError):
    NPCoupledModel = None
    _HAS_COUPLED = False

# v2.0 可选模块 — 虚拟药物筛选
try:
    from analysis.drug_screening import VirtualDrugScreening
    _HAS_DRUG = True
except (ImportError, ModuleNotFoundError):
    VirtualDrugScreening = None
    _HAS_DRUG = False

# v3.0 新模块 — 线粒体动力学
try:
    from simulation.mitochondrial_dynamics import MitochondrialDynamicsModel
    _HAS_MITO = True
except (ImportError, ModuleNotFoundError):
    MitochondrialDynamicsModel = None
    _HAS_MITO = False

# v3.0 新模块 — 亚细胞区室
try:
    from simulation.subcellular_compartments import SubcellularCompartmentsModel
    _HAS_SUBCELL = True
except (ImportError, ModuleNotFoundError):
    SubcellularCompartmentsModel = None
    _HAS_SUBCELL = False

# v3.0 新模块 — mRNA/相分离
try:
    from regulation.rna_dynamics import RNADynamicsModel
    _HAS_RNA = True
except (ImportError, ModuleNotFoundError):
    RNADynamicsModel = None
    _HAS_RNA = False

    VirtualDrugScreening = None
    _HAS_DRUG = False


class VirtualNPCell:
    """
    虚拟髓核细胞 — 主入口 v3.0

    功能:
    1. 差异表达分析 & 火山图
    2. 基因表达热图
    3. 生物标志物预测 & 趋势分析
    4. 信号通路 ODE 仿真
    5. ECM 代谢动力学模型
    6. 知识问答 (NP 生物学)
    --- v2.0 新增 ---
    7. MRTF-A 机械力传导模型
    8. NP 代谢可塑性模型
    9. NP 细胞衰老多维度 ODE 模型
    10. m6A 表观转录组调控模型
    11. IVD 空间转录组模拟
    12. 多尺度耦合仿真 (可选)
    13. 虚拟药物筛选 (可选)
    """

    def __init__(self):
        """延迟加载所有模块，防止 ImportError"""
        self.kb = NP_KNOWLEDGE_BASE
        self.de_results = None

        # v1.0 模块 (延迟实例化)
        self.signal_model = None
        self.ecm_model = None

        # v2.0 模块 (延迟实例化)
        self.mechano_model = None
        self.metabolism_model = None
        self.senescence_model = None
        self.spatial_model = None
        self.m6a_model = None
        self.coupled_model = None
        self.drug_screening = None
        self.mito_model = None
        self.subcellular_model = None
        self.rna_model = None

        # 运行缓存
        self._module_results = {}

    def _lazy_load(self, attr: str, cls, *args, **kwargs):
        """延迟加载模式：仅当使用时才实例化"""
        if getattr(self, attr) is None and cls is not None:
            setattr(self, attr, cls(*args, **kwargs))
        return getattr(self, attr)

    # ==================== 模块路由 ====================

    def run_module(self, module_name: str, **kwargs) -> dict:
        """
        根据名称动态路由到对应模块执行

        Args:
            module_name: 模块名称
                'signaling'         — NPSignalingModel (信号通路ODE)
                'ecm'               — ECMDegradationModel (ECM代谢动力学)
                'mechanotransduction' — MRTFAMechanotransductionModel (力传导)
                'metabolism'        — NPMetabolismModel (代谢可塑性)
                'senescence'        — NPSenescenceModel (细胞衰老)
                'spatial'           — SpatialTranscriptomics (空间转录组)
                'm6a'               — M6AEpigeneticModel (m6A表观)
                'couple'            — NPCoupledModel (多尺度耦合, 可选)
                'drug_screen'       — VirtualDrugScreening (虚拟药物筛选, 可选)\n                'mitochondrial'     — MitochondrialDynamicsModel (线粒体动力学)\n                'subcellular'       — SubcellularCompartmentsModel (亚细胞区室)\n                'rna_dynamics'      — RNADynamicsModel (mRNA/相分离)

        Returns:
            dict: { 'module': name, 'success': bool, 'result': ..., 'error': str }
        """
        module_map = {
            # v1.0 核心
            'signaling': ('signal_model', NPSignalingModel, _HAS_SIGNALING),
            'ecm': ('ecm_model', ECMDegradationModel, _HAS_ECM),
            # v2.0 新增
            'mechanotransduction': ('mechano_model', MRTFAMechanotransductionModel, _HAS_MECHANO),
            'metabolism': ('metabolism_model', NPMetabolismModel, _HAS_METABOLISM),
            'senescence': ('senescence_model', NPSenescenceModel, _HAS_SENESCENCE),
            'spatial': ('spatial_model', SpatialTranscriptomics, _HAS_SPATIAL),
            'm6a': ('m6a_model', M6AEpigeneticModel, _HAS_M6A),
            # 可选
            'couple': ('coupled_model', NPCoupledModel, _HAS_COUPLED),
            'drug_screen': ('drug_screening', VirtualDrugScreening, _HAS_DRUG),
            # v3.0 亚细胞模块
            'mitochondrial': ('mito_model', MitochondrialDynamicsModel, _HAS_MITO),
            'subcellular': ('subcellular_model', SubcellularCompartmentsModel, _HAS_SUBCELL),
            'rna_dynamics': ('rna_model', RNADynamicsModel, _HAS_RNA),
        }

        if module_name not in module_map:
            return {
                'module': module_name,
                'success': False,
                'error': f"未知模块: '{module_name}'. 可选: {list(module_map.keys())}"
            }

        attr, cls, available = module_map[module_name]
        if not available or cls is None:
            return {
                'module': module_name,
                'success': False,
                'error': f"模块 '{module_name}' 不可用 (可能未安装依赖)"
            }

        try:
            model = self._lazy_load(attr, cls)

            # ---- 各模块的特定调用参数 ----
            if module_name == 'signaling':
                perturbations = kwargs.get('perturbation', {})
                if perturbations:
                    t, y = model.simulate(
                        t_span=kwargs.get('t_span', (0, 100)),
                        n_points=kwargs.get('n_points', 500),
                        perturbations=perturbations,
                    )
                else:
                    t, y = model.simulate()
                result = {'t': t, 'y': y, 'model': model}

            elif module_name == 'ecm':
                t, y = model.simulate(
                    t_span=kwargs.get('t_span', (0, 400)),
                    n_points=kwargs.get('n_points', 500),
                    perturbation=kwargs.get('perturbation', {}),
                )
                result = {'t': t, 'y': y, 'model': model}

            elif module_name == 'mechanotransduction':
                stiffness = kwargs.get('stiffness_level', 1.0)
                perturb = kwargs.get('perturbation', None)
                t, y = model.simulate(
                    stiffness_level=stiffness,
                    perturbation=perturb,
                )
                metrics = model.get_steady_state_metrics(y[:, -1], stiffness)
                result = {'t': t, 'y': y, 'metrics': metrics, 'model': model}

            elif module_name == 'metabolism':
                oxygen = kwargs.get('oxygen', 0.05)
                glucose = kwargs.get('glucose', 1.0)
                perturbation = kwargs.get('perturbation', None)
                sim_result = model.simulate(
                    oxygen_level=oxygen,
                    glucose_level=glucose,
                    perturbation=perturbation,
                )
                profile = model.get_metabolic_profile(sim_result)
                result = {**sim_result, 'profile': profile, 'model': model}

            elif module_name == 'senescence':
                stress = kwargs.get('stress_level', 1.0)
                perturb = kwargs.get('perturbation', None)
                t, y = model.simulate(
                    stress_level=stress,
                    perturbation=perturb,
                )
                score = model.compute_senescence_score(y[:, -1])
                result = {
                    't': t, 'y': y,
                    'senescence_score': score,
                    'state': model.get_state_dict(y[:, -1]),
                    'model': model,
                }

            elif module_name == 'spatial':
                st = self._lazy_load('spatial_model', SpatialTranscriptomics)
                grid_size = kwargs.get('grid_size', 50)
                n_spots = kwargs.get('n_spots', 2000)
                pipeline_result = st.run_pipeline(
                    grid_size=grid_size,
                    n_spots=n_spots,
                    save_plot=False,
                )
                summary = st.summarize()
                result = {**pipeline_result, 'summary': summary, 'model': st}

            elif module_name == 'm6a':
                perturbation = kwargs.get('perturbation', None)
                sim_result = model.simulate(
                    perturbation=perturbation,
                )
                y_final = sim_result['y'][:, -1]
                steady = {sim_result['var_names'][i]: float(y_final[i])
                          for i in range(len(sim_result['var_names']))}
                result = {**sim_result, 'steady_state': steady, 'model': model}

            elif module_name == 'mitochondrial':
                if hasattr(model, 'simulate'):
                    pert = kwargs.get('perturbation', None)
                    sim_result = model.simulate(perturbation=pert)
                    yf = sim_result['y'][:, -1]
                    state = model.get_mito_state(yf)
                    metrics = model.get_steady_state_metrics(yf)
                    result = {
                        'result': sim_result, 'y': sim_result['y'], 't': sim_result['t'],
                        'mito_state': state, 'metrics': metrics, 'model': model,
                    }
                else:
                    result = {'info': 'mitochondrial module loaded but simulate() not available', 'model': model}

            elif module_name == 'subcellular':
                if hasattr(model, 'simulate'):
                    pert = kwargs.get('perturbation', None)
                    sim_result = model.simulate(perturbation=pert)
                    state = model.get_subcellular_state(sim_result['y'][:, -1])
                    result = {
                        'result': sim_result, 'y': sim_result['y'], 't': sim_result['t'],
                        'subcellular_state': state, 'model': model,
                    }
                else:
                    result = {'info': 'subcellular module loaded but simulate() not available', 'model': model}

            elif module_name == 'rna_dynamics':
                if hasattr(model, 'simulate'):
                    pert = kwargs.get('perturbation', None)
                    sim_result = model.simulate(perturbation=pert)
                    state = model.get_transcriptome_state(sim_result['y'][:, -1])
                    result = {
                        'result': sim_result, 'y': sim_result['y'], 't': sim_result['t'],
                        'rnai_state': state, 'model': model,
                    }
                else:
                    result = {'info': 'rna_dynamics module loaded but simulate() not available', 'model': model}

            elif module_name == 'couple':
                # NPCoupledModel.simulate() returns {'normal': {'t','y','steady_state'}, ...}
                sim_type = kwargs.get('sim_type', 'normal')
                if hasattr(model, 'simulate'):
                    sim_result = model.simulate(sim_type=sim_type)
                    y = sim_result['normal']['y']
                    result = {
                        'result': sim_result,
                        'y': y,
                        't': sim_result['normal']['t'],
                        'steady_state': sim_result['normal']['steady_state'],
                        'model': model,
                    }
                else:
                    result = {'info': 'couple module loaded but simulate() not available', 'model': model}

            elif module_name == 'drug_screen':
                # VirtualDrugScreening: screen() + compute_drug_score()
                if kwargs.get('action') == 'default':
                    model.add_default_drugs()
                    screen_res = model.screen(modalities=['senescence'])
                    scores = model.compute_drug_score(screen_res)
                    result = {'scores': scores, 'screen_results': screen_res, 'model': model}
                elif kwargs:
                    screen_res = model.screen(**kwargs)
                    scores = model.compute_drug_score(screen_res) if hasattr(model, 'compute_drug_score') else {}
                    result = {'scores': scores, 'screen_results': screen_res, 'model': model}
                else:
                    model.add_default_drugs()
                    screen_res = model.screen(modalities=['senescence'])
                    scores = model.compute_drug_score(screen_res)
                    result = {'scores': scores, 'screen_results': screen_res, 'model': model}

            else:
                result = {'info': f"{module_name} 执行完成"}

            self._module_results[module_name] = result
            return {'module': module_name, 'success': True, 'result': result}

        except Exception as e:
            return {'module': module_name, 'success': False, 'error': str(e)}

    # ==================== 知识查询 ====================

    def query_knowledge(self, topic: str) -> dict:
        """
        查询 NP 细胞知识库 (扩充支持 v2.0 新主题)

        v2.0 新增主题: '力传导','代谢','衰老','表观','空间','治疗靶点'
        """
        topic_map = {
            # v1.0 原有
            "marker": "marker_genes",
            "标志物": "marker_genes",
            "通路": "signaling_pathways",
            "信号": "signaling_pathways",
            "ecm": "ecm_components",
            "基质": "ecm_components",
            "细胞外基质": "ecm_components",
            "退变": "degeneration_genes",
            "degeneration": "degeneration_genes",
            "炎症": "inflammation_aging_genes",
            "衰老": "inflammation_aging_genes",
            "差异": "differential_genes_vs_af",
            "代谢": "metabolic_features",
            "转录因子": "np_phenotype_TFs",
            "tf": "np_phenotype_TFs",
            "模型": "model_systems",
            # v2.0 新增主题
            "力传导": "mechanotransduction",
            "mechano": "mechanotransduction",
            "aged": "aging_senescence",
            "aging": "aging_senescence",
            "表观": "epigenetics",
            "epigenetics": "epigenetics",
            "空间": "spatial_transcriptomics",
            "治疗靶点": "therapeutic_targets",
            "therapeutic": "therapeutic_targets",
            "线粒体": "mitochondria",
            "mitochondria": "mitochondria",
            "亚细胞": "subcellular",
            "内质网": "er_stress",
            "er": "er_stress",
            "rna": "rna_dynamics",
            "相分离": "phase_separation",
            "mrna": "rna_dynamics",
        }
        for key, val in topic_map.items():
            if key in topic.lower():
                section_data = self.kb.get(val)
                if section_data is not None:
                    return {val: section_data}
                else:
                    return {val: f"[v2.0 主题] 知识库中 '{val}' 的专有章节待补充。可用模拟实验输出相关结果。"}

        # 智能匹配
        results = {}
        query_lower = topic.lower()
        for section, content in self.kb.items():
            section_str = str(content).lower()
            if query_lower in section_str[:200]:
                results[section] = content
                if len(results) >= 3:
                    break

        if not results:
            results = {"info": "未找到匹配主题。可用主题: marker, 通路, ECM, 退变, 炎症, 差异, 代谢, 转录因子, 模型, 力传导, 衰老, 表观, 空间, 治疗靶点"}
        return results

    # ==================== 状态摘要 ====================

    def status_summary(self) -> str:
        """NP 细胞系统状态摘要 (v3.0 — 包含所有v2.0模块)"""
        status_v1 = (
            f"🧬 虚拟髓核细胞 (Virtual NP Cell) 系统状态 — v3.0\n"
            f"{'='*55}\n"
            f"✓ 知识库: {sum(len(v) if isinstance(v, list) else 1 for v in self.kb.values())} 条记录\n"
            f"✓ 信号通路: {len(self.kb['signaling_pathways'])} 条核心通路\n"
            f"✓ 标志基因: {len(self.kb['marker_genes'])} 个\n"
            f"✓ ECM 成分: {len(self.kb['ecm_components'])} 种\n"
            f"✓ 退变基因: {len(self.kb['degeneration_genes'])} 个\n"
            f"✓ 可分析: 火山图 · 热图 · ROC · 时序趋势 · ODE 仿真\n"
        )
        status_v2 = (
            f"\n"
            f"📦 v2.0 模块清单:\n"
            f"  {'[✓]' if _HAS_MECHANO else '[✗]'} MRTF-A 机械力传导    (simulation/mechanotransduction.py)\n"
            f"  {'[✓]' if _HAS_METABOLISM else '[✗]'} NP 代谢可塑性       (simulation/metabolism_model.py)\n"
            f"  {'[✓]' if _HAS_SENESCENCE else '[✗]'} NP 细胞衰老模型     (simulation/senescence_model.py)\n"
            f"  {'[✓]' if _HAS_M6A else '[✗]'} m6A 表观转录组       (regulation/epigenetics_m6a_model.py)\n"
            f"  {'[✓]' if _HAS_SPATIAL else '[✗]'} IVD 空间转录组       (analysis/spatial_transcriptomics.py)\n"
            f"  {'[✓]' if _HAS_COUPLED else '[✗]'} 多尺度耦合仿真       (simulation/coupled_engine.py, 可选)\n"
            f"  {'[✓]' if _HAS_DRUG else '[✗]'} 虚拟药物筛选         (simulation/drug_screening.py, 可选)\n"
        )
        return status_v1 + status_v2

    # ==================== 多模块按序管道 ====================

    def run_pipeline(self, modules: list = None) -> dict:
        """
        按序执行多个模块并收集结果

        Args:
            modules: 模块名列表, 例如 ['signaling', 'mechanotransduction', 'metabolism']
                     默认: 运行所有可用模块

        Returns:
            dict: {module_name: run_module()结果}
        """
        if modules is None:
            all_modules = [
                'signaling', 'ecm',
                'mechanotransduction', 'metabolism',
                'senescence', 'm6a', 'spatial',
            ]
            modules = all_modules

        print(f"🧪 运行管道: {len(modules)} 个模块")
        print(f"   模块顺序: {' → '.join(modules)}\n")

        results = {}
        for i, mod in enumerate(modules, 1):
            print(f"  [{i}/{len(modules)}] 运行 '{mod}'...")
            res = self.run_module(mod)
            status = "✅" if res['success'] else "❌"
            err = f" — {res.get('error', '')}" if not res['success'] else ""
            print(f"    {status} {mod}{err}")
            results[mod] = res

        success_count = sum(1 for r in results.values() if r['success'])
        print(f"\n📊 管道完成: {success_count}/{len(modules)} 成功")
        return results

    # ==================== 快速报告生成 ====================

    def quick_report(self, topic: str = None) -> str:
        """
        快速生成某个主题的 Markdown 格式报告

        Args:
            topic: 主题关键词 (None=综合报告)

        Returns:
            str: Markdown 格式报告
        """
        if topic is None:
            return self._full_report()

        topic_lower = topic.lower()

        # 知识查询
        knowledge = self.query_knowledge(topic)

        # 尝试运行对应模块
        module_name = None
        if any(k in topic_lower for k in ['力传导', 'mechano', 'mrtf']):
            module_name = 'mechanotransduction'
        elif any(k in topic_lower for k in ['代谢', 'metabolism', 'glycolysis']):
            module_name = 'metabolism'
        elif any(k in topic_lower for k in ['衰老', 'senescence', 'aging']):
            module_name = 'senescence'
        elif any(k in topic_lower for k in ['表观', 'epigenetic', 'm6a', 'rna修饰']):
            module_name = 'm6a'
        elif any(k in topic_lower for k in ['空间', 'spatial', 'transcriptom']):
            module_name = 'spatial'
        elif any(k in topic_lower for k in ['信号', 'signaling', '通路']):
            module_name = 'signaling'
        elif any(k in topic_lower for k in ['ecm', '基质', '细胞外基质']):
            module_name = 'ecm'

        report_lines = [
            f"# Virtual NP Cell 快速报告: {topic}\n",
            f"> 生成时间: 实时生成\n",
        ]

        # 知识部分
        report_lines.append("## 📖 知识库摘要\n")
        for section, data in knowledge.items():
            if isinstance(data, list):
                report_lines.append(f"- **{section}**: {len(data)} 条记录\n")
                for item in data[:5]:
                    if isinstance(item, dict):
                        name = item.get('gene', item.get('pathway', item.get('component', '')))
                        report_lines.append(f"  - {name}\n")
                    else:
                        report_lines.append(f"  - {item}\n")
                if len(data) > 5:
                    report_lines.append(f"  - ... 还有 {len(data)-5} 条\n")
            else:
                report_lines.append(f"- **{section}**: {str(data)[:100]}...\n")

        # 仿真结果
        if module_name:
            report_lines.append(f"\n## 🔬 模块仿真: {module_name}\n")
            res = self.run_module(module_name)
            if res['success']:
                report_lines.append(f"- ✅ 仿真成功\n")
                result = res['result']
                if 'metrics' in result:
                    for k, v in result['metrics'].items():
                        report_lines.append(f"  - {k}: {v:.4f}\n")
                if 'profile' in result:
                    for k, v in result['profile'].items():
                        report_lines.append(f"  - {k}: {v:.4f}\n")
                if 'senescence_score' in result:
                    report_lines.append(f"  - **衰老评分**: {result['senescence_score']:.4f}\n")
            else:
                report_lines.append(f"- ❌ 仿真失败: {res.get('error', '未知错误')}\n")

        return ''.join(report_lines)

    def _full_report(self) -> str:
        """综合状态报告"""
        lines = [
            "# 🧬 Virtual NP Cell — 综合系统报告\n",
            self.status_summary(),
            "\n",
            "## 📊 可用的 run_module() 模块\n",
            "\n",
        ]
        module_info = {
            'signaling': '信号通路 ODE 仿真 — 7条核心通路集成',
            'ecm': 'ECM 代谢动力学 — 合成-降解平衡 + 多因素扰动',
            'mechanotransduction': 'MRTF-A 机械力传导 — 基质刚度→糖酵解 (Bone Research 2025)',
            'metabolism': 'NP 代谢可塑性 — 糖酵解/OXPHOS/谷氨酰胺/线粒体',
            'senescence': 'NP 细胞衰老 — p53-p21/p16/SASP/线粒体/氧化应激',
            'm6a': 'm6A 表观转录组 — KDM5A→WTAP→m6A-NORAD→E2F3 (Nat Commun 2022)',
            'spatial': 'IVD 空间转录组 — NP/AF/CEP 分区 + 伪时间 + 生态位',
            'couple': '多尺度耦合仿真 (可选)',
            'drug_screen': '虚拟药物筛选 (可选)',
            'mitochondrial': '线粒体动力学 — 融合/分裂/Δψm/SIRT3/PINK1-Parkin/自噬/凋亡',
            'subcellular': '亚细胞区室 — ER应激/UPR/核纤层/溶酶体/NLRP3/外泌体/cfDNA-STING',
            'rna_dynamics': 'mRNA/相分离 — 应激颗粒/P-body/HuR-TTP/NEAT1/LLPS/翻译',
        }
        for mod, desc in module_info.items():
            status = '✅ 可用' if (globals().get(f'_HAS_{mod.upper().replace("SCREEN","DRUG")}', False) or
                                   mod not in ['couple', 'drug_screen']) else '❌ 不可用'
            if mod == 'couple':
                status = '✅ 可用' if _HAS_COUPLED else '📦 未安装'
            elif mod == 'drug_screen':
                status = '✅ 可用' if _HAS_DRUG else '📦 未安装'
            elif mod == 'mitochondrial':
                status = '✅ 可用' if _HAS_MITO else '📦 未安装'
            elif mod == 'subcellular':
                status = '✅ 可用' if _HAS_SUBCELL else '📦 未安装'
            elif mod == 'rna_dynamics':
                status = '✅ 可用' if _HAS_RNA else '📦 未安装'
            lines.append(f"- `{mod}` {status} — {desc}\n")

        lines.append("\n## 📝 使用示例\n")
        lines.append("```python\n")
        lines.append("from orchestrator import VirtualNPCell\n")
        lines.append("v = VirtualNPCell()\n")
        lines.append("print(v.status_summary())\n")
        lines.append("\n# 运行单个模块\n")
        lines.append("r = v.run_module('mechanotransduction', stiffness_level=3.0)\n")
        lines.append("\n# 运行管道\n")
        lines.append("results = v.run_pipeline(['signaling', 'mechanotransduction', 'metabolism'])\n")
        lines.append("\n# 快速报告\n")
        lines.append("report = v.quick_report('力传导')\n")
        lines.append("```\n")
        return ''.join(lines)


# ==================== 统一输出目录 ====================

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")


def ensure_output_dir():
    """确保输出目录存在"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR


# ==================== 自检 ====================

if __name__ == "__main__":
    print("Virtual NPCell v3.0 — 自检")
    print("=" * 55)
    v = VirtualNPCell()
    print(v.status_summary())
    print("\n测试 run_module ('mechanotransduction'):")
    r = v.run_module('mechanotransduction', stiffness_level=1.0)
    print(f"  → {'✅' if r['success'] else '❌'} {r.get('error', '')}")
    print("\n✅ Orchestrator v3.0 加载成功")
