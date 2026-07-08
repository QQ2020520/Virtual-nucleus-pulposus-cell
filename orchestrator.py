"""
Virtual NP Cell — 虚拟髓核细胞主系统
Orchestrator: 智能问答 + 功能路由
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.np_knowledge_base import NP_KNOWLEDGE_BASE
import numpy as np


class VirtualNPCell:
    """
    虚拟髓核细胞 — 主入口

    功能:
    1. 差异表达分析 & 火山图
    2. 基因表达热图
    3. 生物标志物预测 & 趋势分析
    4. 信号通路 ODE 仿真
    5. ECM 代谢动力学模型
    6. 知识问答 (NP 生物学)
    """

    def __init__(self):
        self.kb = NP_KNOWLEDGE_BASE
        self.de_results = None
        self.signal_model = None
        self.ecm_model = None

    def query_knowledge(self, topic: str) -> dict:
        """查询 NP 细胞知识库"""
        topic_map = {
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
        }
        for key, val in topic_map.items():
            if key in topic.lower():
                return {val: self.kb[val]}

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
            results = {"info": "未找到匹配主题。可用主题: marker, 通路, ECM, 退变, 炎症, 差异, 代谢, 转录因子, 模型"}
        return results

    def status_summary(self) -> str:
        """NP 细胞状态摘要"""
        return (
            f"🧬 虚拟髓核细胞 (Virtual NP Cell) 系统状态\n"
            f"{'='*40}\n"
            f"✓ 知识库: {sum(len(v) if isinstance(v, list) else 1 for v in self.kb.values())} 条记录\n"
            f"✓ 信号通路: {len(self.kb['signaling_pathways'])} 条核心通路\n"
            f"✓ 标志基因: {len(self.kb['marker_genes'])} 个\n"
            f"✓ ECM 成分: {len(self.kb['ecm_components'])} 种\n"
            f"✓ 退变基因: {len(self.kb['degeneration_genes'])} 个\n"
            f"✓ 可分析: 火山图 · 热图 · ROC · 时序趋势 · ODE 仿真\n"
        )


# ========== 统一输出目录 ==========
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR
