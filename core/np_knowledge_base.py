"""
NP Cell Knowledge Base — 髓核细胞知识库
核心生物学、通路、ECM、标志基因等完整知识图谱
"""

NP_KNOWLEDGE_BASE = {
    "marker_genes": [
        {"gene": "KRT19", "full_name": "Keratin 19", "function": "经典 NP 细胞标志物，中间丝蛋白，脊索源性 NP 细胞高表达"},
        {"gene": "KRT18", "full_name": "Keratin 18", "function": "与 KRT19 共表达，脊索 NP 表型的标志"},
        {"gene": "PAX1", "full_name": "Paired box 1", "function": "NP 细胞规格化和脊索发育的主转录因子"},
        {"gene": "FOXF1", "full_name": "Forkhead box F1", "function": "NP 细胞身份维持和脊索细胞存活的关键"},
        {"gene": "CD24", "full_name": "CD24 molecule", "function": "表面标志物，NP 细胞比 AF 和软骨细胞显著富集"},
        {"gene": "TBXT", "full_name": "Brachyury (T)", "function": "脊索转录因子，髓核细胞谱系特异性标志"},
        {"gene": "ACAN", "full_name": "Aggrecan", "function": "主要蛋白聚糖，提供渗透膨胀压力维持椎间盘高度"},
        {"gene": "SOX9", "full_name": "SRY-box 9", "function": "软骨形成和 NP 细胞 ECM 生成的主调控因子"},
        {"gene": "COL2A1", "full_name": "Collagen type II alpha 1", "function": "NP 细胞合成的主要胶原，形成纤维网络"},
        {"gene": "NOG", "full_name": "Noggin", "function": "BMP 拮抗剂，NP 高表达，调控椎间盘稳态"},
        {"gene": "CLEC3A", "full_name": "C-type lectin 3A", "function": "NP 富集标志物，参与基质组织的潜在作用"},
        {"gene": "CA12", "full_name": "Carbonic anhydrase 12", "function": "低氧 NP 微环境 pH 调节"},
        {"gene": "HAPLN1", "full_name": "Hyaluronan proteoglycan link 1", "function": "稳定 aggrecan-hyaluronan 聚集体"},
        {"gene": "SPON1", "full_name": "Spondin-1", "function": "ECM 蛋白，NP 富集，参与细胞-基质相互作用"}
    ],
    "signaling_pathways": [
        {
            "pathway": "TGF-β/BMP",
            "key_genes": ["TGFB1", "TGFBR1", "TGFBR2", "BMP2", "BMP7", "SMAD2", "SMAD3", "SMAD4", "SMAD1", "SMAD5"],
            "role": "促进 ECM 合成 (COL2A1, ACAN)，通过 SOX9 上调维持 NP 表型；功能失调导致基质退变"
        },
        {
            "pathway": "Wnt/β-catenin",
            "key_genes": ["CTNNB1", "WNT3A", "WNT5A", "LEF1", "TCF7", "AXIN2", "DKK1", "SFRP1"],
            "role": "调控 NP 细胞增殖和分化；异常激活加速 NP 退变，诱导基质分解和衰老"
        },
        {
            "pathway": "HIF-1α/Hypoxia",
            "key_genes": ["HIF1A", "EPAS1", "VEGFA", "SLC2A1", "LDHA", "PDK1"],
            "role": "NP 适应无血管低氧环境的主调控；维持糖酵解，抑制氧化代谢，促进 NP 标志物表达"
        },
        {
            "pathway": "MAPK/ERK",
            "key_genes": ["MAPK1", "MAPK3", "MAP2K1", "EGFR", "FGFR1", "FGFR2", "JUN", "MAPK14"],
            "role": "介导炎症细胞因子应答；ERK1/2 促分解 MMP 表达；p38/JNK 驱动退变 NP 细胞凋亡"
        },
        {
            "pathway": "NF-κB",
            "key_genes": ["NFKB1", "RELA", "IKBKB", "TNFRSF1A", "IL1R1"],
            "role": "NP 退变中的核心炎症通路；被 IL-1β/TNF-α 激活，驱动 MMP/ADAMTS 表达，抑制 ECM 合成"
        },
        {
            "pathway": "Notch",
            "key_genes": ["NOTCH1", "NOTCH2", "JAG1", "DLL1", "HES1", "HEY1"],
            "role": "调控 NP 细胞命运决定和祖细胞维持；异常 Notch 信号与 NP 衰老和椎间盘退变相关"
        },
        {
            "pathway": "PI3K/Akt/mTOR",
            "key_genes": ["PIK3CA", "AKT1", "MTOR", "PTEN", "RPS6KB1"],
            "role": "促进 NP 细胞存活、增殖和基质合成；退变椎间盘中受抑制；mTOR 与自噬失调相关"
        }
    ],
    "ecm_components": [
        {"component": "Aggrecan (ACAN)", "type": "蛋白聚糖", "function": "提供渗透膨胀压力，维持椎间盘高度和抗压能力"},
        {"component": "Versican (VCAN)", "type": "蛋白聚糖", "function": "年轻 NP 中的大蛋白聚糖，维持水合和抗压刚度"},
        {"component": "Collagen II (COL2A1)", "type": "纤维胶原", "function": "NP 主要胶原，形成松散纤维网络，抵抗张力和维持结构"},
        {"component": "Collagen IX (COL9A1/2/3)", "type": "FACIT 胶原", "function": "促进胶原 II 纤维组装和稳定；突变与椎间盘退变相关"},
        {"component": "Collagen VI (COL6A1/2/3)", "type": "微纤维胶原", "function": "NP 细胞周基质中的珠状微纤维；细胞-基质互作和力感知关键"},
        {"component": "Hyaluronan (HA)", "type": "糖胺聚糖", "function": "聚集蛋白聚糖的非硫酸化骨架；锁水并提供粘弹性"},
        {"component": "Elastin (ELN)", "type": "弹性纤维蛋白", "function": "压缩后提供弹性回缩；NP 中含量少但功能重要"},
        {"component": "Fibromodulin (FMOD)", "type": "小富含亮氨酸蛋白聚糖", "function": "胶原纤维形成调控；调节 NP 中 TGF-β 生物利用度"}
    ],
    "degeneration_genes": [
        {"gene": "MMP3", "role": "基质金属蛋白酶 3 (stromelysin-1): 降解聚集蛋白聚糖、II/IX/X/XI 型胶原"},
        {"gene": "MMP13", "role": "胶原酶 3: 优先切割 II 型胶原；退变 NP 中强烈上调"},
        {"gene": "ADAMTS4", "role": "聚集蛋白聚糖酶-1: 在特异性位点切割聚集蛋白聚糖"},
        {"gene": "ADAMTS5", "role": "聚集蛋白聚糖酶-2: IVD 退变中主要的聚集蛋白聚糖降解酶"},
        {"gene": "IL1B", "role": "白介素-1β: 主促炎细胞因子；诱导 MMP/ADAMTS 表达，抑制 ECM 合成"},
        {"gene": "TNF", "role": "肿瘤坏死因子-α: 与 IL-1β 协同促进 NP 分解代谢"},
        {"gene": "GDF5", "role": "生长分化因子 5 (BMP14): 调控 NP 分化；多态性与腰椎退变相关"},
        {"gene": "VDR", "role": "维生素 D 受体: 多态性 (TaqI, FokI) 与 IVD 退变风险相关"},
        {"gene": "COL9A2", "role": "IX 型胶原 α2 链: Trp2 等位基因增加椎间盘退变风险"},
        {"gene": "CHST3", "role": "硫酸软骨素 3: 突变导致脊柱骨骺发育不良伴椎间盘早衰"},
        {"gene": "TIMP1", "role": "金属蛋白酶组织抑制因子 1: MMP 内源性抑制；MMP/TIMP 失衡导致基质降解"},
        {"gene": "CASP3", "role": "Caspase-3: 细胞凋亡执行者；退变 NP 中活性升高"}
    ],
    "inflammation_aging_genes": [
        {"gene": "IL6", "role": "促炎细胞因子，退变 NP 中升高，诱导急性期反应"},
        {"gene": "CXCL8", "role": "IL-8: 趋化因子，招募免疫细胞至退变椎间盘"},
        {"gene": "CCL2", "role": "MCP-1: 单核细胞趋化蛋白，巨噬细胞浸润的关键趋化因子"},
        {"gene": "NFKB1", "role": "NF-κB 亚基 p50: 炎症级联主开关"},
        {"gene": "SIRT1", "role": "NAD+ 依赖性去乙酰化酶：抗衰老蛋白，退变 NP 中下调"},
        {"gene": "TP53", "role": "p53: 细胞衰老和凋亡调控因子，退变 NP 中激活"},
        {"gene": "CDKN2A", "role": "p16INK4a: 细胞衰老标志物，退变 NP 中表达升高"},
        {"gene": "SOD2", "role": "超氧化物歧化酶 2: 线粒体抗氧化防御，退变中失调"},
        {"gene": "CAT", "role": "过氧化氢酶: 抗氧化酶，退变 NP 中活性降低"},
        {"gene": "TERT", "role": "端粒酶逆转录酶: 端粒维持，退变 NP 中端粒缩短"}
    ],
    "differential_genes_vs_af": [
        {"gene": "KRT19", "direction": "up", "note": "NP vs AF: 最显著的差异标志之一"},
        {"gene": "PAX1", "direction": "up", "note": "NP 特异性，AF 几乎不表达"},
        {"gene": "FOXF1", "direction": "up", "note": "NP 特异性转录因子"},
        {"gene": "CD24", "direction": "up", "note": "NP 高表达，AF 低表达"},
        {"gene": "TBXT", "direction": "up", "note": "脊索标志，NP 特异性"},
        {"gene": "NOG", "direction": "up", "note": "NP 特异性 BMP 拮抗剂"},
        {"gene": "CLEC3A", "direction": "up", "note": "NP 显著富集"},
        {"gene": "CA12", "direction": "up", "note": "NP 低氧适应相关高表达"},
        {"gene": "ACAN", "direction": "up", "note": "NP 中聚集蛋白聚糖表达高于 AF"},
        {"gene": "COL1A1", "direction": "down", "note": "I 型胶原 AF 高表达，NP 低表达"},
        {"gene": "COL5A1", "direction": "down", "note": "AF 高表达，NP 相对低"},
        {"gene": "THY1", "direction": "down", "note": "AF 细胞表面标志物"},
        {"gene": "FBLN1", "direction": "down", "note": "AF 中较高表达"}
    ],
    "metabolic_features": {
        "energy_metabolism": "NP 细胞在无血管低氧环境中主要依赖糖酵解产生 ATP，而非氧化磷酸化",
        "hif_axis": "HIF-1α/EPAS1 是代谢适应的主调控因子，维持糖酵解酶 (LDHA, PDK1, SLC2A1) 高表达",
        "glucose_utilization": "葡萄糖转运体 GLUT1 (SLC2A1) 高表达，乳酸脱氢酶 LDHA 活跃",
        "lactate": "大量乳酸产生，通过 CA12 和 MCT 转运体调节 pH",
        "mitochondrial": "线粒体功能受抑制（PDK1 抑制丙酮酸进入 TCA 循环），但仍有基础活性",
        "nutrient_environment": "椎间盘中央 PO₂ ~1-5%，pH ~6.9-7.2，葡萄糖浓度低 (~2.5mM)",
        "oxidative_stress": "抗氧化系统 (SOD2, CAT, GPX) 高度依赖以应对氧化应激",
        "autophagy": "基础自噬水平高以维持细胞稳态；退变时自噬失调"
    },
    "np_phenotype_TFs": [
        {"tf": "PAX1", "role": "NP 细胞身份的主要决定因子，脊索发育的主控转录因子"},
        {"tf": "FOXF1", "role": "NP 身份维持的关键转录因子，调控 KRT19 和 CD24"},
        {"tf": "TBXT (Brachyury)", "role": "脊索谱系转录因子，NP 细胞谱系标志"},
        {"tf": "SOX9", "role": "软骨形成主控因子，调控 ECM 基因 (COL2A1, ACAN)"},
        {"tf": "HIF1A", "role": "低氧适应和 NP 表型维持的转录因子"},
        {"tf": "EPAS1 (HIF2A)", "role": "低氧应答，调控 NP 代谢适应"},
        {"tf": "KLF4", "role": "参与 NP 细胞干性维持和抗衰老"}
    ],
    "model_systems": {
        "cell_lines": ["大鼠原代 NP 细胞", "人原代 NP 细胞（手术标本）", "小鼠原代 NP 细胞", "猪原代 NP 细胞"],
        "immortalized_lines": ["大鼠 NP 细胞系", "人髓核永生化细胞系 (e.g., SV40 转化)"],
        "culture_conditions": "低氧 (2-5% O₂), 低糖 DMEM, 3D 培养（藻酸盐/琼脂糖/胶原凝胶）以维持表型",
        "in_vivo_models": ["针刺尾椎退变模型 (大鼠/小鼠/兔)", "自发退变模型 (沙鼠)", "力学生物学退变模型"]
    }
}


def get_gene_group(group_name: str) -> list:
    """获取指定基因组的基因名列表"""
    groups = {
        "marker": [g["gene"] for g in NP_KNOWLEDGE_BASE["marker_genes"]],
        "degeneration": [g["gene"] for g in NP_KNOWLEDGE_BASE["degeneration_genes"]],
        "inflammation_aging": [g["gene"] for g in NP_KNOWLEDGE_BASE["inflammation_aging_genes"]],
        "ecm": [c["component"].split(" ")[0] for c in NP_KNOWLEDGE_BASE["ecm_components"]],
    }
    return groups.get(group_name, [])


def get_all_pathway_genes() -> list:
    """获取所有通路涉及基因的去重列表"""
    genes = set()
    for p in NP_KNOWLEDGE_BASE["signaling_pathways"]:
        for g in p["key_genes"]:
            genes.add(g)
    return sorted(genes)
