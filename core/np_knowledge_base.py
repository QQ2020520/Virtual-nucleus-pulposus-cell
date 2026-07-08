"""
NP Cell Knowledge Base — 髓核细胞知识库
核心生物学、通路、ECM、标志基因等完整知识图谱
v2.0 扩展: 力传导+代谢+衰老+表观遗传+空间转录组+脊索亚群
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
        {"gene": "SPON1", "full_name": "Spondin-1", "function": "ECM 蛋白，NP 富集，参与细胞-基质相互作用"},
        # ====== v2.0 新增: 力传导/代谢/衰老/表观/空间 ======
        {"gene": "MRTFA", "full_name": "Myocardin related transcription factor A", "function": "基质刚度力传导关键因子; 退变NP中上调, 抑制糖酵解通过Kidins220/AMPK"},
        {"gene": "KIDINS220", "full_name": "Kinase D-interacting substrate 220", "function": "MRTF-A下游效应器; 激活AMPK促进糖酵解; 退变NP中下调"},
        {"gene": "WTAP", "full_name": "WT1 associated protein", "function": "m6A甲基转移酶复合体核心组件; 退变NP中上调, 通过m6A修饰NORAD促进衰老"},
        {"gene": "METTL3", "full_name": "Methyltransferase 3, N6-adenosine-methyltransferase complex catalytic subunit", "function": "m6A甲基转移酶催化亚基; 调控RNA代谢和NPC衰老"},
        {"gene": "METTL14", "full_name": "Methyltransferase 14, N6-adenosine-methyltransferase complex non-catalytic subunit", "function": "m6A甲基转移酶复合体; 与METTL3形成异二聚体"},
        {"gene": "NORAD", "full_name": "Non-coding RNA activated by DNA damage", "function": "lncRNA; WTAP介导的m6A修饰后稳定性增加, 通过PUMILIO/E2F3轴促进NPC衰老"},
        {"gene": "E2F3", "full_name": "E2F transcription factor 3", "function": "NORAD/PUMILIO轴的靶转录因子; 调控细胞周期和衰老"},
        {"gene": "PFKFB3", "full_name": "6-phosphofructo-2-kinase/fructose-2,6-biphosphatase 3", "function": "糖酵解关键调控酶; 生成F-2,6-BP激活PFK1; 退变NP中显著下调"},
        {"gene": "PFKM", "full_name": "Phosphofructokinase, muscle", "function": "糖酵解限速酶PFK1的肌肉亚型; 退变NP中下调"},
        {"gene": "KDM5A", "full_name": "Lysine demethylase 5A", "function": "H3K4me3去甲基化酶; 上调WTAP启动子的H3K4me3修饰激活WTAP转录"},
        {"gene": "CCL2", "full_name": "C-C motif chemokine ligand 2", "function": "TonEBP靶基因; 退变NP中上调, 募集巨噬细胞促进炎症"},
        {"gene": "CTSK", "full_name": "Cathepsin K", "function": "组织蛋白酶K; NP外周区域表达, 参与NP形态构建"},
        {"gene": "FOXA2", "full_name": "Forkhead box A2", "function": "脊索谱系标志物; 与TBXT共表达于脊索源性NP祖细胞"},
        {"gene": "GLUT1/SLC2A1", "full_name": "Glucose transporter 1", "function": "NP细胞主要葡萄糖转运体; 受HIF-1α调控, 维持无血管区糖酵解"},
        {"gene": "LDHA", "full_name": "Lactate dehydrogenase A", "function": "糖酵解终末酶; 催化丙酮酸→乳酸, 再生NAD+维持糖酵解"},
        {"gene": "PKM2", "full_name": "Pyruvate kinase M2", "function": "糖酵解终末限速酶; 调控有氧糖酵解和代谢重编程"},
        {"gene": "SIRT1", "full_name": "Sirtuin 1", "function": "NAD+依赖去乙酰化酶; 调控线粒体生物合成和NPC衰老"},
        {"gene": "PINK1", "full_name": "PTEN induced kinase 1", "function": "线粒体自噬调控因子; 退变NP中下调, 导致受损线粒体堆积"},
        {"gene": "BNIP3", "full_name": "BCL2 interacting protein 3", "function": "HIF-1α靶基因; 调控线粒体自噬; 低氧NP中维持线粒体质量"},
        {"gene": "NOTO", "full_name": "Notochord homeobox", "function": "脊索形成关键转录因子; NP谱系早期决定"},
        {"gene": "SHH", "full_name": "Sonic hedgehog", "function": "脊索分泌信号分子; 调控椎体和椎间盘模式形成"},
        {"gene": "YAP1", "full_name": "Yes associated protein 1", "function": "Hippo通路效应器; 机械力感知核效应器; 调控NP细胞增殖"},
        {"gene": "TONEBP/NFAT5", "full_name": "Tonicity-responsive enhancer binding protein", "function": "高渗应答TF; 调控CCL2/IL-6/TNF; 退变NP中TonEBP信号改变"},
        {"gene": "NOX4", "full_name": "NADPH oxidase 4", "function": "ROS主要来源; 退变NP中上调; 驱动氧化应激和衰老"},
        {"gene": "PIEZO1", "full_name": "Piezo type mechanosensitive ion channel component 1", "function": "机械力门控离子通道; 感知基质刚度和流体剪切力"},
        {"gene": "TRPV4", "full_name": "Transient receptor potential cation channel subfamily V member 4", "function": "渗透压和力学感受器; 维持NP细胞渗透压稳态"},
    ],
    "notochordal_markers": [
        "TBXT (Brachyury) - 脊索转录因子, NP谱系决定",
        "FOXA2 - 脊索共标志, 与TBXT共表达于NP祖细胞",
        "KRT19/KRT18 - 中间丝蛋白, 脊索NP表型标志",
        "CD24 - 表面标志, 脊索源性NP富集",
        "PAX1 - NP规格和脊索发育的主TF",
        "FOXF1 - 脊索细胞存活和NP身份维持",
        "NOTO - 脊索形成关键转录因子",
        "SHH - 脊索分泌信号分子, 调控椎体模式形成",
    ],
    "npc_subpopulations": [
        {
            "type": "Notochordal progenitor NPC",
            "markers": ["TBXT+", "FOXA2+", "KRT19+", "KRT18+", "CD24+"],
            "function": "脊索源性祖细胞; 具有自我更新和多向分化潜能; 存在于出生后NP核心区",
            "niche": "NP核心, 高氧张力区域, 脊索管残留",
            "in_vivo_evidence": "scRNA-seq在小鼠出生后NP中鉴定出TBXT+FOXA2+祖细胞和ACAN+COL2A1+成熟NPC (PMID: 38085183)"
        },
        {
            "type": "Mature chondrocyte-like NPC",
            "markers": ["ACAN+", "COL2A1+", "SOX9+", "HAPLN1+", "aggrecan+"],
            "function": "成熟NP细胞; 主要功能为ECM合成和维持; 占成年NP细胞主体",
            "niche": "NP外周, 靠近AF交界区",
            "note": "具有部分软骨样特征但仍维持NP特异性(如TBXT弱表达)"
        },
        {
            "type": "Senescent NPC",
            "markers": ["p16INK4a/CDKN2A+", "p21/CDKN1A+", "p53/TP53+", "SA-β-gal+", "γH2AX+"],
            "function": "衰老细胞; SASP分泌; 增殖停滞; 退变NP中显著累积",
            "niche": "退变NP全域, 炎症微环境浓度梯度区",
            "sasp_factors": ["IL-1β", "IL-6", "TNF-α", "CCL2", "MMP3", "MMP13", "ADAMTS5"]
        },
        {
            "type": "Ctsk+ NP peripheral cell",
            "markers": ["CTSK+", "SOX9+", "COL2A1+", "Tie2-"],
            "function": "NP外周区域Ctsk阳性细胞; 参与NP形态构建和边界维持",
            "niche": "NP外周区域, Ctsk表达在NP外周特定区域",
            "spatial_note": "空间转录组揭示Ctsk+细胞在NP外周特定分布, Tie2在NP亚群中缺失"
        },
        {
            "type": "Progenitor/NPPCs",
            "markers": ["CTSK+", "Tie2+", "GD2+", "CD155+"],
            "function": "NP祖细胞/前体细胞; 具有多向分化潜能; 维持NP再生能力",
            "niche": "NP外周区域, 靠近软骨终板侧",
            "spatial_note": "空间转录组首次揭示NPPCs的空间分布和NP核心→外周的潜在分化轨迹"
        },
    ],
    "mechanotransduction": [
        {
            "pathway": "Integrin-FAK-MRTF-A",
            "sensor": "Integrin αVβ3/α5β1 - ECM基质刚度感知",
            "transducer": "FAK/SRC → Actin polymerization → MRTF-A核转位 (G-actin↓ → MRTF-A释放)",
            "effector": "MRTF-A/SRF → Kidins220 → AMPK_p → PFKFB3/PFKM → Glycolysis",
            "role": "退变NP中基质刚度从~2kPa激增至~15kPa, 通过MRTF-A过度激活抑制糖酵解",
            "therapeutic": "CCG-1423 (MRTF-A/SRF抑制剂) 恢复糖酵解并改善NP退变表型",
            "reference": "Bone Research 2025 (PMID: pending)"
        },
        {
            "pathway": "YAP/TAZ-Hippo",
            "sensor": "ECM stiffness + cell shape + F-actin",
            "transducer": "F-actin → YAP/TAZ核定位 (LATS1/2磷酸化抑制)",
            "effector": "YAP/TAZ-TEAD → CTGF, CYR61, ANKRD1",
            "role": "NP细胞中YAP感知基质刚度和细胞形态变化, 调控增殖和分化"
        },
        {
            "sensor": "Piezo1/TRPV4机械力门控离子通道",
            "pathway": "Piezo1/TRPV4 → Ca2+ influx → Calcineurin → NFAT → 基因表达",
            "downstream": "NFATc1 → Cox2, MMP3; CaMKII → CREB",
            "role": "感知渗透压变化和力学负载; 退变中Piezo1表达改变",
        },
    ],
    "metabolism": [
        {
            "type": "Glycolysis (厌氧糖酵解)",
            "key_enzymes": ["HK2", "PFKM", "PFKFB3", "PKM2", "LDHA"],
            "regulation": "HIF-1α↑ → PFKFB3↑ → F-2,6-BP↑ → PFK1异位激活 → 糖酵解通量↑",
            "role": "NP主要ATP来源(无血管环境, PO2~1-5%); 退变中PFKFB3下调~7倍, 糖酵解严重受损",
            "therapeutic": "PFKFB3激活剂(如TEPP-46)可恢复糖酵解, 改善NP退变",
            "pathology": "基质刚度↑→MRTF-A↑→Kidins220↓→AMPK↓→PFKFB3↓→糖酵解↓→ECM失衡"
        },
        {
            "type": "OXPHOS (氧化磷酸化)",
            "key_enzymes": ["SDHA", "COX4I1", "ATP5A1", "PDH", "CS"],
            "regulation": "低氧→HIF-1α→PDK1↑→PDH磷酸化抑制→减少乙酰CoA进入TCA→OXPHOS↓",
            "role": "正常NP中OXPHOS受HIF-1α/PDK1轴抑制; 退变NP中随血管化和氧张力↑可能代偿性激活",
            "note": "线粒体功能障碍(膜电位Δψm↓, mtROS↑)是NPC衰老和IDD的关键驱动因素"
        },
        {
            "type": "Glutaminolysis (谷氨酰胺代谢)",
            "key_enzymes": ["GLS (谷氨酰胺酶)", "GLUL", "GLUD1", "GOT1", "GOT2"],
            "regulation": "c-Myc→GLS转录↑; mTOR→谷氨酰胺转运体SLC1A5活性↑",
            "role": "补充TCA循环中间产物(α-KG); 为脂肪酸合成提供碳源; 退变NP中谷氨酰胺代谢重编程"
        },
        {
            "type": "Fatty acid oxidation (FAO)",
            "key_enzymes": ["CPT1A", "CPT2", "ACADM", "ACOX1", "HADHA"],
            "regulation": "AMPK→pACC→CPT1去抑制→脂肪酸转运入线粒体",
            "role": "替代能源底物; NP中FAO的作用尚未完全阐明, 但在退变中可能改变",
        },
        {
            "type": "AMPK-mTOR 营养感知轴",
            "regulation": "能量应激(AMP/ATP↑)→AMPK↑→(1) 促进分解代谢(自噬/FAO) (2) 抑制mTORC1→蛋白合成↓",
            "role": "退变NP中Kidins220↓→AMPK磷酸化↓→(1)自噬受损(p62积累) (2) mTOR异常激活→衰老",
        },
    ],
    "epigenetics": [
        {
            "category": "m6A RNA methylation",
            "writers": ["METTL3", "METTL14", "WTAP", "KIAA1429", "RBM15", "ZC3H13"],
            "erasers": ["FTO", "ALKBH5"],
            "readers": ["YTHDF1", "YTHDF2", "YTHDF3", "YTHDC1", "YTHDC2", "IGF2BP1-3"],
            "cascade": "退变刺激→KDM5A→WTAP启动子H3K4me3↑→WTAP转录↑→全局m6A↑→NORAD m6A稳定性↑→NORAD积累→PUM1/2隔离→E2F3↑→细胞周期/DNA复制→NPC衰老",
            "note": "m6A是NPC衰老中最重要的表观转录组调控层之一 (PMID: 35232991, Nature Comms 2022)"
        },
        {
            "category": "Histone modifications (组蛋白修饰)",
            "h3k4me3": "KDM5A↑ → WTAP启动子H3K4me3↑ → WTAP转录激活 → m6A全局变化",
            "h3k27ac": "BRD4识别超增强子 → 炎症基因表达(TNF, IL1B) → NF-κB通路正反馈",
            "h3k27me3": "EZH2/PRC2 → 沉默NP分化基因 → NP表型维持; EZH2抑制→去分化",
            "h3k9me3": "异染色质维持; 衰老NPC中异染色质丢失→LINE-1转座子激活→cGAS-STING炎症",
            "hdacs": "HDAC1/2/3在退变NP中失调; HDAC抑制剂(TSA, SAHA)恢复COL2A1和ACAN表达",
        },
        {
            "category": "DNA methylation",
            "dnmts": ["DNMT1 (维持甲基化)", "DNMT3A (从头甲基化)", "DNMT3B (从头甲基化)"],
            "tdgs": ["TET1", "TET2", "TET3 (去甲基化)"],
            "role": "退变NP中全局低甲基化; MMP13启动子低甲基化→MMP13↑; SOX9启动子CpG超甲基化→沉默",
            "m6A_crosstalk": "TET2 → 去甲基化 → 调控m6A阅读器(如YTHDF2)表达 → 表观转录组交叉调控"
        },
        {
            "category": "Non-coding RNA regulatory network",
            "lncrna": ["NORAD (促衰老, m6A稳定)", "HOTAIR (抗凋亡, miR-34a海绵)", "KLF3-AS1 (ceRNA, miR-10a-3p海绵)"],
            "circrna": ["circRNA_104670 (促退变, 海绵)", "circRNA_CDH11 (调控ECM)", "circRNA_4099 (抗凋亡)"],
            "role": "lncRNA/circRNA通过ceRNA/海绵机制调控miRNA可用度 → 影响靶mRNA表达 → NP退变",
        },
    ],
    "senescence": [
        {
            "hallmark": "Cell cycle arrest (细胞周期停滞)",
            "markers": ["p16INK4a/CDKN2A↑", "p21/CDKN1A↑", "p53/TP53↑", "Rb低磷酸化"],
            "telomere_pathway": "端粒缩短→DDR (ATM/ATR→CHK1/2) → p53→p21→Rb→G1/S停滞",
            "stress_pathway": "氧化应激/致癌信号→p16→Rb→G1/S停滞 (不可逆)",
            "note": "p16-Rb通路在晚期NPC衰老中占主导地位(与复制性衰老不同)"
        },
        {
            "hallmark": "SASP (Senescence-Associated Secretory Phenotype)",
            "pro_inflammatory": ["IL-1β", "IL-6", "IL-8/CXCL8", "TNF-α", "CCL2/MCP-1", "CCL5/RANTES"],
            "matrix_degrading": ["MMP3", "MMP13", "ADAMTS4", "ADAMTS5"],
            "growth_factors": ["VEGF", "TGF-β1", "CTGF", "HGF"],
            "mechanisms": "NF-κB (p65/p50) 和 C/EBPβ驱动SASP转录程序; DNA损伤→DDR→NF-κB; 线粒体ROS→GATA4→NF-κB→IL-1A→SASP正反馈",
            "therapeutic_target": "senolytics (达沙替尼+槲皮素) 清除衰老NPC; JAK抑制剂减弱SASP; 二甲双胍(SASP调节)"
        },
        {
            "hallmark": "Mitochondrial dysfunction (线粒体功能障碍)",
            "changes": ["线粒体膜电位(Δψm)↓", "mtROS↑ (mitoSox/Superoxide↑)", "ATP合成↓", "线粒体碎片化(DRP1↑)", "线粒体嵴结构紊乱"],
            "machinery": "PINK1/Parkin → 线粒体自噬; BNIP3/NIX → 低氧下适应性线粒体自噬; DRP1 → 线粒体分裂; OPA1 → 线粒体融合/嵴重塑",
            "consequence": "受损线粒体堆积→mtROS↑→DDR↑→p53/p21↑→衰老强化 (恶性循环)",
        },
        {
            "hallmark": "Metabolic reprogramming (代谢重编程)",
            "glycolysis": "糖酵解受损 (PFKFB3↓~7倍, PFKM↓, GLUT1↓)",
            "glutamine": "谷氨酰胺代谢改变 (GLS活性变化→α-KG/2-HG比例改变→表观遗传修饰连接)",
            "NAD+_sirtuin": "NAD+↓ / NADH↑ → SIRT1↓ → PGC-1α低乙酰化 → 线粒体生物合成↓",
            "AMPK_autophagy": "AMPK磷酸化↓(通过Kidins220↓) → p62↑ → 自噬流受损 → 蛋白毒性应激",
        },
        {
            "hallmark": "Epigenetic alterations (表观遗传改变)",
            "global_hypomethylation": "全局DNA低甲基化 → 转座子激活 → cGAS-STING → 炎症性SASP",
            "h3k9_loss": "衰老相关异染色质灶(SAHF)松散 → H3K9me3丢失 → 核结构紊乱",
            "m6A_epitranscriptome": "WTAP/m6A/NORAD轴 → E2F3↑ → 细胞周期基因去抑制 → 异常增殖信号→衰老",
        },
    ],
    "signaling_pathways": [
        {
            "pathway": "TGF-β/BMP",
            "key_genes": ["TGFB1", "TGFBR1", "TGFBR2", "BMP2", "BMP7", "SMAD2", "SMAD3", "SMAD4", "SMAD1", "SMAD5", "SMAD7"],
            "role": "促进 ECM 合成 (COL2A1, ACAN)，通过 SOX9 上调维持 NP 表型；功能失调导致基质退变"
        },
        {
            "pathway": "Wnt/β-catenin",
            "key_genes": ["CTNNB1", "WNT3A", "WNT5A", "LEF1", "TCF7", "AXIN2", "DKK1", "SFRP1", "SFRP2"],
            "role": "调控 NP 细胞增殖和分化；异常激活加速 NP 退变，诱导基质分解和衰老"
        },
        {
            "pathway": "HIF-1α/Hypoxia",
            "key_genes": ["HIF1A", "EPAS1/HIF2A", "VEGFA", "SLC2A1/GLUT1", "LDHA", "PDK1", "BNIP3", "CA12"],
            "role": "NP 适应无血管低氧环境的主调控；维持糖酵解，抑制氧化代谢，促进 NP 标志物表达"
        },
        {
            "pathway": "MAPK/ERK",
            "key_genes": ["MAPK1/ERK2", "MAPK3/ERK1", "MAP2K1/MEK1", "EGFR", "FGFR1", "FGFR2", "JUN/c-Jun", "MAPK14/p38"],
            "role": "介导炎症细胞因子应答；ERK1/2 促分解 MMP 表达；p38/JNK 驱动退变 NP 细胞凋亡"
        },
        {
            "pathway": "NF-κB",
            "key_genes": ["NFKB1/p50", "RELA/p65", "IKBKB/IKKβ", "CHUK/IKKα", "TNFRSF1A/TNFR1", "IL1R1"],
            "role": "NP 退变中的核心炎症通路；被 IL-1β/TNF-α 激活，驱动 MMP/ADAMTS 表达，抑制 ECM 合成；维持SASP正反馈"
        },
        {
            "pathway": "Notch",
            "key_genes": ["NOTCH1", "NOTCH2", "JAG1", "DLL1", "HES1", "HEY1", "HES5"],
            "role": "调控 NP 细胞命运决定和祖细胞维持；异常 Notch 信号与 NP 衰老和椎间盘退变相关"
        },
        {
            "pathway": "PI3K/Akt/mTOR",
            "key_genes": ["PIK3CA", "AKT1", "MTOR", "PTEN", "RPS6KB1/S6K1", "EIF4EBP1/4E-BP1"],
            "role": "促进 NP 细胞存活、增殖和基质合成；退变椎间盘中受抑制；mTOR 与自噬失调相关"
        },
        # v2.0 新增通路
        {
            "pathway": "MRTF-A/SRF Mechanotransduction",
            "key_genes": ["MRTFA", "SRF", "KIDINS220", "PRKAA1/AMPKα1", "PFKFB3", "PFKM"],
            "role": "基质刚度→整合素→FAK→Actin→MRTF-A核转位→Kidins220→AMPK→糖酵解"
        },
        {
            "pathway": "Hippo/YAP",
            "key_genes": ["YAP1", "WWTR1/TAZ", "LATS1", "LATS2", "TEAD1-4", "CTGF", "CYR61", "ANKRD1"],
            "role": "机械力感知→YAP/TAZ核定位→增殖/分化; 与MRTF-A通路交互作用"
        },
        {
            "pathway": "m6A Epitranscriptome (WTAP-NORAD-PUM-E2F3)",
            "key_genes": ["WTAP", "METTL3", "METTL14", "KDM5A", "NORAD", "PUM1", "PUM2", "E2F3", "YTHDF2"],
            "role": "KDM5A→H3K4me3→WTAP→m6A→NORAD稳定→PUM海绵→E2F3→衰老"
        },
        {
            "pathway": "cGAS-STING",
            "key_genes": ["CGAS/MB21D1", "STING/TMEM173", "TBK1", "IRF3", "IFNB1"],
            "role": "衰老/NP退变中异染色质丢失→LINE-1转座→细胞质DNA→cGAS-STING→干扰素→炎症性SASP"
        },
        {
            "pathway": "TonEBP/NFAT5 Osmotic Stress",
            "key_genes": ["NFAT5/TonEBP", "CCL2", "NOS2", "PTGS2/COX2", "IL6", "TNF"],
            "role": "高渗环境适应性; TonEBP调控CCL2, IL6, NOS2; 退变中TonEBP信号改变与促炎通路交叉"
        },
    ],
    "ecm_components": [
        {"component": "Aggrecan (ACAN)", "type": "蛋白聚糖", "function": "提供渗透膨胀压力，维持椎间盘高度和抗压能力"},
        {"component": "Versican (VCAN)", "type": "蛋白聚糖", "function": "年轻 NP 中的大蛋白聚糖，维持水合和抗压刚度"},
        {"component": "Collagen II (COL2A1)", "type": "纤维胶原", "function": "NP 主要胶原，形成松散纤维网络，抵抗张力和维持结构"},
        {"component": "Collagen IX (COL9A1/2/3)", "type": "FACIT 胶原", "function": "促进胶原 II 纤维组装和稳定；突变与椎间盘退变相关"},
        {"component": "Collagen VI (COL6A1/2/3)", "type": "微纤维胶原", "function": "NP 细胞周基质中的珠状微纤维；细胞-基质互作和力感知关键"},
        {"component": "Hyaluronan (HA)", "type": "糖胺聚糖", "function": "聚集蛋白聚糖的非硫酸化骨架；锁水并提供粘弹性"},
        {"component": "Elastin (ELN)", "type": "弹性纤维蛋白", "function": "压缩后提供弹性回缩；NP 中含量少但功能重要"},
        {"component": "Fibromodulin (FMOD)", "type": "小富含亮氨酸蛋白聚糖", "function": "胶原纤维形成调控；调节 NP 中 TGF-β 生物利用度"},
        {"component": "Lumican (LUM)", "type": "小富含亮氨酸蛋白聚糖", "function": "胶原纤维组装; NP中高度表达"},
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
        {"gene": "CASP3", "role": "Caspase-3: 细胞凋亡执行者；退变 NP 中活性升高"},
        # v2.0
        {"gene": "IL6", "role": "促炎细胞因子; SASP主要成分; 退变NP中升高"},
        {"gene": "CCL2", "role": "MCP-1趋化因子; TonEBP靶基因; 巨噬细胞募集"},
        {"gene": "NGF", "role": "神经生长因子; 退变NP中↑→感觉神经长入→椎间盘源性疼痛"},
        {"gene": "BDNF", "role": "脑源性神经营养因子; 退变NP中↑→疼痛信号传导"},
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
        {"gene": "TERT", "role": "端粒酶逆转录酶: 端粒维持，退变 NP 中端粒缩短"},
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
        {"gene": "FBLN1", "direction": "down", "note": "AF 中较高表达"},
    ],
    "metabolic_features": {
        "energy_metabolism": "NP 细胞在无血管低氧环境中主要依赖糖酵解产生 ATP，而非氧化磷酸化",
        "hif_axis": "HIF-1α/EPAS1 是代谢适应的主调控因子，维持糖酵解酶 (LDHA, PDK1, SLC2A1) 高表达",
        "glucose_utilization": "葡萄糖转运体 GLUT1 (SLC2A1) 高表达，乳酸脱氢酶 LDHA 活跃",
        "lactate": "大量乳酸产生，通过 CA12 和 MCT 转运体调节 pH",
        "mitochondrial": "线粒体功能受抑制（PDK1 抑制丙酮酸进入 TCA 循环），但仍有基础活性",
        "nutrient_environment": "椎间盘中央 PO₂ ~1-5%，pH ~6.9-7.2，葡萄糖浓度低 (~2.5mM)",
        "oxidative_stress": "抗氧化系统 (SOD2, CAT, GPX) 高度依赖以应对氧化应激",
        "autophagy": "基础自噬水平高以维持细胞稳态；退变时自噬失调",
        "glycolytic_decline": "退变NP中PFKFB3↓~7倍, PFKM↓, PLD1↓, 糖酵解严重受损 (Bone Research 2025)",
        "mechano_metabolic_coupling": "基质刚度↑→MRTF-A↑→Kidins220↓→AMPK↓→PFKFB3↓ (力学生物学-代谢耦合)",
    },
    "np_phenotype_TFs": [
        {"tf": "PAX1", "role": "NP 细胞身份的主要决定因子，脊索发育的主控转录因子"},
        {"tf": "FOXF1", "role": "NP 身份维持的关键转录因子，调控 KRT19 和 CD24"},
        {"tf": "TBXT (Brachyury)", "role": "脊索谱系转录因子，NP 细胞谱系标志"},
        {"tf": "SOX9", "role": "软骨形成主控因子，调控 ECM 基因 (COL2A1, ACAN)"},
        {"tf": "HIF1A", "role": "低氧适应和 NP 表型维持的转录因子"},
        {"tf": "EPAS1 (HIF2A)", "role": "低氧应答，调控 NP 代谢适应"},
        {"tf": "KLF4", "role": "参与 NP 细胞干性维持和抗衰老"},
        {"tf": "MRTFA", "role": "力传导转录因子; 基质刚度→核转位→Kidins220/糖酵素控制"},
        {"tf": "FOXA2", "role": "脊索谱系共标志; 与TBXT在胚胎脊索中共表达"},
    ],
    "model_systems": {
        "cell_lines": ["大鼠原代 NP 细胞", "人原代 NP 细胞（手术标本）", "小鼠原代 NP 细胞", "猪原代 NP 细胞"],
        "immortalized_lines": ["大鼠 NP 细胞系", "人髓核永生化细胞系 (e.g., SV40 转化)"],
        "culture_conditions": "低氧 (2-5% O₂), 低糖 DMEM, 3D 培养（藻酸盐/琼脂糖/胶原凝胶）以维持表型",
        "in_vivo_models": ["针刺尾椎退变模型 (大鼠/小鼠/兔)", "自发退变模型 (沙鼠)", "力学生物学退变模型 (小鼠IVD加压)"],
        "advanced_models": ["空间转录组 (10x Visium, MERFISH)", "scRNA-seq (10x Genomics)", "多组学整合 (代谢组/表观组/转录组)"],
    },
    "spatial_reference": {
        "technique": "10x Visium Spatial Transcriptomics, MERFISH",
        "tissue": "出生后小鼠 IVD (P7-P28)",
        "compartments": ["Nucleus pulposus (NP 核心)", "Annulus fibrosus (AF 环状外区)", "Cartilaginous endplates (CEP 软骨终板)"],
        "key_findings": [
            "Ctsk在NP外周区域表达缺失, 部分Ctsk+细胞参与NP形态构建",
            "Tie2在NP亚群中表达缺失",
            "NP祖细胞(NPPCs)具有特定的空间分布和分化轨迹",
            "NP, AF和CEP具有不同的空间分子特征和功能富集"
        ],
        "np_zone_markers": {"core": ["TBXT", "KRT19", "CD24", "FOXF1"], "peripheral": ["CTSK", "COL2A1", "ACAN"]},
        "reference": "Xu R et al. Adv Sci 2024. Characterization of the Nucleus Pulposus Progenitor Cells via Spatial Transcriptomics"
    },
    "therapeutic_targets": [
        {"target": "MRTF-A", "agent": "CCG-1423 (MRTF-A/SRF抑制剂)", "rationale": "恢复退变NP中MRTF-A过度激活导致的糖酵解抑制", "evidence": "Bone Research 2025"},
        {"target": "PFKFB3", "agent": "TEPP-46 / AZ67 (PFKFB3激活剂)", "rationale": "恢复NP细胞糖酵解通量, 改善ECM合成", "evidence": "糖酵解恢复→NP表型维持"},
        {"target": "Senescent NPCs", "agent": "Dasatinib + Quercetin (senolytic)", "rationale": "选择性清除衰老NPC, 减少SASP分泌", "evidence": "临床前IVD模型"},
        {"target": "JAK-STAT", "agent": "Tofacitinib / Baricitinib", "rationale": "减弱SASP的旁分泌效应", "evidence": "SASP信号抑制→炎症微环境改善"},
        {"target": "mTOR", "agent": "Rapamycin / Metformin", "rationale": "恢复自噬流, 延缓NPC衰老", "evidence": "mTOR抑制→自噬↑→衰老标志↓"},
        {"target": "WTAP/m6A", "agent": "STM2457 (METTL3抑制剂)", "rationale": "阻断m6A/NORAD/E2F3衰老轴", "evidence": "Nature Comms 2022概念验证"},
        {"target": "AMPK", "agent": "Metformin / AICAR", "rationale": "激活AMPK→恢复自噬→改善代谢→抗衰老", "evidence": "AMPK在退变NP中活性↓"},
        {"target": "NAD+/SIRT1", "agent": "NMN / NAD前体", "rationale": "提升NAD+水平→激活SIRT1→线粒体生物合成→抗衰老", "evidence": "SIRT1在退变NP中↓"},
    ],
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
