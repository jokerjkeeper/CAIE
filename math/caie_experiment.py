"""
CAIE 模型模擬器 — Cognitive Adaptation in the Intelligence Era
================================================================
基於 CAIE 動力學方程的數值模擬，使用 Runge-Kutta 求解 ODE 系統。

核心方程：
  dS/dt = α·E_eff(t) + β·∇P(t) - γ·R(S,age,n) - δ·D_cum(t)

7 維認知-情感能力向量 S = [WM, SW, IX, MA, MR, EI, ID]

參數校準來源：
  - BCG 2026 "AI Brain Fry" 研究（多工具使用峰值 ≈ 3）
  - Monsell 2003（switch cost 基線）
  - Schultz 2016（RPE 多巴胺模型）
  - Gloria Mark（中斷恢復時間 ≈ 23 分鐘）
  - CDC ADHD 盛行率趨勢（10年增長 12.6%）
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial']
matplotlib.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. 參數定義
# ============================================================

# --- 能力維度標籤 ---
DIM_LABELS = ['WM\n工作記憶', 'SW\n任務切換', 'IX\n索引記憶',
              'MA\n多代理管理', 'MR\n元認知', 'EI\n情緒智力', 'ID\n自我認同']
N_DIM = 7

# --- 技術壓力參數 ---
# Sigmoid: P_k(t) = P_max / (1 + exp(-λ(t - t_k)))
TECH_PARAMS = {
    'T1_engine':  {'t_k': 1850, 'lam': 0.05, 'P_max': 0.6,
                   'w': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1])},
    'T2_internet': {'t_k': 2007, 'lam': 0.30, 'P_max': 0.7,
                    'w': np.array([0.2, 0.5, 0.3, 0.1, 0.1, 0.3, 0.2])},
    'T3_ai':       {'t_k': 2023, 'lam': 0.80, 'P_max': 1.0,
                    'w': np.array([0.8, 0.9, 0.95, 1.0, 0.7, 0.6, 0.85])},
}

# --- 認知阻力參數 ---
R_PLASTICITY_R0 = 0.01      # 基線可塑性阻力
R_PLASTICITY_KAPPA = 0.02   # 年齡衰減率
R_SWITCH_C0 = 0.05          # 切換成本係數
B_COG = 3.0                 # 認知頻寬 (bits/sec 歸一化)
R_HABIT_ETA = 0.1           # 習慣慣性係數

# --- 教育/動機參數 ---
ALPHA = 0.5    # 教育驅動係數
BETA = 0.3     # 壓力適應係數
GAMMA = 1.0    # 認知阻力係數
DELTA = 0.05   # 累積損傷係數
MU_RPE = 0.5   # RPE 調制強度
D0 = 1.0       # 多巴胺基線
PHI = 0.1      # 熟練度學習率

# --- 發育上界 Gompertz 參數 ---
# s_max(age) = s_inf * exp(-a * exp(-b * age))
S_INF = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
GOMP_A = np.array([3.0, 3.0, 2.5, 4.0, 4.0, 2.0, 2.0])
GOMP_B = np.array([0.15, 0.15, 0.12, 0.10, 0.10, 0.12, 0.12])


# ============================================================
# 2. 模型函數
# ============================================================

def s_max(age):
    """Gompertz 發育上界"""
    return S_INF * np.exp(-GOMP_A * np.exp(-GOMP_B * age))


def tech_pressure(t):
    """合成技術壓力向量"""
    p = np.zeros(N_DIM)
    for params in TECH_PARAMS.values():
        sigmoid = params['P_max'] / (1.0 + np.exp(-params['lam'] * (t - params['t_k'])))
        p += sigmoid * params['w']
    return p


def cognitive_resistance(s, age, n_agents, s_old):
    """三部分認知阻力"""
    # 1. 神經可塑性衰減
    r_plast = R_PLASTICITY_R0 * np.exp(R_PLASTICITY_KAPPA * age)

    # 2. 信息論頻寬約束（含自動化效應）
    info_load = n_agents * 0.3  # 每個代理基礎信息負載（歸一化）
    r_switch = R_SWITCH_C0 * info_load / (B_COG - info_load + 0.01)

    # 3. 習慣慣性
    r_habit = R_HABIT_ETA * np.sum((s - s_old)**2)

    return (r_plast + r_switch + r_habit) * np.ones(N_DIM)


def education_function(t, scenario='none'):
    """教育函數：不同情境下的教育投入"""
    e = np.zeros(N_DIM)
    if scenario == 'none':
        return e
    elif scenario == 'traditional':
        # 傳統教育：主要強化 WM 和 IX，對 MA/MR/EI/ID 投入少
        e = np.array([0.3, 0.1, 0.3, 0.05, 0.1, 0.1, 0.05])
    elif scenario == 'caie':
        # CAIE 教育模型：全維度均衡訓練
        e = np.array([0.4, 0.35, 0.4, 0.45, 0.35, 0.3, 0.3])
        # 漸進式導入（2025 年後逐步增強）
        ramp = 1.0 / (1.0 + np.exp(-0.5 * (t - 2025)))
        e *= ramp
    elif scenario == 'caie_late':
        # CAIE 教育但延遲 5 年（2030 年才開始）
        e = np.array([0.4, 0.35, 0.4, 0.45, 0.35, 0.3, 0.3])
        ramp = 1.0 / (1.0 + np.exp(-0.5 * (t - 2030)))
        e *= ramp
    return e


def rpe_drive(s, s_prev):
    """基於 RPE 的多巴胺驅動函數"""
    improvement = s - s_prev
    rpe = np.clip(improvement, -0.5, 0.5)
    return D0 + MU_RPE * rpe


# ============================================================
# 3. ODE 系統
# ============================================================

def caie_ode(t, state, age_at_t0, t0, scenario, n_agents):
    """
    CAIE 動力學方程
    state[:7] = S(t) 能力向量
    state[7:14] = D_cum(t) 累積損傷
    state[14:21] = proficiency(t) 熟練度
    """
    s = np.clip(state[:7], 0.01, None)
    d_cum = state[7:14]
    prof = np.clip(state[14:21], 0, 0.99)

    age = age_at_t0 + (t - t0)

    # 發育上界
    s_ceiling = s_max(age)

    # 技術壓力
    p = tech_pressure(t)

    # 壓力梯度（壓力方向）
    grad_p = np.clip(p - s, 0, None)

    # 教育
    e = education_function(t, scenario)

    # RPE 驅動（用簡化的壓力差當作 RPE 代理）
    d_drive = D0 + MU_RPE * np.clip(grad_p * 0.3, -0.3, 0.3)

    # 有效教育
    e_eff = e * d_drive * np.clip(1.0 - s / s_ceiling, 0, 1)

    # 認知阻力
    s_old = np.array([0.3, 0.3, 0.3, 0.1, 0.2, 0.4, 0.5])  # 前AI時代基線
    r = cognitive_resistance(s, age, n_agents, s_old)

    # 動力學方程
    ds_dt = ALPHA * e_eff + BETA * grad_p - GAMMA * r - DELTA * d_cum

    # 發育約束：不超過上界
    for i in range(N_DIM):
        if s[i] >= s_ceiling[i] and ds_dt[i] > 0:
            ds_dt[i] = 0

    # 累積損傷演化
    dd_cum = np.clip(p - s, 0, None) * 0.1

    # 熟練度演化
    dprof = PHI * e * (1.0 - prof)

    return np.concatenate([ds_dt, dd_cum, dprof])


# ============================================================
# 4. 模擬場景
# ============================================================

def run_simulation(scenario='none', age_start=25, n_agents=3,
                   t_span=(2000, 2040), label=''):
    """運行單一場景模擬"""
    # 初始條件：前 AI 時代的認知基線
    s0 = np.array([0.5, 0.4, 0.3, 0.1, 0.3, 0.5, 0.6])
    d_cum0 = np.zeros(N_DIM)
    prof0 = np.zeros(N_DIM)
    y0 = np.concatenate([s0, d_cum0, prof0])

    t_eval = np.linspace(t_span[0], t_span[1], 500)

    sol = solve_ivp(
        caie_ode, t_span, y0,
        args=(age_start, t_span[0], scenario, n_agents),
        t_eval=t_eval, method='RK45',
        max_step=0.1
    )
    return sol


def plot_scenario_comparison():
    """圖 1：三種教育場景對比"""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('CAIE 模型模擬：三種教育場景下的認知能力演化 (2000-2040)',
                 fontsize=16, fontweight='bold')

    scenarios = [
        ('none', '無教育干預', 'red', '--'),
        ('traditional', '傳統教育', 'blue', '-.'),
        ('caie', 'CAIE 教育模型', 'green', '-'),
    ]

    results = {}
    for sc, label, color, ls in scenarios:
        sol = run_simulation(scenario=sc)
        results[sc] = sol

    for i in range(N_DIM):
        ax = axes[i // 4][i % 4]
        for sc, label, color, ls in scenarios:
            sol = results[sc]
            ax.plot(sol.t, sol.y[i], color=color, linestyle=ls, label=label, linewidth=2)
        ax.set_title(DIM_LABELS[i], fontsize=12)
        ax.set_xlabel('年份')
        ax.set_ylabel('能力值')
        ax.axvline(x=2023, color='gray', linestyle=':', alpha=0.5, label='AI 時代起點')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(2000, 2040)

    # 第 8 格：技術壓力曲線
    ax = axes[1][3]
    t_range = np.linspace(1800, 2040, 500)
    total_pressure = np.array([np.sum(tech_pressure(t)) for t in t_range])
    ax.plot(t_range, total_pressure, 'k-', linewidth=2)
    ax.fill_between(t_range, 0, total_pressure, alpha=0.2, color='orange')
    ax.set_title('技術壓力總和', fontsize=12)
    ax.set_xlabel('年份')
    ax.set_ylabel('壓力強度')
    ax.axvline(x=1850, color='brown', linestyle=':', alpha=0.5, label='$T_1$ 燃氣機')
    ax.axvline(x=2007, color='blue', linestyle=':', alpha=0.5, label='$T_2$ 物聯網')
    ax.axvline(x=2023, color='red', linestyle=':', alpha=0.5, label='$T_3$ AI')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('docs/caie_fig1_scenarios.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("圖 1 已保存: docs/caie_fig1_scenarios.png")


def plot_age_effect():
    """圖 2：不同年齡群體的適應能力差異"""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('CAIE 模型模擬：不同年齡群體在 CAIE 教育下的認知適應 (2020-2040)',
                 fontsize=16, fontweight='bold')

    age_groups = [
        (15, '15 歲 (Z 世代)', '#2ecc71'),
        (25, '25 歲 (千禧世代)', '#3498db'),
        (40, '40 歲 (X 世代)', '#e67e22'),
        (55, '55 歲 (嬰兒潮)', '#e74c3c'),
    ]

    results = {}
    for age, label, color in age_groups:
        sol = run_simulation(scenario='caie', age_start=age, t_span=(2020, 2040))
        results[age] = sol

    for i in range(N_DIM):
        ax = axes[i // 4][i % 4]
        for age, label, color in age_groups:
            sol = results[age]
            ax.plot(sol.t, sol.y[i], color=color, linewidth=2, label=label)
        ax.set_title(DIM_LABELS[i], fontsize=12)
        ax.set_xlabel('年份')
        ax.set_ylabel('能力值')
        ax.axvline(x=2023, color='gray', linestyle=':', alpha=0.5)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # 第 8 格：累積損傷對比
    ax = axes[1][3]
    for age, label, color in age_groups:
        sol = results[age]
        total_damage = np.sum(sol.y[7:14], axis=0)
        ax.plot(sol.t, total_damage, color=color, linewidth=2, label=label)
    ax.set_title('累積認知損傷', fontsize=12)
    ax.set_xlabel('年份')
    ax.set_ylabel('損傷總量')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('docs/caie_fig2_age_effect.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("圖 2 已保存: docs/caie_fig2_age_effect.png")


def plot_agent_scaling():
    """圖 3：管理不同數量 AI 代理的認知負荷"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('CAIE 模型模擬：AI 代理數量 vs 認知負荷與生產力',
                 fontsize=16, fontweight='bold')

    # 3a: 認知阻力隨代理數量的變化
    ax = axes[0]
    n_range = np.linspace(1, 10, 100)
    for info_per_agent in [0.2, 0.3, 0.5]:
        resistance = R_SWITCH_C0 * (n_range * info_per_agent) / (B_COG - n_range * info_per_agent + 0.01)
        resistance = np.clip(resistance, 0, 5)
        ax.plot(n_range, resistance,
                label=f'r_i = {info_per_agent} ({"低" if info_per_agent == 0.2 else "中" if info_per_agent == 0.3 else "高"}信息負載)',
                linewidth=2)
    ax.set_xlabel('AI 代理數量 (n)')
    ax.set_ylabel('認知阻力 R_switch')
    ax.set_title('(a) 認知阻力 vs 代理數量')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 3)

    # 3b: 有效生產力 = n * (1 - R_switch/R_max)
    ax = axes[1]
    for info_per_agent, label in [(0.2, '訓練後 (索引式)'), (0.5, '未訓練 (全信息)')]:
        load = n_range * info_per_agent
        resistance = R_SWITCH_C0 * load / (B_COG - load + 0.01)
        resistance = np.clip(resistance, 0, 10)
        productivity = n_range * np.exp(-resistance)
        ax.plot(n_range, productivity, linewidth=2, label=label)
    ax.set_xlabel('AI 代理數量 (n)')
    ax.set_ylabel('有效生產力')
    ax.set_title('(b) 生產力峰值 (BCG 2026 驗證)')
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # BCG 數據標註
    ax.annotate('BCG 2026:\n生產力峰值 ≈ 3 工具',
                xy=(3, 2.5), fontsize=10,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    # 3c: 教育延遲效應
    ax = axes[2]
    delays = [0, 2, 5, 10]
    t_eval = np.linspace(2020, 2040, 200)

    for delay in delays:
        sol = run_simulation(
            scenario='caie' if delay == 0 else 'caie_late',
            t_span=(2020, 2040)
        )
        # 計算能力向量的 L2 範數
        s_norm = np.sqrt(np.sum(sol.y[:7]**2, axis=0))
        label = f'延遲 {delay} 年' if delay > 0 else '即時導入'
        ax.plot(sol.t, s_norm, linewidth=2, label=label)

    ax.set_xlabel('年份')
    ax.set_ylabel('認知能力總量 ||S||')
    ax.set_title('(c) 教育延遲效應')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('docs/caie_fig3_agents.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("圖 3 已保存: docs/caie_fig3_agents.png")


def plot_education_delay():
    """圖 4：教育延遲 τ 的臨界效應"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.suptitle('CAIE 預測：教育延遲 τ 對 2040 年認知水平的影響',
                 fontsize=14, fontweight='bold')

    # 模擬不同延遲
    delays = np.arange(0, 15, 1)
    final_s_norms = []

    for delay in delays:
        # 修改 CAIE 教育的啟動時間
        s0 = np.array([0.5, 0.4, 0.3, 0.1, 0.3, 0.5, 0.6])
        d_cum0 = np.zeros(N_DIM)
        prof0 = np.zeros(N_DIM)
        y0 = np.concatenate([s0, d_cum0, prof0])

        def ode_delayed(t, state):
            s = np.clip(state[:7], 0.01, None)
            d_cum = state[7:14]
            prof = np.clip(state[14:21], 0, 0.99)
            age = 25 + (t - 2020)
            s_ceil = s_max(age)
            p = tech_pressure(t)
            grad_p = np.clip(p - s, 0, None)

            # 教育在 2023+delay 年開始
            e_base = np.array([0.4, 0.35, 0.4, 0.45, 0.35, 0.3, 0.3])
            ramp = 1.0 / (1.0 + np.exp(-0.5 * (t - (2023 + delay))))
            e = e_base * ramp

            d_drive = D0 + MU_RPE * np.clip(grad_p * 0.3, -0.3, 0.3)
            e_eff = e * d_drive * np.clip(1.0 - s / s_ceil, 0, 1)
            s_old = np.array([0.3, 0.3, 0.3, 0.1, 0.2, 0.4, 0.5])
            r = cognitive_resistance(s, age, 3, s_old)

            ds_dt = ALPHA * e_eff + BETA * grad_p - GAMMA * r - DELTA * d_cum
            for i in range(N_DIM):
                if s[i] >= s_ceil[i] and ds_dt[i] > 0:
                    ds_dt[i] = 0

            dd_cum = np.clip(p - s, 0, None) * 0.1
            dprof = PHI * e * (1.0 - prof)
            return np.concatenate([ds_dt, dd_cum, dprof])

        sol = solve_ivp(ode_delayed, (2020, 2040), y0,
                        t_eval=[2040], method='RK45', max_step=0.1)
        final_s = sol.y[:7, -1]
        final_s_norms.append(np.sqrt(np.sum(final_s**2)))

    ax.plot(delays, final_s_norms, 'bo-', linewidth=2, markersize=8)
    ax.axvline(x=1.25, color='red', linestyle='--', linewidth=2,
               label=f'臨界延遲 τ* = 1/λ₃ ≈ 1.25 年')
    ax.fill_between(delays, min(final_s_norms), final_s_norms,
                    where=[d <= 1.25 for d in delays], alpha=0.2, color='green',
                    label='安全窗口')
    ax.fill_between(delays, min(final_s_norms), final_s_norms,
                    where=[d > 1.25 for d in delays], alpha=0.2, color='red',
                    label='危險區域')
    ax.set_xlabel('教育延遲 τ（年）', fontsize=12)
    ax.set_ylabel('2040 年認知能力總量 ||S||', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('docs/caie_fig4_delay.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("圖 4 已保存: docs/caie_fig4_delay.png")


# ============================================================
# 5. 主程式
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("CAIE 模型模擬器")
    print("Cognitive Adaptation in the Intelligence Era")
    print("=" * 60)

    print("\n[1/4] 生成圖 1：三種教育場景對比...")
    plot_scenario_comparison()

    print("\n[2/4] 生成圖 2：年齡效應...")
    plot_age_effect()

    print("\n[3/4] 生成圖 3：代理數量與認知負荷...")
    plot_agent_scaling()

    print("\n[4/4] 生成圖 4：教育延遲臨界效應...")
    plot_education_delay()

    print("\n" + "=" * 60)
    print("所有模擬完成。圖表已保存至 docs/ 目錄。")
    print("=" * 60)
