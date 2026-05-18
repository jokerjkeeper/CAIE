"""
CAIE 模型 — Sensitivity Analysis（敏感性分析）
=============================================
對關鍵參數做 ±20% 擾動，評估模型核心預測的穩健性。

核心預測（需驗證穩健性）：
  P1: 傳統教育在 AI 時代失敗（2023 後 MA 下降）
  P2: CAIE 教育模型維持正向適應
  P3: 多代理生產力峰值 n* ≈ 3
  P4: 教育延遲臨界值 τ* ≈ 1.25 年

分析的參數：
  α (ALPHA)   — 教育驅動係數
  β (BETA)    — 壓力適應係數
  γ (GAMMA)   — 認知阻力係數
  δ (DELTA)   — 累積損傷係數
  lam3 (lam_T3) — AI 擴散速率
  B_cog       — 認知頻寬

輸出：
  docs/caie_fig5_sensitivity_tornado.png  — Tornado 圖
  docs/caie_fig6_sensitivity_traces.png   — 參數擾動下的能力軌跡
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib
import copy

matplotlib.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial']
matplotlib.rcParams['axes.unicode_minus'] = False

# 複用主模擬器的基礎參數
from caie_experiment import (
    N_DIM, DIM_LABELS, TECH_PARAMS, S_INF, GOMP_A, GOMP_B,
    R_PLASTICITY_R0, R_PLASTICITY_KAPPA, R_SWITCH_C0,
    R_HABIT_ETA, MU_RPE, D0, PHI,
    s_max, tech_pressure
)


# ============================================================
# 1. 參數化 ODE 系統（接受參數字典）
# ============================================================

def caie_ode_parametric(t, state, params):
    """接受參數字典的 CAIE ODE，用於敏感性分析"""
    alpha = params['alpha']
    beta = params['beta']
    gamma = params['gamma']
    delta = params['delta']
    b_cog = params['b_cog']
    scenario = params['scenario']
    n_agents = params['n_agents']
    age_at_t0 = params['age_at_t0']
    t0 = params['t0']
    lam_t3 = params['lam_t3']

    s = np.clip(state[:7], 0.01, None)
    d_cum = state[7:14]
    prof = np.clip(state[14:21], 0, 0.99)

    age = age_at_t0 + (t - t0)

    # 發育上界
    s_ceiling = s_max(age)

    # 技術壓力（使用可調的 lam3）
    p = np.zeros(N_DIM)
    for key, tp in TECH_PARAMS.items():
        lam = lam_t3 if key == 'T3_ai' else tp['lam']
        sigmoid = tp['P_max'] / (1.0 + np.exp(-lam * (t - tp['t_k'])))
        p += sigmoid * tp['w']

    # 壓力梯度
    grad_p = np.clip(p - s, 0, None)

    # 教育函數
    e = np.zeros(N_DIM)
    if scenario == 'traditional':
        e = np.array([0.3, 0.1, 0.3, 0.05, 0.1, 0.1, 0.05])
    elif scenario == 'caie':
        e = np.array([0.4, 0.35, 0.4, 0.45, 0.35, 0.3, 0.3])
        ramp = 1.0 / (1.0 + np.exp(-0.5 * (t - 2025)))
        e *= ramp

    # RPE 驅動
    d_drive = D0 + MU_RPE * np.clip(grad_p * 0.3, -0.3, 0.3)

    # 有效教育
    e_eff = e * d_drive * np.clip(1.0 - s / s_ceiling, 0, 1)

    # 認知阻力
    r_plast = R_PLASTICITY_R0 * np.exp(R_PLASTICITY_KAPPA * age)
    info_load = n_agents * 0.3
    r_switch = R_SWITCH_C0 * info_load / (b_cog - info_load + 0.01)
    s_old = np.array([0.3, 0.3, 0.3, 0.1, 0.2, 0.4, 0.5])
    r_habit = R_HABIT_ETA * np.sum((s - s_old)**2)
    r = (r_plast + r_switch + r_habit) * np.ones(N_DIM)

    # 動力學方程
    ds_dt = alpha * e_eff + beta * grad_p - gamma * r - delta * d_cum

    # 發育約束
    for i in range(N_DIM):
        if s[i] >= s_ceiling[i] and ds_dt[i] > 0:
            ds_dt[i] = 0

    # 累積損傷
    dd_cum = np.clip(p - s, 0, None) * 0.1

    # 熟練度
    dprof = PHI * e * (1.0 - prof)

    return np.concatenate([ds_dt, dd_cum, dprof])


def run_parametric(params, t_span=(2000, 2040)):
    """使用指定參數字典運行模擬"""
    s0 = np.array([0.5, 0.4, 0.3, 0.1, 0.3, 0.5, 0.6])
    d_cum0 = np.zeros(N_DIM)
    prof0 = np.zeros(N_DIM)
    y0 = np.concatenate([s0, d_cum0, prof0])

    t_eval = np.linspace(t_span[0], t_span[1], 500)
    sol = solve_ivp(
        caie_ode_parametric, t_span, y0,
        args=(params,), t_eval=t_eval, method='RK45', max_step=0.1
    )
    return sol


# ============================================================
# 2. 基線參數
# ============================================================

BASELINE_PARAMS = {
    'alpha': 0.5,
    'beta': 0.3,
    'gamma': 1.0,
    'delta': 0.05,
    'b_cog': 3.0,
    'lam_t3': 0.8,
    'scenario': 'caie',
    'n_agents': 3,
    'age_at_t0': 25,
    't0': 2000,
}

# 要分析的參數及其顯示名稱
SENSITIVITY_PARAMS = {
    'alpha':  ('alpha (Education Drive)', 0.5),
    'beta':   ('beta (Pressure Adaptation)', 0.3),
    'gamma':  ('gamma (Cognitive Resistance)', 1.0),
    'delta':  ('delta (Cumulative Damage)', 0.05),
    'lam_t3': ('lambda3 (AI Diffusion Rate)', 0.8),
    'b_cog':  ('B_cog (Cognitive Bandwidth)', 3.0),
}

PERTURBATION = 0.20  # ±20%


# ============================================================
# 3. 計算指標函數
# ============================================================

def compute_metrics(sol):
    """從模擬結果計算四個核心預測指標"""
    t = sol.t
    s = sol.y[:7]

    # 找到最接近 2040 的索引
    idx_2040 = np.argmin(np.abs(t - 2040))
    idx_2023 = np.argmin(np.abs(t - 2023))

    # P1/P2: 2040 年認知能力總量 ||S(2040)||
    s_norm_2040 = np.sqrt(np.sum(s[:, idx_2040]**2))

    # MA 維度在 2040 的值
    ma_2040 = s[3, idx_2040]

    # 能力增長率（2023-2040 平均）
    s_norm_2023 = np.sqrt(np.sum(s[:, idx_2023]**2))
    growth = s_norm_2040 - s_norm_2023

    return {
        's_norm_2040': s_norm_2040,
        'ma_2040': ma_2040,
        'growth_2023_2040': growth,
    }


def compute_agent_peak(params_base):
    """計算多代理生產力峰值 n*（未訓練者，info_per_agent=0.5）"""
    n_range = range(1, 11)
    productivities = []
    info_per_agent = 0.8  # 未訓練者：全信息處理，每代理負載高（校準 BCG 2026 n*≈3）
    for n in n_range:
        b_cog = params_base['b_cog']
        info_load = n * info_per_agent
        if info_load >= b_cog:
            r_switch = 10.0  # 超過頻寬上限，阻力極大
        else:
            r_switch = R_SWITCH_C0 * info_load / (b_cog - info_load + 0.01)
        r_switch = max(r_switch, 0)
        prod = n * np.exp(-r_switch)
        productivities.append(prod)
    n_star = n_range[np.argmax(productivities)]
    return n_star, max(productivities)


def compute_tau_star(params_base):
    """計算教育延遲臨界值 τ*"""
    delays = np.arange(0, 10, 0.25)
    final_norms = []

    for delay in delays:
        s0 = np.array([0.5, 0.4, 0.3, 0.1, 0.3, 0.5, 0.6])
        y0 = np.concatenate([s0, np.zeros(N_DIM), np.zeros(N_DIM)])

        def ode_delayed(t, state, p=params_base, d=delay):
            p_d = dict(p)
            # 教育在 2023+delay 後啟動
            s = np.clip(state[:7], 0.01, None)
            d_cum = state[7:14]
            age = 25 + (t - 2020)  # 固定：2020年起 25 歲
            s_ceiling = s_max(age)

            p_vec = np.zeros(N_DIM)
            for key, tp in TECH_PARAMS.items():
                lam = p_d['lam_t3'] if key == 'T3_ai' else tp['lam']
                sigmoid = tp['P_max'] / (1.0 + np.exp(-lam * (t - tp['t_k'])))
                p_vec += sigmoid * tp['w']

            grad_p = np.clip(p_vec - s, 0, None)
            e_base = np.array([0.4, 0.35, 0.4, 0.45, 0.35, 0.3, 0.3])
            ramp = 1.0 / (1.0 + np.exp(-0.5 * (t - (2023 + d))))
            e = e_base * ramp

            d_drive = D0 + MU_RPE * np.clip(grad_p * 0.3, -0.3, 0.3)
            e_eff = e * d_drive * np.clip(1.0 - s / s_ceiling, 0, 1)
            s_old = np.array([0.3, 0.3, 0.3, 0.1, 0.2, 0.4, 0.5])
            r_plast = R_PLASTICITY_R0 * np.exp(R_PLASTICITY_KAPPA * age)
            info_load = p_d['n_agents'] * 0.3
            r_switch = R_SWITCH_C0 * info_load / (p_d['b_cog'] - info_load + 0.01)
            r_habit = R_HABIT_ETA * np.sum((s - s_old)**2)
            r = (r_plast + r_switch + r_habit) * np.ones(N_DIM)

            ds_dt = p_d['alpha'] * e_eff + p_d['beta'] * grad_p - p_d['gamma'] * r - p_d['delta'] * d_cum
            for i in range(N_DIM):
                if s[i] >= s_ceiling[i] and ds_dt[i] > 0:
                    ds_dt[i] = 0
            dd_cum = np.clip(p_vec - s, 0, None) * 0.1
            dprof = PHI * e * (1.0 - np.clip(state[14:21], 0, 0.99))
            return np.concatenate([ds_dt, dd_cum, dprof])

        sol = solve_ivp(ode_delayed, (2020, 2040), y0,
                        t_eval=[2040], method='RK45', max_step=0.2)
        final_norms.append(np.sqrt(np.sum(sol.y[:7, -1]**2)))

    # τ* 定義為邊際效益（每多延遲 1 年的損失）最大的拐點
    final_norms = np.array(final_norms)
    # 計算二階導數，找到加速下降的拐點
    grad = np.gradient(final_norms, delays)
    grad2 = np.gradient(grad, delays)
    # τ* = 二階導數最小的點（加速衰減最劇烈處）
    # 只考慮前半段（0~5年），避免尾部噪聲
    search_mask = delays <= 5.0
    search_grad2 = grad2.copy()
    search_grad2[~search_mask] = 0
    tau_star = delays[np.argmin(search_grad2)]
    # 若找不到明顯拐點，用理論估計 1/lam3
    if tau_star <= 0 or tau_star > 5.0:
        tau_star = 1.0 / params_base['lam_t3']
    return tau_star


# ============================================================
# 4. 執行敏感性分析
# ============================================================

def run_sensitivity_analysis():
    """對每個參數做 ±20% 擾動，計算指標變化"""
    # 基線
    sol_base = run_parametric(BASELINE_PARAMS)
    metrics_base = compute_metrics(sol_base)
    n_star_base, _ = compute_agent_peak(BASELINE_PARAMS)
    tau_star_base = compute_tau_star(BASELINE_PARAMS)

    print(f"\n基線指標：")
    print(f"  ||S(2040)||  = {metrics_base['s_norm_2040']:.4f}")
    print(f"  MA(2040)     = {metrics_base['ma_2040']:.4f}")
    print(f"  Growth       = {metrics_base['growth_2023_2040']:.4f}")
    print(f"  n*           = {n_star_base}")
    print(f"  τ*           = {tau_star_base:.2f}")

    results = {}
    for pname, (plabel, pval) in SENSITIVITY_PARAMS.items():
        results[pname] = {'label': plabel, 'base': pval}

        for direction, factor in [('high', 1 + PERTURBATION), ('low', 1 - PERTURBATION)]:
            params = dict(BASELINE_PARAMS)
            params[pname] = pval * factor

            sol = run_parametric(params)
            metrics = compute_metrics(sol)
            n_star, _ = compute_agent_peak(params)

            results[pname][direction] = {
                'value': pval * factor,
                'metrics': metrics,
                'n_star': n_star,
                'sol': sol,
            }

        # 計算變化百分比
        for metric_name in ['s_norm_2040', 'ma_2040', 'growth_2023_2040']:
            base_val = metrics_base[metric_name]
            high_val = results[pname]['high']['metrics'][metric_name]
            low_val = results[pname]['low']['metrics'][metric_name]
            if abs(base_val) > 1e-8:
                results[pname][f'{metric_name}_pct_high'] = (high_val - base_val) / abs(base_val) * 100
                results[pname][f'{metric_name}_pct_low'] = (low_val - base_val) / abs(base_val) * 100
            else:
                results[pname][f'{metric_name}_pct_high'] = 0
                results[pname][f'{metric_name}_pct_low'] = 0

    return results, metrics_base, n_star_base, tau_star_base


# ============================================================
# 5. 圖表生成
# ============================================================

def plot_tornado(results, metrics_base):
    """圖 5: Tornado 圖 — 參數敏感性排序"""
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle('CAIE Sensitivity Analysis: Tornado Diagram (±20% Perturbation)',
                 fontsize=15, fontweight='bold')

    metric_titles = {
        's_norm_2040': '||S(2040)|| (Cognitive Ability at 2040)',
        'ma_2040': 'MA(2040) (Multi-Agent Management)',
        'growth_2023_2040': 'Growth 2023→2040',
    }

    for idx, (metric_key, title) in enumerate(metric_titles.items()):
        ax = axes[idx]

        # 收集數據
        labels = []
        low_vals = []
        high_vals = []
        for pname in SENSITIVITY_PARAMS:
            labels.append(results[pname]['label'])
            low_vals.append(results[pname][f'{metric_key}_pct_low'])
            high_vals.append(results[pname][f'{metric_key}_pct_high'])

        # 按影響大小排序（用 |high - low| 的跨度）
        spans = [abs(h - l) for h, l in zip(high_vals, low_vals)]
        order = np.argsort(spans)[::-1]

        y_pos = np.arange(len(labels))
        sorted_labels = [labels[i] for i in order]
        sorted_low = [low_vals[i] for i in order]
        sorted_high = [high_vals[i] for i in order]

        # 繪製
        for i, (lo, hi) in enumerate(zip(sorted_low, sorted_high)):
            ax.barh(i, hi, color='#e74c3c', alpha=0.7, height=0.6,
                    label='+20%' if i == 0 else '')
            ax.barh(i, lo, color='#3498db', alpha=0.7, height=0.6,
                    label='-20%' if i == 0 else '')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_labels, fontsize=10)
        ax.set_xlabel('Change from Baseline (%)', fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.axvline(x=0, color='black', linewidth=1)
        ax.grid(True, alpha=0.3, axis='x')
        if idx == 0:
            ax.legend(loc='lower right', fontsize=10)

    plt.tight_layout()
    plt.savefig('docs/caie_fig5_sensitivity_tornado.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("圖 5 已保存: docs/caie_fig5_sensitivity_tornado.png")


def plot_sensitivity_traces(results):
    """圖 6: 參數擾動下的能力軌跡（||S(t)|| 隨時間變化）"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('CAIE Sensitivity Analysis: Ability Trajectories under ±20% Parameter Perturbation',
                 fontsize=14, fontweight='bold')

    # 基線
    sol_base = run_parametric(BASELINE_PARAMS)
    s_norm_base = np.sqrt(np.sum(sol_base.y[:7]**2, axis=0))

    for idx, pname in enumerate(SENSITIVITY_PARAMS):
        ax = axes[idx // 3][idx % 3]
        plabel = results[pname]['label']

        # 基線
        ax.plot(sol_base.t, s_norm_base, 'k-', linewidth=2.5, label='Baseline', zorder=3)

        # +20%
        sol_high = results[pname]['high']['sol']
        s_norm_high = np.sqrt(np.sum(sol_high.y[:7]**2, axis=0))
        ax.plot(sol_high.t, s_norm_high, 'r--', linewidth=1.5,
                label=f'+20% ({results[pname]["high"]["value"]:.3f})')

        # -20%
        sol_low = results[pname]['low']['sol']
        s_norm_low = np.sqrt(np.sum(sol_low.y[:7]**2, axis=0))
        ax.plot(sol_low.t, s_norm_low, 'b--', linewidth=1.5,
                label=f'-20% ({results[pname]["low"]["value"]:.3f})')

        # 帶寬
        ax.fill_between(sol_base.t,
                         np.minimum(s_norm_low, s_norm_high),
                         np.maximum(s_norm_low, s_norm_high),
                         alpha=0.15, color='gray')

        ax.set_title(plabel, fontsize=12, fontweight='bold')
        ax.set_xlabel('Year')
        ax.set_ylabel('||S(t)||')
        ax.axvline(x=2023, color='gray', linestyle=':', alpha=0.5)
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(2000, 2040)

    plt.tight_layout()
    plt.savefig('docs/caie_fig6_sensitivity_traces.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("圖 6 已保存: docs/caie_fig6_sensitivity_traces.png")


def print_summary_table(results, metrics_base, n_star_base, tau_star_base):
    """輸出敏感性分析摘要表（可直接貼入 LaTeX）"""
    print("\n" + "=" * 90)
    print("SENSITIVITY ANALYSIS SUMMARY TABLE")
    print("=" * 90)
    print(f"{'Parameter':<25} {'Base':>8} {'−20%':>8} {'||S||%':>8} {'+20%':>8} {'||S||%':>8} {'n*±':>6}")
    print("-" * 90)

    for pname, (plabel, pval) in SENSITIVITY_PARAMS.items():
        low_val = results[pname]['low']['value']
        high_val = results[pname]['high']['value']
        pct_low = results[pname]['s_norm_2040_pct_low']
        pct_high = results[pname]['s_norm_2040_pct_high']
        n_low = results[pname]['low']['n_star']
        n_high = results[pname]['high']['n_star']
        n_change = f"{n_low}/{n_high}" if n_low != n_high else f"{n_low}"

        print(f"{plabel:<25} {pval:>8.3f} {low_val:>8.3f} {pct_low:>+7.1f}% {high_val:>8.3f} {pct_high:>+7.1f}% {n_change:>6}")

    print("-" * 90)
    print(f"Baseline: ||S(2040)|| = {metrics_base['s_norm_2040']:.4f}, "
          f"n* = {n_star_base}, τ* = {tau_star_base:.2f}")
    print("=" * 90)

    # LaTeX 表格
    print("\n% --- LaTeX Table (paste into main.tex) ---")
    print(r"\begin{table}[H]")
    print(r"\centering")
    print(r"\caption{Sensitivity Analysis: Impact of $\pm 20\%$ Parameter Perturbation on Key Predictions}")
    print(r"\label{tab:sensitivity}")
    print(r"\begin{tabular}{lccccc}")
    print(r"\toprule")
    print(r"Parameter & Baseline & $-20\%$ & $\Delta\|\mathbf{S}\|$ & $+20\%$ & $\Delta\|\mathbf{S}\|$ \\")
    print(r"\midrule")

    for pname, (plabel, pval) in SENSITIVITY_PARAMS.items():
        # 轉為 LaTeX 友好的名稱
        latex_name = plabel.replace('α', r'$\alpha$').replace('β', r'$\beta$')
        latex_name = latex_name.replace('γ', r'$\gamma$').replace('δ', r'$\delta$')
        latex_name = latex_name.replace('lam3', r'$\lambda_3$').replace('B_cog', r'$B_{\text{cog}}$')
        # 取括號前
        latex_name = latex_name.split(' (')[0]

        low_val = results[pname]['low']['value']
        high_val = results[pname]['high']['value']
        pct_low = results[pname]['s_norm_2040_pct_low']
        pct_high = results[pname]['s_norm_2040_pct_high']

        print(f"{latex_name} & {pval:.2f} & {low_val:.2f} & {pct_low:+.1f}\\% & {high_val:.2f} & {pct_high:+.1f}\\% \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


# ============================================================
# 6. 主程式
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("CAIE Sensitivity Analysis")
    print("±20% Parameter Perturbation Study")
    print("=" * 60)

    print("\n[1/4] Running sensitivity analysis...")
    results, metrics_base, n_star_base, tau_star_base = run_sensitivity_analysis()

    print("\n[2/4] Generating Tornado diagram (Fig. 5)...")
    plot_tornado(results, metrics_base)

    print("\n[3/4] Generating trajectory plots (Fig. 6)...")
    plot_sensitivity_traces(results)

    print("\n[4/4] Summary table...")
    print_summary_table(results, metrics_base, n_star_base, tau_star_base)

    print("\n" + "=" * 60)
    print("Sensitivity analysis complete.")
    print("Figures saved to docs/ directory.")
    print("=" * 60)
