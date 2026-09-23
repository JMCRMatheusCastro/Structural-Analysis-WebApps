import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
import scipy.integrate as integrate
import pandas as pd

# ==========================================
# 0. CONFIGURAÇÕES DA PÁGINA STREAMLIT
# ==========================================
st.set_page_config(page_title="ColImpact", layout="wide")

st.title("🏗️ ColImpact (Column Impact Analysis Tool)")
st.markdown("""
**Programador:** José Matheus de Castro Rodrigues  
Este código foi desenvolvido para analisar o comportamento não linear de pilares de concreto armado sujeitos a carregamentos laterais, incluindo os efeitos de impacto de veículos através de Forças Estáticas Equivalentes (FEE).
""")

# ==========================================
# 2. FUNÇÕES DE PROCESSAMENTO (NÚCLEO MATEMÁTICO)
# ==========================================
def DeslocamentoLateralENGASTE(nNos, DeltaZ, Nk, q, QT, MT, EIef, Pos_Fveic=None, Fveic=0):
    A = lil_matrix((nNos+1, nNos+1))
    B = np.zeros((nNos+1, 1))
    tem_impacto = (Pos_Fveic is not None) and (Fveic != 0)
    if tem_impacto:
        No_impacto_fisico = int(round(Pos_Fveic / DeltaZ)) + 1
        No_impacto_matriz = No_impacto_fisico - 2 

    for i in range(nNos+1):
        if i == 0:
            A[i, 0] = (EIef[0]) + (-2*Nk*(DeltaZ**2)+EIef[0]+4*EIef[1]+  EIef[2])
            A[i, 1] =                 Nk*(DeltaZ**2)        -2*EIef[1]-2*EIef[2]
            A[i, 2] =                                                    EIef[2]
            B[i] = q * DeltaZ**4
        elif i == 1:
            A[i, 0] =    Nk*DeltaZ**2-2*EIef[1]-2*EIef[2]
            A[i, 1] = -2*Nk*DeltaZ**2+  EIef[1]+4*EIef[2]+  EIef[3]
            A[i, 2] =    Nk*DeltaZ**2          -2*EIef[2]-2*EIef[3]
            A[i, 3] =                                       EIef[3]
            B[i] = q * DeltaZ**4
        elif i == nNos-1:
            A[i, i+1] = 0
            A[i, i]   =  -EIef[-2]
            A[i, i-1] = 2*EIef[-2]
            A[i, i-2] =  -EIef[-2]
            A[i, i-3] = 0
            B[i] = (MT * DeltaZ**2)
        elif i == nNos:
            A[i, i]   =                  -EIef[-1]
            A[i, i-1] = -Nk*(DeltaZ**2)+2*EIef[-1]
            A[i, i-2] =                  -EIef[-1]+  EIef[-3]
            A[i, i-3] =  Nk*(DeltaZ**2)           -2*EIef[-3]
            A[i, i-4] =                              EIef[-3]
            B[i] = 2 * QT * DeltaZ**3
        else:
            A[i, i-2] =                     EIef[i]
            A[i, i-1] =    Nk*(DeltaZ**2)-2*EIef[i]-2*EIef[i+1]
            A[i, i]   = -2*Nk*(DeltaZ**2)+  EIef[i]+4*EIef[i+1]+  EIef[i+2]
            A[i, i+1] =    Nk*(DeltaZ**2)          -2*EIef[i+1]-2*EIef[i+2]
            A[i, i+2] =                                           EIef[i+2]
            if tem_impacto and i == No_impacto_matriz:
                B[i] = q*(DeltaZ**4) + Fveic * DeltaZ**3 
            else:
                B[i] = q*(DeltaZ**4)

    A = A.tocsr()
    Resultado = spsolve(A, B)
    Resultado = Resultado[:-2]
    W = np.zeros((nNos, 1))
    W[1:nNos,0] = Resultado
    return W

def DeslocamentoLateralBIAPOIADOS(nNos, DeltaZ, Nk, q, MB, MT, EIef, Pos_Fveic=None, Fveic=0):
    A = lil_matrix((nNos+2, nNos+2))
    B = np.zeros((nNos+2, 1))
    tem_impacto = (Pos_Fveic is not None) and (Fveic != 0)
    if tem_impacto:
        No_impacto_fisico = int(round(Pos_Fveic / DeltaZ)) + 2
        No_impacto_matriz = No_impacto_fisico - 1

    for i in range(nNos+2):
        if i == 0:
            A[i, 1] = -EIef[1]
            A[i, 2] = -EIef[1]
            B[i] = MB * DeltaZ**2
        elif i == 1:
            A[i, 0] =                   EIef[0]
            A[i, 1] =  Nk*(DeltaZ**2)-2*EIef[0]-2*EIef[1]
            A[i, 2] =  Nk*(DeltaZ**2)          -2*EIef[1]-2*EIef[2]
            A[i, 3] =                                       EIef[2]
            B[i] = q * DeltaZ**4
        elif i == 2:
            A[i, 1] =                   EIef[1]
            A[i, 2] = -2*Nk*(DeltaZ**2)+EIef[1]+4*EIef[2]+  EIef[3]
            A[i, 3] =    Nk*(DeltaZ**2)        -2*EIef[2]-2*EIef[3]
            A[i, 4] =                                       EIef[3]
            B[i] = q * DeltaZ**4
        elif i == 3:
            A[i, 2] =    Nk*(DeltaZ**2)-2*EIef[2]-2*EIef[3]
            A[i, 3] = -2*Nk*(DeltaZ**2)  +EIef[2]+4*EIef[3]+  EIef[4]
            A[i, 4] =    Nk*(DeltaZ**2)          -2*EIef[3]-2*EIef[4]
            A[i, 5] =                                         EIef[4]
            B[i] = q * DeltaZ**4
        elif i == nNos-2:
            A[i, i+1] =    Nk*(DeltaZ**2)           -2*EIef[-4]-2*EIef[-3]
            A[i, i]   = -2*Nk*(DeltaZ**2)+  EIef[-5]+4*EIef[-4]+  EIef[-3]
            A[i, i-1] =    Nk*(DeltaZ**2)-2*EIef[-5]-2*EIef[-4]
            A[i, i-2] =                     EIef[-5]
            B[i] = q * DeltaZ**4
        elif i == nNos-1:
            A[i, i+1] =                                         EIef[-2]
            A[i, i]   = -2*Nk*(DeltaZ**2)+  EIef[-4]+4*EIef[-3]+EIef[-2]
            A[i, i-1] =    Nk*(DeltaZ**2)-2*EIef[-4]-2*EIef[-3]
            A[i, i-2] =                     EIef[-4]
            B[i] = q * DeltaZ**4
        elif i == nNos:
            A[i, i+1] =                                         EIef[-1]
            A[i, i]   =  Nk*(DeltaZ**2)           -2*EIef[-2]-2*EIef[-1]
            A[i, i-1] =  Nk*(DeltaZ**2)-2*EIef[-3]-2*EIef[-2]
            A[i, i-2] =                   EIef[-3]
            B[i] = q * DeltaZ**4
        elif i == nNos+1:
            A[i, i-1] = - EIef[-2]
            A[i, i-2] = - EIef[-2]
            B[i] = MT * DeltaZ**2
        else:
            A[i, i-2] =                     EIef[i-1]
            A[i, i-1] =    Nk*(DeltaZ**2)-2*EIef[i-1]-2*EIef[i]
            A[i, i]   = -2*Nk*(DeltaZ**2)+  EIef[i-1]+4*EIef[i]+  EIef[i+1]
            A[i, i+1] =    Nk*(DeltaZ**2)            -2*EIef[i]-2*EIef[i+1]
            A[i, i+2] =                                           EIef[i+1]
            if tem_impacto and i == No_impacto_matriz:
                B[i] = q * DeltaZ**4 + Fveic * DeltaZ**3
            else:
                B[i] = q * DeltaZ**4

    A = A.tocsr()
    Resultado = spsolve(A, B)
    Resultado = Resultado[2:-2]
    Resultado = Resultado.reshape(-1, 1)
    W = np.vstack([0, Resultado, 0])
    return W

def MomentoFletorENGASTE(nNos, DeltaZ, EIef, MT, W):
    M = np.zeros(nNos)
    for i in range(1, nNos-1):
        M[i] = -EIef[i] * (W[i-1,0] - 2*W[i,0] + W[i+1,0]) / DeltaZ**2
    M[0] = -EIef[0] * (2 * W[1,0]) / DeltaZ**2
    M[-1] = MT
    return M

def MomentoFletorBIAPOIADO(nNos, DeltaZ, EIef, MB, MT, W):
    M = np.zeros(nNos)
    for i in range(1, nNos-1):
        M[i] = -EIef[i] * (W[i-1,0] - 2*W[i,0] + W[i+1,0]) / DeltaZ**2
    M[0] = MB
    M[-1] = MT
    return M

def RigidezFlexaoConstante(b, h, r, fck, E, coef):
    if r != 0:
        I = (np.pi * r**4)/4
    else:
        I = (b * h**3)/12
    if E != 0:
        EIef = coef * I * E
    else:
        Ei = 5600 *(fck/1000)**0.5
        Ecs = (0.8 + 0.2 * ((fck/1000) / 80))*Ei
        EIef = coef * I * Ecs * 1000
    return EIef

def rigidezsecante(χ_values, MF_values, MF, Mr):
    interp_chi = interp1d(MF_values, χ_values, kind='quadratic', fill_value="extrapolate")
    MF_limited = np.minimum(MF, MF_values[-1])
    χ_interp = interp_chi(MF_limited)
    χ_fiss = interp_chi(Mr)
    EI_const = Mr / χ_fiss
    EI_values = np.where(MF > Mr, MF / χ_interp, EI_const)
    return EI_values

def ε(χ,d,εm):
    return εm + χ*(d)

def εlim(d, x, h, ε0, εu, k, ds):
    xlim = (εu / (εu + 0.01)) * ds[0]
    if isinstance(d, np.ndarray):
        epsilon = np.full_like(d, np.nan, dtype=float)
    else:
        epsilon = np.nan
    x_scalar = x.item() if isinstance(x, np.ndarray) and x.size == 1 else x
    if np.isnan(x_scalar): return epsilon

    if 0 <= x_scalar < xlim:
        denominator = ds[0] - x_scalar
        if np.isclose(denominator, 0.0):
            epsilon = np.inf * np.sign(x_scalar - d)
        else:
            epsilon = 0.01 * ((x_scalar - d) / denominator)
    elif xlim <= x_scalar <= h:
        if np.isclose(x_scalar, 0.0):
            epsilon = np.inf * np.sign(x_scalar - d)
        else:
            epsilon = εu * ((x_scalar - d) / x_scalar)
    elif x_scalar > h:
        denominator = x_scalar - k*h
        if np.isclose(denominator, 0.0):
            epsilon = np.inf * np.sign(x_scalar - d)
        else:
            epsilon = ε0 * ((x_scalar - d) / denominator)
    return epsilon

def σconcrete(εc, ε0, εu, fcd, coef):
    σc = np.zeros_like(εc)
    fck_MPa = (fcd * 1.4) / 1000
    ηc = 1.0 if fck_MPa <= 40 else (40/fck_MPa)**(1/3)
    n = 2.0 if fck_MPa <= 50 else 1.4 + 23.4 * ((90 - fck_MPa)/100)**4

    mask_parabolic = (εc > 0) & (εc <= ε0)
    mask_rectangular = (εc > ε0) & (εc <= εu)

    if coef == 1.1:
        σc[mask_parabolic] = 1.1 * fcd * (1 - (1 - εc[mask_parabolic]/ε0)**n)
        σc[mask_rectangular] = 1.1 * fcd
    else:
        σc[mask_parabolic] = coef * ηc * fcd * (1 - (1 - εc[mask_parabolic]/ε0)**n)
        σc[mask_rectangular] = coef * ηc * fcd
    return σc

def σsteel(εs,fyd,Es):
    εyd = fyd/Es
    σsd = np.zeros_like(εs)
    for i in range(len(εs)):
        if abs(εs[i]) <= εyd:
            σsd[i]=Es*εs[i]
        else:
            σsd[i]=np.sign(εs[i])*fyd
    return σsd

def EquiDasForcasEMomentos(x, b, h, ds, n, fcd, fyd, Es, ε0, εu, k, N, M):
    dc = np.arange(0, h, 0.01)
    dc = np.append(dc, h)
    εc=εlim(dc, x, h, ε0, εu, k, ds)
    εs=εlim(ds, x, h, ε0, εu, k, ds)
    σc=σconcrete(εc,ε0,εu,fcd,0.85)
    σs=σsteel(εs,fyd,Es)
    Rcc=integrate.trapezoid(σc*b, dc)
    Mc=integrate.trapezoid(σc*dc*b, dc)
    EQ=(N-Rcc)*np.sum(n*σs*ds)+(M-N*(h/2)+Mc)*np.sum(n*σs)
    return EQ

def EquiDasForcas(b,h,dc,ds,As,fcd,fyd,Es,ε0,εu,N,εm,χ,coef):
    εc=ε(χ,dc,εm)
    εs=ε(χ,ds,εm)
    σc=σconcrete(εc,ε0,εu,fcd,coef)
    σs=σsteel(εs,fyd,Es)
    Rcc=integrate.trapezoid(σc*b, dc)
    Rs=np.sum(As*σs)
    EQ=N-Rcc-Rs
    return EQ

def MomentoEquilibrante(b,h,dc,ds,As,fcd,fyd,Es,ε0,εu,N,εm,χ,coef):
    εc=ε(χ,dc,εm)
    εs=ε(χ,ds,εm)
    σc=σconcrete(εc,ε0,εu,fcd,coef)
    σs=σsteel(εs,fyd,Es)
    Mc=integrate.trapezoid(σc*dc*b, dc)
    Ms=np.sum(As*σs*ds)
    return Mc+Ms

def momento_fissuracao(fck, b, h, D, x_raiz, alpha=1.5):
    fct_m = 0.3 * (fck / 1000) ** (2/3) * 1000
    fct = 0.7 * fct_m
    if D != 0:
        Ic = (np.pi * (D/2)**4)/4
        yt = D-x_raiz
    else:
        Ic = (b * h**3)/12
        yt = h-x_raiz
    return alpha * fct * (Ic / yt)

# ==========================================
# 3. FUNÇÕES DE PÓS-PROCESSAMENTO (GRÁFICOS P/ STREAMLIT)
# ==========================================
def plot_Momento_curvatura(data, language='pt', xlabel=None, ylabel=None):
    fig = plt.figure(figsize=(9, 6))
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.alpha'] = 0.7

    if not data:
        st.warning("Nenhum dado fornecido.")
        return

    for label, (x, y) in data.items():
        x = np.array(x)
        y = np.array(y)
        color = 'red' if 'Código Implementado' in label or 'Implemented Code' in label else 'black'
        
        if '0.85 fcd' in label: linestyle = '-'
        elif '1.1 fcd' in label: linestyle = '--'
        else: linestyle = '-'
        
        plt.plot(x, y, linestyle=linestyle, color=color, label=label, linewidth=2)

    xlabel = 'Curvatura ($\chi$) [1/m]' if language == 'pt' else 'Curvature ($\chi$) [1/m]'
    ylabel = 'Momento Fletor [kN.m]' if language == 'pt' else 'Bending Moment [kN.m]'
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def plot_Momento_vs_Rigidez(data, language='pt'):
    fig = plt.figure(figsize=(9, 6))
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.alpha'] = 0.7

    colors = ['r', 'b', 'g', 'm', 'c', 'y', 'k']
    for i, (label, (x, y)) in enumerate(data.items()):
        plt.plot(x, y, linestyle=['-', '--', '-.', ':'][i % 4], color=colors[i % len(colors)], label=label)

    xlabel = 'Momento Fletor [kN.m]' if language == 'pt' else 'Bending Moment [kN.m]'
    ylabel = 'Rigidez (EI) [kN.m²]' if language == 'pt' else 'Flexural Stiffness (EI) [kN.m²]'
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='best', frameon=True)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def plot_deslocamentos(data, L, language='pt'):
    fig = plt.figure(figsize=(9, 6))
    plt.rcParams.update({'font.family': 'serif', 'font.size': 12, 'axes.grid': True, 'grid.linestyle': '--', 'grid.alpha': 0.7})
    
    line_styles = ['-', '--', '-.', ':']
    plt.plot([0, 0], [0, L], color='black', linewidth=1.5, zorder=1)

    for i, series in enumerate(data):
        n_points = len(series['Z'])
        y = np.linspace(0, L, n_points)
        label = series['label']
        color = 'red' if 'Código Implementado' in label or 'Implemented Code' in label else 'black'
        linestyle = line_styles[i % len(line_styles)]
        plt.plot(series['Z'], y, linestyle=linestyle, color=color, label=label, linewidth=1.5, zorder=2)

    xlabel = 'Deslocamento [m]' if language == 'pt' else 'Displacement [m]'
    ylabel = 'Posição ao longo do pilar [m]' if language == 'pt' else 'Position along the column [m]'
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='best', frameon=True)
    plt.ylim(0, L)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def plot_momentos_fletores(data, L, language='pt'):
    fig = plt.figure(figsize=(9, 6))
    plt.rcParams.update({'font.family': 'serif', 'font.size': 12, 'axes.grid': True, 'grid.linestyle': '--', 'grid.alpha': 0.7})
    
    line_styles = ['-', '--', '-.', ':']
    plt.plot([0, 0], [0, L], color='black', linewidth=1.5, linestyle='-', label='_nolegend_', zorder=1)

    for i, series in enumerate(data):
        n_points = len(series['M'])
        y = np.linspace(0, L, n_points)
        label = series['label']
        color = 'red' if 'Código Implementado' in label or 'Implemented Code' in label else 'black'
        linestyle = line_styles[i % len(line_styles)]
        plt.plot(series['M'], y, linestyle=linestyle, color=color, label=label, linewidth=1.5, zorder=2)

    xlabel = 'Momento Fletor [kN.m]' if language == 'pt' else 'Bending Moment [kN.m]'
    ylabel = 'Posição ao longo do pilar [m]' if language == 'pt' else 'Position along the column [m]'
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='best', frameon=True)
    plt.ylim(0, L)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def tabela_deslocamentos_st(datatable):
    records = []
    for item in datatable:
        Z = item['Z'].flatten()
        records.append({'Curvas': item['label'], 'Desl. Máx. (m)': max(Z), 'Desl. Mín. (m)': min(Z)})
    return pd.DataFrame(records)

def tabela_momentos_fletores_st(datatable):
    records = []
    for item in datatable:
        Z = item['M'].flatten()
        records.append({'Curvas': item['label'], 'MF. Máx. (kN.m)': max(Z), 'MF Mín. (kN.m)': min(Z)})
    return pd.DataFrame(records)

def tabela_fs(arr_momentos, m_resistente):
    fs_numericos = []
    for m in arr_momentos:
        if abs(m) < 0.001:
            fs_numericos.append(float('inf'))
        else:
            fs = abs(m_resistente / m)
            fs_numericos.append(fs)

    min_fs = min(fs_numericos)
    fs_display = []
    for val in fs_numericos:
        if val == float('inf'):
            fs_display.append("(OK!)")
        else:
            val_fmt = round(val, 2)
            if val == min_fs: fs_display.append(f"{val_fmt} ⚠️")
            else: fs_display.append(val_fmt)

    num_pontos = len(arr_momentos)
    labels_z = []
    for i in range(num_pontos):
        ratio = i / (num_pontos - 1) if num_pontos > 1 else 0.0
        if ratio == 0.0: labels_z.append("0 (Base)")
        elif ratio >= 0.999: labels_z.append("L (Topo)")
        else: labels_z.append(f"{ratio:.2f} L")

    df = pd.DataFrame({'z': labels_z, 'Msd': np.round(arr_momentos, 2), 'F.S.': fs_display})
    return df.iloc[::-1].reset_index(drop=True)

def plot_fator_seguranca_final(arr_momentos, m_resistente, L, limite_escala=35.0):
    z_vals = np.linspace(0, L, len(arr_momentos))
    with np.errstate(divide='ignore', invalid='ignore'):
        fs_vals = np.abs(m_resistente / arr_momentos)
    fs_plot = np.clip(fs_vals, 0, limite_escala)
    min_fs_idx = np.argmin(fs_vals)
    min_fs = fs_vals[min_fs_idx]
    z_critico = z_vals[min_fs_idx]

    fig, ax = plt.subplots(figsize=(9, 6))
    plt.rcParams.update({'font.family': 'serif', 'font.size': 12, 'axes.grid': True, 'grid.linestyle': '--'})
    
    ax.axvspan(0, 1.0, color='#ffe6e6', alpha=1.0, zorder=0)
    ax.axvline(x=1.0, color='#d62728', linestyle='--', linewidth=1.5, zorder=1, label='Limite ($M_{sd} = M_{rd}$)')
    ax.text(1.15, L*0.96, '$M_{sd} = M_{rd}$', color='#d62728', fontsize=10, ha='left', va='top', fontweight='bold')
    ax.plot(fs_plot, z_vals, color='#003366', linestyle='-', linewidth=1.5, zorder=3, label='FSI Calculado')
    ax.fill_betweenx(z_vals, 0, fs_plot, color='#003366', alpha=0.1, zorder=2)

    is_inf = (fs_vals >= limite_escala)
    if np.any(is_inf):
        changes = np.diff(np.concatenate(([0], is_inf.astype(int), [0])))
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]
        for start, end in zip(starts, ends):
            z_inf = z_vals[(start + end) // 2]
            ax.annotate(r'$\infty$', xy=(limite_escala, z_inf + 0.1), xytext=(limite_escala - 1.5, z_inf + 0.1),
                        arrowprops=dict(arrowstyle='->', color='#003366', linewidth=1.5),
                        fontsize=16, color='#003366', va='center', ha='right', zorder=5)

    ax.scatter([min_fs], [z_critico], color='white', edgecolors='red', s=60, linewidth=1.5, zorder=4)
    ax.annotate(f'{min_fs:.2f}', xy=(min_fs, z_critico), xytext=(min_fs + 2.0, z_critico),
                 arrowprops=dict(arrowstyle='-|>', color='black', shrinkA=5, shrinkB=5, linewidth=1.0),
                 fontsize=12, fontweight='bold', color='red', va='center', ha='left',
                 bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.7))

    ax.set_xlabel('Fator de Segurança de Impacto (FSI)')
    ax.set_ylabel('Posição ao longo do pilar [m]')
    ax.set_xlim(0, limite_escala)
    ax.set_ylim(0, L)
    
    ticks = np.unique(np.concatenate(([0, 1], np.arange(5, limite_escala + 1, 5))))
    ax.set_xticks(ticks)
    labels = [str(int(t)) for t in ticks]
    labels[-1] = r'$\geq$' + str(int(limite_escala))
    ax.set_xticklabels(labels)
    
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color('black')

    ax.legend(loc='best', frameon=True)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ==========================================
# 4. INTERFACE E EXECUÇÃO
# ==========================================
st.sidebar.header("📊 Dados de Entrada")
tipo_secao = st.sidebar.selectbox("Tipo de Seção", ["Retangular", "Circular"])
cond_contorno = st.sidebar.selectbox("Condição de Contorno", ["Biapoiado", "Engastado (Em balanço)"])

st.sidebar.subheader("Geometria")
if tipo_secao == "Retangular":
    b = st.sidebar.number_input("Largura b (m)", value=0.3)
    h = st.sidebar.number_input("Altura h (m)", value=0.3)
    D = 0
else:
    D = st.sidebar.number_input("Diâmetro D (m)", value=0.4)
    b, h = 0, D
L = st.sidebar.number_input("Comprimento L (m)", value=3.4)

st.sidebar.subheader("Armadura e Materiais")
bitola = st.sidebar.number_input("Bitola (mm)", value=20.0 if tipo_secao=="Retangular" else 25.0)
dLinha = st.sidebar.number_input("d' (m)", value=0.0413 if tipo_secao=="Retangular" else 0.0475)
fck = st.sidebar.number_input("fck (kN/m²)", value=30000)
fyk = st.sidebar.number_input("fyk (kN/m²)", value=500000)
Es = st.sidebar.number_input("Es (kN/m²)", value=210000000)

fcd = fck / 1.2
fyd = fyk / 1.0
E = 2.454e7
ε0 = 0.002
εu = 0.0035
k = 1 - (ε0 / εu)

st.sidebar.subheader("Carregamentos")
Nd = st.sidebar.number_input("Nd (kN)", value=1127.9 if tipo_secao=="Retangular" else 300.0)
qd = st.sidebar.number_input("qd (kN/m)", value=-5.32 if tipo_secao=="Retangular" else 0.0)
if cond_contorno == "Biapoiado":
    MBd = st.sidebar.number_input("MBd (kN.m) - Base", value=-10.8)
else:
    MBd = 0.0
MTd = st.sidebar.number_input("MTd (kN.m) - Topo", value=10.0 if tipo_secao=="Retangular" else 0.0)
if cond_contorno == "Engastado (Em balanço)":
    QTd = st.sidebar.number_input("QTd (kN) - Topo", value=30.0)
else:
    QTd = 0.0

Fveic = st.sidebar.number_input("Força do impacto do veículo Fveic (kN)", value=50.0 if tipo_secao=="Retangular" else 0.0)
Pos_Fveic = st.sidebar.number_input("Posição do impacto Pos_Fveic (m)", value=0.5 if Fveic > 0 else 0.0)
if Fveic == 0: Pos_Fveic = None

DeltaZ = 0.05
nNos = int(L/DeltaZ + 1)
γf3 = 1.1

if st.sidebar.button("🚀 Executar Análise"):
    with st.spinner("Realizando análise das seções transversais e esforços não lineares..."):
        
        # --- PREPARAÇÃO DA ARMADURA ---
        if tipo_secao == "Retangular":
            n = np.array([2,2])
            nLinha = len(n)
            ds = np.linspace(-h/2 + dLinha, h/2 - dLinha, nLinha)
        else:
            R_efet = (D/2) - (dLinha)
            nLinha = 8 
            Δθ = (2*np.pi)/nLinha
            θ = np.pi/ 2 + np.arange(nLinha)*Δθ
            ds = R_efet * np.sin(θ)
            ds = np.unique(np.round(ds, 6))
            n = np.ones_like(ds)
            n[1:-1] = 2

        bitolas_array = np.array([10.0, 12.5, 16.0, 20.0, 25.0, 32.0])
        areas = np.array([0.0000785, 0.000122, 0.000201, 0.000314, 0.000491, 0.000804])
        
        try:
            As = n * areas[bitolas_array == bitola][0]
        except IndexError:
            st.error("Bitola não encontrada. Selecione 10, 12.5, 16, 20, 25 ou 32 mm.")
            st.stop()

        # --- ANÁLISE LG E NLG INICIAL ---
        if cond_contorno == "Biapoiado":
            r_val = 0 if tipo_secao == "Retangular" else D/2
            E_val = 0 if tipo_secao == "Retangular" else fck
            EIef = RigidezFlexaoConstante(b, h, r_val, E_val, 0, 1) * np.ones(nNos+2)
            
            W_P0 = DeslocamentoLateralBIAPOIADOS(nNos, DeltaZ, 0, qd, MBd, MTd, EIef, Pos_Fveic, Fveic)
            M_P0 = MomentoFletorBIAPOIADO(nNos, DeltaZ, EIef, MBd, MTd, W_P0)
            
            W_NLG = DeslocamentoLateralBIAPOIADOS(nNos, DeltaZ, Nd, qd, MBd, MTd, EIef, Pos_Fveic, Fveic)
            M_NLG = MomentoFletorBIAPOIADO(nNos, DeltaZ, EIef, MBd, MTd, W_NLG)
        else: # Engastado
            r_val = 0 if tipo_secao == "Retangular" else D/2
            E_val = 0 if tipo_secao == "Retangular" else E
            EIef = RigidezFlexaoConstante(b, h, r_val, 0, E_val, 1) * np.ones(nNos+1)
            
            W_P0 = DeslocamentoLateralENGASTE(nNos, DeltaZ, 0, qd, QTd, MTd, EIef, Pos_Fveic, Fveic)
            M_P0 = MomentoFletorENGASTE(nNos, DeltaZ, EIef, MTd, W_P0)
            
            W_NLG = DeslocamentoLateralENGASTE(nNos, DeltaZ, Nd, qd, QTd, MTd, EIef, Pos_Fveic, Fveic)
            M_NLG = MomentoFletorENGASTE(nNos, DeltaZ, EIef, MTd, W_NLG)

        # --- DIMENSIONAMENTO (x_raiz) ---
        Md = max(np.absolute(M_NLG))
        x_raiz = fsolve(lambda x: EquiDasForcasEMomentos(x, b, h, ds, n, fcd, fyd, Es, ε0, εu, k, Nd, Md), h/2)
        
        # --- DIAGRAMA N, M, 1/r ---
        dc = np.arange(-h/2, h/2, 0.01) if tipo_secao == "Retangular" else np.arange(-D/2, D/2, 0.01)
        dc = np.append(dc, h/2) if tipo_secao == "Retangular" else np.append(dc, D/2)

        def gerar_curva_NM(coef, limit_M=None):
            χ_arr, M_arr = [], []
            εm_chute, χ, inc_χ = 0.001, 0, 0.00001
            while True:
                εm_r = fsolve(lambda εm: EquiDasForcas(b,h,dc,ds,As,fcd,fyd,Es,ε0,εu,Nd/coef if coef==1.1 else Nd,εm,χ,coef), εm_chute)
                M_val = MomentoEquilibrante(b,h,dc,ds,As,fcd,fyd,Es,ε0,εu,Nd/coef if coef==1.1 else Nd,εm_r,χ,coef)
                deforms = ε(χ, dc, εm_r[0])
                if limit_M and M_val > limit_M: break
                if not limit_M and (np.any(deforms > 0.0035) or np.any(deforms < -0.01)): break
                
                χ_arr.append(χ)
                M_arr.append(M_val)
                εm_chute = εm_r[0]
                χ += inc_χ
            return np.array(χ_arr), np.array(M_arr)

        χ_values_85, M_values_85 = gerar_curva_NM(0.85)
        Mdbreak = M_values_85[-1] / 1.1 if len(M_values_85) > 0 else 0
        χ_values_11, M_values_11 = gerar_curva_NM(1.1, limit_M=Mdbreak)

        # --- NLF ---
        Mr_val = momento_fissuracao(fck, b if tipo_secao == "Retangular" else 0, h if tipo_secao == "Retangular" else 0, D if tipo_secao == "Circular" else 0, x_raiz)

        if cond_contorno == "Biapoiado":
            W_prev = DeslocamentoLateralBIAPOIADOS(nNos, DeltaZ, Nd/γf3, qd/γf3, MBd/γf3, MTd/γf3, EIef, Pos_Fveic, Fveic)
            M_prev = MomentoFletorBIAPOIADO(nNos, DeltaZ, EIef, MBd/γf3, MTd/γf3, W_prev)
            pad_mode = (1, 1)
        else:
            W_prev = DeslocamentoLateralENGASTE(nNos, DeltaZ, Nd/γf3, qd/γf3, QTd/γf3, MTd/γf3, EIef, Pos_Fveic, Fveic)
            M_prev = MomentoFletorENGASTE(nNos, DeltaZ, EIef, MTd/γf3, W_prev)
            pad_mode = (0, 1)
            
        EIef_NL = rigidezsecante(χ_values_11, M_values_11, np.abs(M_prev), Mr_val.item() if isinstance(Mr_val, np.ndarray) else Mr_val)
        EIef_NL_corrigido = np.pad(EIef_NL, pad_mode, mode='edge')
        EIef_NL_final = np.copy(EIef_NL)

        tol = 1e-4
        max_iter = 100
        iter_count = 0

        while iter_count < max_iter:
            iter_count += 1
            if cond_contorno == "Biapoiado":
                W_NLG_NLF = DeslocamentoLateralBIAPOIADOS(nNos,DeltaZ,Nd/γf3,qd/γf3,MBd/γf3,MTd/γf3,EIef_NL_corrigido,Pos_Fveic,Fveic)
                M_NLG_NLF = MomentoFletorBIAPOIADO(nNos, DeltaZ, EIef_NL_final,MBd/γf3, MTd/γf3, W_NLG_NLF)
            else:
                W_NLG_NLF = DeslocamentoLateralENGASTE(nNos,DeltaZ,Nd/γf3,qd/γf3,QTd/γf3,MTd/γf3,EIef_NL_corrigido,Pos_Fveic,Fveic)
                M_NLG_NLF = MomentoFletorENGASTE(nNos, DeltaZ, EIef_NL_final, MTd/γf3, W_NLG_NLF)
                
            EIef_NL_final = rigidezsecante(χ_values_11, M_values_11, np.absolute(M_NLG_NLF), Mr_val.item() if isinstance(Mr_val, np.ndarray) else Mr_val)
            diff_W = np.linalg.norm(W_NLG_NLF - W_prev) / np.linalg.norm(W_NLG_NLF)
            
            if diff_W < tol:
                break
            
            W_prev = np.copy(W_NLG_NLF)
            EIef_NL_corrigido = np.pad(EIef_NL_final, pad_mode, mode='edge')

        st.success(f"✅ Análise iterativa concluída com sucesso (Iterações: {iter_count}).")

        # ==========================================
        # IMPRIMINDO RESULTADOS (UI STREAMLIT)
        # ==========================================
        st.header("📈 Resultados da Análise")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Diagrama N, M, 1/r", "M x EIef", "Deslocamentos", "Momentos Fletores", "Fator de Segurança (FSI)"])
        
        with tab1:
            st.subheader("Diagrama N, M, 1/r")
            data_curv = {
                '0.85 fcd - Código Implementado': (χ_values_85, M_values_85),
                '1.1 fcd - Código Implementado': (χ_values_11, M_values_11)
            }
            plot_Momento_curvatura(data_curv)

        with tab2:
            st.subheader("Diagrama M x EIef")
            EIef_plt = EIef[0]*np.ones_like(M_NLG)
            data_rigid = {'Linear': (M_P0, EIef_plt), 'Não Linear': (M_NLG_NLF, EIef_NL_final)}
            plot_Momento_vs_Rigidez(data_rigid)

        with tab3:
            st.subheader("Deslocamento ao longo do pilar")
            data_desl = [{'Z': W_P0, 'label': "LG & LF - Código Implementado"},
                         {'Z': W_NLG, 'label': "NLG & LF - Código Implementado"},
                         {'Z': W_NLG_NLF, 'label': "NLG & NLF - Código Implementado"}]
            
            st.dataframe(tabela_deslocamentos_st(data_desl), use_container_width=True)
            plot_deslocamentos(data_desl, L)

        with tab4:
            st.subheader("Momento Fletor ao longo do pilar")
            data_mf = [{'M': M_P0, 'label': "LG & LF - Código Implementado"},
                       {'M': M_NLG, 'label': "NLG & LF - Código Implementado"},
                       {'M': γf3*M_NLG_NLF, 'label': "NLG & NLF - Código Implementado"}]
                       
            st.dataframe(tabela_momentos_fletores_st(data_mf), use_container_width=True)
            plot_momentos_fletores(data_mf, L)

        with tab5:
            st.subheader("Fator de Segurança de Impacto (FSI)")
            m_res_max = M_values_85[-1] if len(M_values_85) > 0 else 0
            st.info(f"**Momento Fletor Resistente ($M_{{rd}}$):** {m_res_max:.2f} kN.m")
            
            df_fsi = tabela_fs(M_NLG_NLF*γf3, m_res_max)
            colA, colB = st.columns([1, 2])
            with colA:
                st.dataframe(df_fsi, use_container_width=True)
            with colB:
                plot_fator_seguranca_final(M_NLG_NLF*γf3, m_res_max, L)
