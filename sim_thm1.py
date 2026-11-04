#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Theorem 1"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import solve_ivp
from scipy.special import ndtr
from scipy.stats import qmc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RESULTS, FIGURES, TABLES = (ROOT / x for x in ('results', 'figures', 'tables'))
for directory in (RESULTS, FIGURES, TABLES):
    directory.mkdir(parents=True, exist_ok=True)
COLORS = ('#1767a1', '#dc792d', '#328375')
MODELS = ('G1', 'G2')
DIMENSIONS = ((50, 10), (50, 50), (50, 100), (50, 200), (50, 300),
              (100, 10), (100, 50), (100, 100), (100, 200), (100, 300),
              (200, 10), (200, 50), (200, 100), (200, 200), (200, 300))
# # Terminal-time sensitivity
# HORIZONS = (5., 10., 20., 40., 50., 80., 100.)
HORIZONS = np.arange(5, 101, dtype=float)
# vary p,d
HORIZON_DIMENSIONS = ((50, 10), (50, 50), (50, 100), (50, 200), (50, 300),
              (100, 10), (100, 50), (100, 100), (100, 200), (100, 300),
              (200, 10), (200, 50), (200, 100), (200, 200), (200, 300))
T0, DELTA, GAMMA_FIXED, T_FIXED = .05, .5, .1, 10.
BIASES = np.linspace(0., .3, 201)
LINE_WIDTH = 1.5
MARKER_SIZE = 3.0
MARK_EVERY = 10


CSV_COLUMN_NAMES = {
    'delta': 'delta_m',
    'bias': 'kappa_b',
    'W2': 'W_act',
    'stop': 'eta_m',
    'terminal_D': 'D_T',
    'init': 'A_D_T',
    'score': 'Score_term',
    'score_loss': 'Score_loss_term',
    'direct': 'B_dir',
    'loss': 'B_loss',
    'SM_loss': 'L_SM',
}


def quadrature(n, lower=0., upper=1.):
    """Gauss-Legendre  [-1,1] """
    nodes, weights = leggauss(n)
    return (lower + (upper - lower) * (nodes + 1) / 2,
            weights * (upper - lower) / 2)


def variance(model, x):
    if model == 'G1':
        return np.full(x.shape[0], .5)
    p = x.shape[1]
    z = np.sqrt(12. / p) * np.sum(x - .5, axis=1)
    return .25 + .5 * ndtr(z)


def variance_max(model, p):
    return .5 if model == 'G1' else .75



# def conditional_mean(x, d):
#     p = x.shape[1]
#     coordinate = np.arange(d) % p
#     return np.sin(2.*np.pi*x[:, coordinate])
def conditional_mean(x, d):
    base = np.sin(2.0 * np.pi * x.mean(axis=1))
    return np.repeat(base[:, None], d, axis=1)

def sample_conditional(x, model, d, rng):
    mu = conditional_mean(x,d)
    scales = np.sqrt(variance(model,x))[:,None]
    return mu+scales*rng.standard_normal(mu.shape)


def test_sampled_moments():
    rng=np.random.default_rng(13579)
    x=np.full((20000,50),.3)
    y=sample_conditional(x,'G2',10,rng)
    mu=conditional_mean(x[:1],10)[0]
    variance_true=variance('G2',x[:1])[0]
    mean_error=float(np.max(np.abs(y.mean(axis=0)-mu)))
    variance_error=float(np.max(np.abs(y.var(axis=0)-variance_true)))
    assert mean_error<.035 and variance_error<.035
    return dict(max_sample_mean_error=mean_error,
                max_sample_variance_error=variance_error)


def phi(t0, t, v0):
    return np.exp(-(np.asarray(t)-t0)/2) * (
        -np.expm1(-t0) + v0 * np.exp(-t0)) / (
        -np.expm1(-np.asarray(t)) + v0 * np.exp(-np.asarray(t)))


def jint(t0, T, v0):
    v0 = np.asarray(v0)
    q = np.sqrt(1.-v0)
    v_t0 = 1.+(v0-1.)*np.exp(-t0)
    return (2.*v_t0/np.exp(-t0/2)/q *
            (np.arctanh(q*np.exp(-t0/2)) -
             np.arctanh(q*np.exp(-T/2))))


def sample_variances(p, model, power, repeats, seed):
    """Sobol """
    if model == 'G1':
        yield np.array([.5])
        return
    for rep in range(repeats):
        sampler = qmc.Sobol(d=p, scramble=True, seed=seed + 1019*rep + p)
        x = sampler.random_base2(m=power)
        yield variance(model, x)


def moments_at_time(v0, T, t0):
    at0, aT = np.exp(-t0/2), np.exp(-T/2)
    vt0 = 1.+(v0-1.)*np.exp(-t0)
    vT = 1.+(v0-1.)*np.exp(-T)
    P = phi(t0, T, v0)
    J = jint(t0, T, v0)
    a = 1.-at0+aT*P
    output_v = vt0+P*P*(1.-vT)
    shape = (np.sqrt(output_v)-np.sqrt(v0))**2
    terminal_shape = ((v0-1.)*np.exp(-T)/(np.sqrt(vT)+1.))**2
    return (np.mean(a*a), np.mean(a*J), np.mean(J*J),
            np.mean(shape), np.mean(terminal_shape), np.mean(v0))


def summarize_moments(p, model, T, power, repeats, seed, t0):
    rows = [moments_at_time(v0,T,t0) for v0 in
            sample_variances(p,model,power,repeats,seed)]
    return np.mean(rows, axis=0), np.asarray(rows)


def build_case(p, d, model, T, bias, moment_vec, ntime=192, t0=T0, delta=DELTA):
    aa, aJ, JJ, shape, terminal_shape, mean_v0 = moment_vec
    w2sq = delta**2*aa+2*delta*bias*aJ+bias**2*JJ+d*shape
    W2 = np.sqrt(max(0.,w2sq))
    D_T = np.sqrt(np.exp(-T)*delta**2+d*terminal_shape)
    stop = np.sqrt((1.-np.exp(-t0/2))**2*(delta**2+d*mean_v0)
                   +d*(-np.expm1(-t0)))
    vmax = variance_max(model,p)
    prop_init = phi(t0,T,vmax)*D_T
    direct_score = bias*jint(t0,T,np.asarray(vmax))
    time_nodes, time_weights = quadrature(ntime,t0,T)
    c2 = np.dot(time_weights,phi(t0,time_nodes,vmax)**2)
    loss_score = np.sqrt(c2*(T-t0))*bias
    B_dir = stop+prop_init+direct_score
    B_loss = stop+prop_init+loss_score
    return dict(model=model,p=p,d=d,T=T,t0=t0,delta=delta,bias=bias,
                W2=float(W2),stop=float(stop),terminal_D=float(D_T),
                init=float(prop_init),score=float(direct_score),
                score_loss=float(loss_score),direct=float(B_dir),
                loss=float(B_loss),SM_loss=(T-t0)*bias**2/2)


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=[CSV_COLUMN_NAMES.get(k,k) for k in rows[0]])
        writer.writeheader()
        writer.writerows({CSV_COLUMN_NAMES.get(k,k):v for k,v in row.items()} for row in rows)


def style_axes(ax, xlabel, ylabel, title):
    ax.set(xlabel=xlabel,ylabel=ylabel,title=title)
    ax.grid(alpha=.18,linewidth=.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=10)


def save_chart(fig, name):
    """ 6.4 x 4.2 、350 dpi。"""
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.set_size_inches(6.4, 4.2, forward=True)
    fig.set_facecolor('white')
    for ax in fig.axes:
        ax.set_facecolor('white')
        ax.grid(False, which='both', axis='both')
    fig.tight_layout()

    with plt.rc_context({
        'pdf.fonttype': 42,
        'pdf.use14corefonts': False,
    }):
        fig.savefig(
            FIGURES / (name + '.pdf'),
            facecolor='white',
            transparent=False,
        )
    fig.savefig(
        FIGURES / (name + '.jpg'),
        dpi=1000,
        facecolor='white',
        transparent=False,
        pil_kwargs={'quality': 95},
    )
    plt.close(fig)


def chart_curves(model,p,d,curves):
    fig,ax=plt.subplots(figsize=(6.4,4.2))
    ax.plot(BIASES,[r['W2'] for r in curves],color=COLORS[0],lw=LINE_WIDTH,
            label=r'$\widehat{W}_{\mathrm{act}}$')
    ax.plot(BIASES,[r['direct'] for r in curves],color=COLORS[1],ls='--',lw=LINE_WIDTH,
            label=r'$\widehat{B}_{\mathrm{dir}}$')
    ax.plot(BIASES,[r['loss'] for r in curves],color=COLORS[2],ls='-.',lw=LINE_WIDTH,
            label=r'$\widehat{B}_{\mathrm{loss}}$')
    style_axes(ax,r'$\kappa_b$','Values',f'{model}: p={p}, d={d}, T={T_FIXED:g}')
    ax.set(xlim=(0,.3),ylim=(0,None))
    ax.set_xticks([0,.1,.2,.3])
    ax.legend(loc='upper left', ncol=1, frameon=False, fontsize=9)
    fig.subplots_adjust(left=.14, right=.97,bottom=.16, top=.88)
    save_chart(fig,f'bound_{model}_p{p}_d{d}')


def chart_horizon(model,p,d,horizon):
    T=[r['T'] for r in horizon]
    fig,ax=plt.subplots(figsize=(6.4,4.2))
    for column,color,label,ls in [
        ('W2',COLORS[0],r'$\widehat{W}_{\mathrm{act}}$','-'),
        ('direct',COLORS[1],r'$\widehat{B}_{\mathrm{dir}}$','--'),
        ('loss',COLORS[2],r'$\widehat{B}_{\mathrm{loss}}$','-.')]:
        ax.plot(T,[r[column] for r in horizon],color=color,label=label,ls=ls,
                lw=LINE_WIDTH,marker='o',markersize=MARKER_SIZE,markevery=MARK_EVERY)
    style_axes(ax,'$T$','Values',
               f'{model}: p={p}, d={d}')#, 'rf'$\kappa_b={GAMMA_FIXED:g}$
    ax.set_xticks([5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ax.set_xlim(5, 100)
    ax.legend(loc='upper left', ncol=1,frameon=False, fontsize=9)
    fig.subplots_adjust(left=.14, right=.97,bottom=.16, top=.88)
    save_chart(fig,f'horizon_bounds_{model}_p{p}_d{d}')


def chart_initialization(model,p,d,horizon):
    T=[r['T'] for r in horizon]
    fig,ax=plt.subplots(figsize=(6.4,4.2))
    ax.semilogy(T,[max(r['terminal_D'],1e-120) for r in horizon],
                'o-',color=COLORS[0],lw=LINE_WIDTH,markersize=MARKER_SIZE,markevery=MARK_EVERY,label=r'$\widehat{D}_T(m_\delta)$') #'Terminal mismatch'
    ax.semilogy(T,[max(r['init'],1e-120) for r in horizon],
                's--',color=COLORS[1],lw=LINE_WIDTH,markersize=MARKER_SIZE,markevery=MARK_EVERY,label=r'$A(t_0,T)\widehat{D}_T(m_\delta)$') # 'Propagated mismatch'
    style_axes(ax,'$T$','Initialization error',
               f'{model}: p={p}, d={d}') #, 'rf'$\delta_m={DELTA:g}$
    ax.set_xticks([5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ax.set_xlim(5, 100)
    ax.legend(loc='upper right', ncol=1,frameon=False, fontsize=9)
    fig.subplots_adjust(left=.14, right=.97,bottom=.16, top=.88)
    save_chart(fig,f'horizon_initialization_{model}_p{p}_d{d}')


def latex_table(model, rows, filepath):
    tex=[r'\begin{table}[t]',r'\centering\small',
         rf'\caption{{High-dimensional sampling diagnostics for {model} at $T=10$, $t_0=0.05$, center bias $0.5$, and score bias $0.1$.}}',
         rf'\label{{tab:highdim_{model.lower()}}}',
         r'\setlength{\tabcolsep}{3pt}',
         r'\begin{tabular}{@{}rrrrrrrr@{}}',r'\toprule',
         r'$p$ & $d$ & Actual $W_2$ & Stop & Init. & Score & Direct bound & Loss bound\\',
         r'\midrule']
    for r in rows:
        tex.append(f"{r['p']} & {r['d']} & {r['W2']:.4f} & {r['stop']:.4f} & "
                   f"{r['init']:.3g} & {r['score']:.4f} & "
                   f"{r['direct']:.4f} & {r['loss']:.4f}"+r'\\')
    tex.extend([r'\bottomrule',r'\end{tabular}',r'\end{table}'])
    filepath.write_text('\n'.join(tex)+'\n',encoding='utf-8')


def test_analytic_formula():
    errors=[]
    for vv in (.3,.5,.7):
        for T in (5.,10.,50.):
            t0=T0
            def rhs(u,y):
                t=T-u
                vt=1.+(vv-1.)*np.exp(-t)
                c=.5-1./vt
                b=-np.exp(-t/2)*DELTA/vt+GAMMA_FIXED
                return (c*y[0]+b,2*c*y[1]+1.)
            sol=solve_ivp(rhs,(0,T-t0),(0.,1.),rtol=1e-11,atol=1e-13)
            P=phi(t0,T,vv)
            mean=-np.exp(-t0/2)*DELTA+P*np.exp(-T/2)*DELTA+GAMMA_FIXED*jint(t0,T,vv)
            var=1.+(vv-1.)*np.exp(-t0)+P*P*(1.-(1.+(vv-1.)*np.exp(-T)))
            errors.append(float(np.max(np.abs(sol.y[:,-1]-[mean,var]))))
    return max(errors)


def main(args):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
    moment_cache={}
    for p in sorted({pp for pp, _ in DIMENSIONS} | {pp for pp, _ in HORIZON_DIMENSIONS}):
        for model in MODELS:
            for T in HORIZONS:
                moment_cache[p,model,T]=summarize_moments(
                    p,model,T,args.power,args.repeats,args.seed,T0)[0]
    all_cases=[]
    qmc_precision=[]
    for model in MODELS:
        table_rows=[]
        for p,d in DIMENSIONS:
            moment_vec=moment_cache[p,model,T_FIXED]
            curves=[build_case(p,d,model,T_FIXED,float(b),moment_vec)
                    for b in BIASES]
            all_cases.extend(curves)
            chart_curves(model,p,d,curves)
            selected=build_case(p,d,model,T_FIXED,GAMMA_FIXED,moment_vec)
            table_rows.append(selected)
            #####
            _, sample_moments = summarize_moments(
                p,model,T_FIXED,args.power,args.repeats,args.seed,T0)
            replicated = [build_case(p,d,model,T_FIXED,GAMMA_FIXED,v)
                          for v in sample_moments]
            for metric in ('W2','direct','loss'):
                values = np.array([r[metric] for r in replicated])
                se = float(np.std(values,ddof=1)/np.sqrt(len(values))) if len(values)>1 else 0.
                qmc_precision.append(dict(model=model,p=p,d=d,metric=metric,
                                          qmc_mean=float(np.mean(values)),qmc_se=se,
                                          scrambles=len(values)))
        write_csv(RESULTS/f'table_{model}.csv',table_rows)
        latex_table(model,table_rows,TABLES/f'table_{model}.tex')
        for p,d in HORIZON_DIMENSIONS:
            horizon=[build_case(p,d,model,T,GAMMA_FIXED,moment_cache[p,model,T])
                     for T in HORIZONS]
            write_csv(RESULTS/f'horizon_{model}_p{p}_d{d}.csv',horizon)
            if (p,d)==(50,50):
                write_csv(RESULTS/f'horizon_{model}.csv',horizon)
            chart_horizon(model,p,d,horizon)
            chart_initialization(model,p,d,horizon)
    write_csv(RESULTS/'all_score_bias_curves.csv',all_cases)
    write_csv(RESULTS/'qmc_precision.csv',qmc_precision)
    comparisons=[r for r in all_cases if r['W2']>r['direct']+1e-9
                 or r['direct']>r['loss']+1e-9]
    ode_error=test_analytic_formula()
    sampled_test=test_sampled_moments()
    assert not comparisons, 'Found a violated computed bound'
    assert ode_error<1e-8, f'ODE cross-check failed: {ode_error}'
    # 单独用两种 QMC 点数，检查高维变方差积分的数值稳定性。
    qa=summarize_moments(100,'G2',10.,args.power,args.repeats,args.seed,T0)[0]
    qb=summarize_moments(100,'G2',10.,args.power+1,args.repeats,args.seed,T0)[0]
    refinement=float(np.max(np.abs(qa-qb)))
    summary={'dimension_pairs':list(DIMENSIONS),'horizon_values':list(HORIZONS),'horizon_dimension_pairs':list(HORIZON_DIMENSIONS),
             'T_fixed':T_FIXED,'score_bias_count':len(BIASES),
             'total_bound_cases':len(all_cases),'qmc_points_per_scramble':2**args.power,
             'qmc_scrambles':args.repeats,'qmc_refinement_max_moment_change':refinement,
             'max_ode_difference':ode_error,'sample_generator_check':sampled_test,
             'computed_inequality_violations':len(comparisons),
             'figure_count':len(list(FIGURES.glob('*.jpg'))),
             'interpretation':'deterministic conditional Gaussian dynamics; numerical covariate integration by QMC'}
    (RESULTS/'validation.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))
    for model in MODELS:
        print('\n',model,'T=10, score bias=0.1')
        for r in csv.DictReader((RESULTS/f'table_{model}.csv').open()):
            print(f"p={r['p']:>3},d={r['d']:>3}: W2={float(r['W_act']):.4f}, "
                  f"direct={float(r['B_dir']):.4f}, loss={float(r['B_loss']):.4f}")


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--power',type=int,default=14,
                        help='Sobol points per scramble = 2**power')
    parser.add_argument('--repeats',type=int,default=16,
                        help='independent scrambled Sobol integrations')
    parser.add_argument('--seed',type=int,default=20261029)
    main(parser.parse_args())

