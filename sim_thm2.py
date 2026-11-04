#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Theorem 2 """
import os
os.environ['OPENBLAS_NUM_THREADS'] = os.environ.get('CARD_BLAS_THREADS', '1')
os.environ['OMP_NUM_THREADS'] = os.environ.get('CARD_BLAS_THREADS', '1')
import argparse
import ast
import hashlib
import csv
import itertools
import json
import multiprocessing as mp
import platform
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.special import ndtr, logsumexp
from scipy.stats import t as student_t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
NAMES = ('A1', 'G3', 'G4')


EXPERIMENT_CONFIG = {
    # mode :capacity / all / quick / pilot / plot / check / learning
    'mode': 'capacity',
    'models': 'G3,G4',
    'dims': '10:10,10:20,10:50,10:80,'
            '20:10,20:20,20:50,20:80,'
            '50:10,50:20,50:50,50:80,50:100,',
    'widths': '16,32,64,128,256',
    'depth': 2,
    'repeats': 30,
    'steps': 2000,
    'batch': 128,
    'ntest': 32768,
    'ntime': 4096,
    'workers': min(4, os.cpu_count() or 1),
    'lr': 0.002,
    't0': 0.05,
    'T': 10.0,
    'time_dim': 'all',
    'target': 'oracle',
    'stress_dims': '80:80',
    'stress_widths': '64,128,256',
    'output': str(ROOT / 'theorem2_output1'),
    'resume': True,
    'time_widths': 'all',
    'time_grid': '0.05,0.1,0.2,0.5,1,2,5,10',
    'save_weights': True,
    'run_dsm': False,
    'run_stress': False,
    'dsm_dims': '20:20',
    'dsm_widths': '32,256',
}


PLOT_CONFIG = {
    'font_family': 'DejaVu Sans',
    'font_size': 11,
    'label_size': 11,
    'title_size': 13.2,
    'tick_size': 10,
    'grid': False,
    'dpi': 1000,
    'formats': ('pdf', 'jpg'),
    'colors': ['#2166ac', '#b35806', '#258069', '#8056a5', '#c44e52', '#8c6d31'],
    'markers': ['o', 's', '^', 'D', 'v', 'P'],
    'linewidth': 1.5,
    'markersize': 3,
    'capsize': 2,
    'model_titles': {'A1': 'Gaussian control', 'G3': 'Gaussian', 'G4': 'mixture'},
    'capacity': {                      # fig:score_capacity
        'figsize': (6.4, 4.2),
        'xlabel': r'Hidden width $w$',
        'ylabel_top': r'$\widehat{\mathcal{L}}_{\mathrm{SM}}$',
        'ylabel_bottom': r'$\widehat{\mathcal{L}}_{\mathrm{SM}}/d$',
        'title_template': '{model}: $p={p}$',
        'legend_template': '$d={d}$',
        'legend_title': None,
        'legend_loc': 'best',          # best / upper left / upper right / lower left
        'legend_ncol': 1,
        'legend_fontsize': 6.5,
        'legend_title_fontsize': 7,
        'legend_frame': False,
        'xscale': 'log',
        'yscale': 'linear',
        'xlim': None,
        'xpad_log2': 0.08,
        'ylim_top': None,
        'ylim_bottom': None,
        'xticks': None,
        'max_columns': 3,
    },
    'time': {                          # fig:score_time
        'figsize': (6.4, 4.2),
        'xlabel': r'Diffusion time $t$',
        'ylabel_top': r'$\widehat E(t)/d$',
        'ylabel_bottom': r'$(1-e^{-t})^2\widehat E(t)/d$',
        'title_template': '{model}: $(p,d)=({p},{d})$',
        'legend_template': '$w={width}$',
        'legend_title': None,
        'legend_loc': 'best',
        'legend_ncol': 1,
        'legend_fontsize': 7,
        'legend_title_fontsize': 7,
        'legend_frame': False,
        'xscale': 'log',
        'yscale': 'log',
        'xlim': None,
        'ylim_top': None,
        'ylim_bottom': None,
        'xticks': None,
    },
    'training': {                      # fig:training_diagnostics
        'figsize': (6.4, 4.2),
        'xlabel': 'Training updates',
        'ylabel': 'Validation score loss / $d$',
        'title_template': '{model}: $(p,d)=({p},{d})$',
        'legend_label': '$w={width}$',
        'show_legend': True,
        'legend_title': None,
        'legend_loc': 'best',
        'legend_ncol': 1,
        'legend_fontsize': 7,
        'legend_title_fontsize': 7,
        'legend_frame': False,
        'xscale': 'linear',
        'yscale': 'linear',
        'xlim': None,
        'ylim': None,
        'xticks': None,
    },
}

TABLE_CONFIG = {
    'main_widths': 'endpoints',  # endpoints / all / '16,64,256'
    'split_main_by_p': True,
}
CODE_SCHEMA = 2



# =====================================================================

def validate_dimensions(p, d):
    if not isinstance(p, (int, np.integer)) or not isinstance(d, (int, np.integer)) or p < 1 or (d < 1):
        raise ValueError('p and d must be positive integers.')

@lru_cache(maxsize=64)
def projection(p, d):
    validate_dimensions(p, d)
    rng = np.random.default_rng(np.random.SeedSequence([31871, p, d]))
    b = rng.normal(size=(d, p))
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    b.setflags(write=False)
    return b

def parameters(name, x, d):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError('X must have shape (batch,p), including when p=1.')
    validate_dimensions(x.shape[1], d)
    if name not in NAMES:
        raise ValueError(f'Unknown model {name}')
    base = np.sin(2 * np.pi * x.mean(axis=1))[:, None] * np.ones((1, d))
    # sqrt(12) 将均匀协变量中心化后的方差标准化；投影分别控制 d 个响应坐标。
    z = np.sqrt(12) * (x - 0.5) @ projection(x.shape[1], d).T
    if name == 'A1':
        return (base, np.full_like(base, 0.5), None)
    if name == 'G3':
        return (base, 0.25 + 0.5 * ndtr(z), None)
    return (base, np.full_like(base, 0.36), 0.2 + 0.6 * ndtr(z))

def mean(name, x, d):
    base, var, w = parameters(name, x, d)
    return base if w is None else base + 0.9 * (2 * w - 1)

def sample(name, x, d, rng):
    base, var, w = parameters(name, x, d)
    if w is not None:
        sign = 2 * (rng.uniform(size=base.shape) < w) - 1
        base = base + 0.9 * sign
    return base + np.sqrt(var) * rng.normal(size=base.shape)

def forward_parameters(name, x, d, t, center):
    base, var, w = parameters(name, x, d)
    tt = np.asarray(t, dtype=float)
    if tt.ndim == 0:
        tt = np.full(len(x), float(tt))
    if tt.shape != (len(x),) or np.any(tt < 0):
        raise ValueError('t must be nonnegative, scalar or shape (batch,).')

    a = np.exp(-tt / 2)[:, None]
    vt = np.exp(-tt)[:, None] * var + (-np.expm1(-tt))[:, None]
    center = np.asarray(center)
    if center.shape != base.shape:
        raise ValueError('center must have shape (batch,d).')
    if w is None:
        return (center + a * (base - center), vt, None)
    return (np.stack([center + a * (base - 0.9 - center), center + a * (base + 0.9 - center)], axis=-1), vt, w)

def score(name, y, x, t, center):
    y = np.asarray(y, dtype=float)
    if y.ndim != 2 or len(y) != len(x):
        raise ValueError('Y must have shape (batch,d).')
    loc, vt, w = forward_parameters(name, x, y.shape[1], t, center)
    if w is None:
        return (loc - y) / vt
    weights = np.stack([1 - w, w], axis=-1)
    logp = np.log(weights) - (y[..., None] - loc) ** 2 / (2 * vt[..., None])
    resp = np.exp(logp - logsumexp(logp, axis=-1, keepdims=True))
    return np.sum(resp * (loc - y[..., None]), axis=-1) / vt

def log_density(name, y, x, t, center):
    y = np.asarray(y, dtype=float)
    loc, vt, w = forward_parameters(name, x, y.shape[1], t, center)
    if w is None:
        return np.sum(-0.5 * np.log(2 * np.pi * vt) - (y - loc) ** 2 / (2 * vt), axis=1)
    weights = np.stack([1 - w, w], axis=-1)
    logp = np.log(weights) - 0.5 * np.log(2 * np.pi * vt[..., None]) - (y[..., None] - loc) ** 2 / (2 * vt[..., None])
    return logsumexp(logp, axis=-1).sum(axis=1)

def forward_sample(name, x, d, t, rng, center=None):
    if center is None:
        center = mean(name, x, d)
    y0 = sample(name, x, d, rng)
    xi = rng.normal(size=(len(x), d))
    tt = np.asarray(t)
    if tt.ndim == 0:
        tt = np.full(len(x), float(tt))

    sig = np.sqrt(-np.expm1(-tt))[:, None]
    if np.any(sig <= 0):
        raise ValueError('Forward score experiments require t>0.')
    r = np.exp(-tt / 2)[:, None] * (y0 - center) + sig * xi
    return (r, -xi / sig, center, xi)

def simulation_rng(stream, name, p, d, replicate):
    return np.random.default_rng(np.random.SeedSequence([20261030, stream, NAMES.index(name), p, d, replicate]))

def mc_summary(values):
    v = np.asarray(values, dtype=float)
    if len(v) < 2 or not np.all(np.isfinite(v)):
        raise ValueError('At least two finite replicates required.')
    sd = float(v.std(ddof=1))
    se = sd / np.sqrt(len(v))
    avg = float(v.mean())
    half = student_t.ppf(0.975, len(v) - 1) * se
    return dict(mean=avg, SD=sd, MCSE=se, CI95_low=avg - half, CI95_high=avg + half, replicates=len(v))

def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError('Cannot write an empty table.')
    with path.open('w', newline='', encoding='utf-8') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

def basis(x):
    x = np.asarray(x)
    return np.column_stack([np.ones(len(x)), x, np.sin(2 * np.pi * x), np.cos(2 * np.pi * x), np.sin(2 * np.pi * x.mean(axis=1))])


# =====================================================================
#  ReLU network
class ScoreNet:

    def __init__(self, rng, p=1, d=1, width=32, T=10.0, depth=2):
        validate_dimensions(p, d)
        if width < 1 or depth < 1 or T <= 0:
            raise ValueError('Require positive width, depth and T.')
        self.pdim = p
        self.d = d
        self.width = width
        self.T = T
        self.depth = depth
        dims = [p + d + 1] + [width] * depth + [d]
        self.params = []
        for j, (nin, nout) in enumerate(zip(dims[:-1], dims[1:])):
            self.params.extend([rng.normal(size=(nin, nout)) * np.sqrt(2 / nin) if j < depth else np.zeros((nin, nout)), np.zeros(nout)])
        self.first = [np.zeros_like(q) for q in self.params]
        self.second = [np.zeros_like(q) for q in self.params]
        self.step = 0

    @property
    def parameter_count(self):
        return sum((q.size for q in self.params))

    def forward(self, r, x, t, cache=False):
        r = np.asarray(r)
        x = np.asarray(x)
        t = np.asarray(t)
        if r.shape != (len(x), self.d) or x.shape[1:] != (self.pdim,):
            raise ValueError('Network input dimensions disagree.')
        if t.ndim == 0:
            t = np.full(len(x), float(t))
        f = np.column_stack([r, 2 * x - 1, 2 * t / self.T - 1])
        acts = [f]
        zs = []
        a = f
        for j in range(self.depth):
            z = a @ self.params[2 * j] + self.params[2 * j + 1]
            a = np.maximum(z, 0)
            zs.append(z)
            acts.append(a)
        out = -r + a @ self.params[-2] + self.params[-1]
        return (out, (acts, zs)) if cache else out

    def gradients(self, r, x, t, target, weights=None):
        pred, (acts, zs) = self.forward(r, x, t, True)
        diff = pred - target
        if weights is None:
            weights = np.ones(len(x))
        weights = np.asarray(weights)
        if weights.shape != (len(x),) or np.any(weights < 0):
            raise ValueError('Use nonnegative batch weights.')
        g = diff * weights[:, None] / (len(x) * self.d)
        grads = [np.empty_like(q) for q in self.params]
        grads[-2] = acts[-1].T @ g
        grads[-1] = g.sum(axis=0)
        g = g @ self.params[-2].T
        for j in reversed(range(self.depth)):
            g = g * (zs[j] > 0)
            grads[2 * j] = acts[j].T @ g
            grads[2 * j + 1] = g.sum(axis=0)
            if j:
                g = g @ self.params[2 * j].T
        return (float(0.5 * np.mean(weights[:, None] * diff ** 2)), grads)

    def update(self, grads, lr=0.002):
        self.step += 1
        norm = np.sqrt(sum((np.sum(g * g) for g in grads)))
        scale = min(1.0, 10 / max(norm, 1e-12))
        for j, g in enumerate(grads):
            g = g * scale
            self.first[j] = 0.9 * self.first[j] + 0.1 * g
            self.second[j] = 0.999 * self.second[j] + 0.001 * g * g
            m = self.first[j] / (1 - 0.9 ** self.step)
            v = self.second[j] / (1 - 0.999 ** self.step)
            self.params[j] -= lr * m / (np.sqrt(v) + 1e-08)

def generate(model, center_function, x, rng, ngen=512, K=512, t0=0.05):
    """Euler–Maruyama of  SDE"""
    x = np.asarray(x)
    xx = np.repeat(x, ngen, axis=0)
    m = center_function(xx)
    r = rng.normal(size=(len(xx), model.d))
    h = (model.T - t0) / K
    if h <= 0 or K < 1:
        raise ValueError('Require T>t0 and K>=1.')
    #  T -- t0
    for k in range(K):
        r += h * (0.5 * r + model.forward(r, xx, np.full(len(xx), model.T - k * h))) + np.sqrt(h) * rng.normal(size=r.shape)
    if not np.all(np.isfinite(r)):
        raise FloatingPointError('Nonfinite generated samples.')
    return (r + m).reshape(len(x), ngen, model.d)

def sliced_w2_squared(a, b, directions):
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError('Use equally sized (sample,d) vector samples.')
    return float(np.mean((np.sort(a @ directions.T, axis=0) - np.sort(b @ directions.T, axis=0)) ** 2))

def learning_one(name, p, d, n, rep, steps=1500, width=64, T=10.0, t0=0.05, K=512, ngen=512, ntest=8192):
    rng = simulation_rng(501, name, p, d, rep)
    xp = rng.uniform(size=(1024, p))
    yp = sample(name, xp, d, rng)
    coef = np.linalg.lstsq(basis(xp), yp, rcond=None)[0]
    center = lambda x: basis(x) @ coef
    xt = rng.uniform(size=(n, p))
    yt = sample(name, xt, d, rng)
    mt = center(xt)
    net = ScoreNet(simulation_rng(502, name, p, d, rep), p, d, width, T)
    for step in range(steps):
        ii = rng.integers(n, size=128)
        tt = rng.uniform(t0, T, size=128)
        sig = np.sqrt(-np.expm1(-tt))[:, None]
        xi = rng.normal(size=(128, d))
        rr = np.exp(-tt / 2)[:, None] * (yt[ii] - mt[ii]) + sig * xi
        _, grads = net.gradients(rr, xt[ii], tt, -xi / sig)
        net.update(grads)
    ev = simulation_rng(503, name, p, d, rep)
    xe = ev.uniform(size=(ntest, p))
    tt = ev.uniform(t0, T, size=ntest)
    rr, _, me, _ = forward_sample(name, xe, d, tt, ev, center(xe))
    loss = (T - t0) / 2 * np.mean(np.sum((net.forward(rr, xe, tt) - score(name, rr + me, xe, tt, me)) ** 2, axis=1))
    grid = ev.uniform(size=(5, p))
    draws = generate(net, center, grid, ev, ngen, K, t0)
    directions = ev.normal(size=(128, d))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    # 高斯基准使用相同的总监督样本预算 1024+n。
    xa = np.concatenate([xp, xt])
    ya = np.concatenate([yp, yt])
    gauss_coef = np.linalg.lstsq(basis(xa), ya, rcond=None)[0]
    residual = ya - basis(xa) @ gauss_coef
    vb = np.column_stack([np.ones(len(xa)), xa, xa * xa])
    var_coef = np.linalg.lstsq(vb, residual ** 2, rcond=None)[0]
    gm = basis(grid) @ gauss_coef
    gv = np.maximum(np.column_stack([np.ones(len(grid)), grid, grid * grid]) @ var_coef, 0.05)
    gaussian = gm[:, None, :] + np.sqrt(gv[:, None, :]) * ev.normal(size=draws.shape)
    sw = []
    gsw = []
    coverage = []
    gcoverage = []

    def marginal_cdf(y, xx):
        base, var, weights = parameters(name, xx[None, :], d)
        if weights is None:
            return ndtr((y - base[0]) / np.sqrt(var[0]))
        return (1 - weights[0]) * ndtr((y - base[0] + 0.9) / np.sqrt(var[0])) + weights[0] * ndtr((y - base[0] - 0.9) / np.sqrt(var[0]))
    for xx, draw, gdraw in zip(grid, draws, gaussian):
        target = sample(name, np.repeat(xx[None, :], ngen, axis=0), d, ev)
        sw.append(sliced_w2_squared(draw, target, directions))
        gsw.append(sliced_w2_squared(gdraw, target, directions))
        lo, hi = np.quantile(draw, [0.05, 0.95], axis=0)
        coverage.append(np.mean(marginal_cdf(hi, xx) - marginal_cdf(lo, xx)))
        lo, hi = np.quantile(gdraw, [0.05, 0.95], axis=0)
        gcoverage.append(np.mean(marginal_cdf(hi, xx) - marginal_cdf(lo, xx)))
    return dict(model=name, p=p, d=d, n=n, replicate=rep, steps=steps, width=width, parameters=net.parameter_count, SM_loss=float(loss), SM_loss_per_dim=float(loss / d), sliced_W2=float(np.sqrt(np.mean(sw))), marginal_coverage90=float(np.mean(coverage)), Gaussian_sliced_W2=float(np.sqrt(np.mean(gsw))), Gaussian_marginal_coverage90=float(np.mean(gcoverage)), K=K, ngen=ngen, ntest=ntest, T=T, t0=t0)



# =====================================================================

def prediction(net, r, x, t, chunk=2048):
    return np.concatenate([net.forward(r[i:i + chunk], x[i:i + chunk], t[i:i + chunk]) for i in range(0, len(x), chunk)])

def evaluation_data(name, p, d, rep, n, T, t0, stream=204, time_value=None):
    rng = simulation_rng(stream, name, p, d, rep)
    x = rng.uniform(size=(n, p))
    t = rng.uniform(t0, T, size=n) if time_value is None else np.full(n, time_value)
    r, g, m, _ = forward_sample(name, x, d, t, rng)
    truth = score(name, r + m, x, t, m)
    return (r, x, t, truth)

def evaluate(net, data, T, t0):
    r, x, t, truth = data
    mse = np.sum((prediction(net, r, x, t) - truth) ** 2, axis=1)
    baseline = np.sum((-r - truth) ** 2, axis=1)
    return (float((T - t0) / 2 * mse.mean()), float((T - t0) / 2 * baseline.mean()))

def fit_one(task):
    started=time.perf_counter(); name=task['model']; p=task['p']; d=task['d']
    width=task['width']; rep=task['replicate']; T=task['T']; t0=task['t0']
    key=f'{name}_p{p}_d{d}_w{width}_r{rep}'
    cache=Path(task['output'])/'replicates'; cache.mkdir(parents=True,exist_ok=True)
    result_path=cache/(key+'.json'); weights_path=cache/(key+'.npz')
    fingerprint=_training_fingerprint(task)
    evaluation_signature=dict(ntest=task['ntest'],ntime=task['ntime'],time_grid=task['time_grid'])
    old=None
    if task['resume'] and result_path.exists():
        old=json.loads(result_path.read_text(encoding='utf-8'))
        if isinstance(old,dict):
            if old.get('training_fingerprint')!=fingerprint:
                raise ValueError(f'{key} 的训练设置已改变，请使用新的 --output 目录。')
            if old.get('evaluation_signature')==evaluation_signature:
                old['status']='cached'
                return old
        else:
            old=None
    net=ScoreNet(simulation_rng(202+width,name,p,d,rep),p,d,width,T,task['depth'])
    checkpoints=[]; status='trained'
    if task['resume'] and weights_path.exists():
        with np.load(weights_path,allow_pickle=False) as snapshot:
            if str(snapshot['fingerprint'])!=fingerprint:
                raise ValueError(f'{key} 的权重指纹不匹配，请更换输出目录。')
            for i,param in enumerate(net.params):
                value=snapshot[f'parameter_{i}']
                if value.shape!=param.shape or not np.all(np.isfinite(value)):
                    raise ValueError(f'{key} 的权重形状或数值异常。')
                net.params[i]=value.copy()
            checkpoints=json.loads(str(snapshot['checkpoints_json']))
        status='evaluated'
    else:
        rng=simulation_rng(201,name,p,d,rep)
        validation=evaluation_data(name,p,d,rep,2048,T,t0,stream=205)
        cuts={0,task['steps']//4,task['steps']//2,3*task['steps']//4,task['steps']}
        for step in range(task['steps']+1):
            if step in cuts:
                val,_=evaluate(net,validation,T,t0)
                checkpoints.append(dict(model=name,p=p,d=d,width=width,replicate=rep,
                    target=task['target'],step=step,validation_loss=val,validation_loss_per_dim=val/d))
            if step==task['steps']:
                break
            x=rng.uniform(size=(task['batch'],p))
            t=np.exp(rng.uniform(np.log(t0),np.log(T),size=task['batch']))
            r,g,m,_=forward_sample(name,x,d,t,rng)
            label=g if task['target']=='dsm' else score(name,r+m,x,t,m)
            ###
            factor=t*np.log(T/t0)/(T-t0)
            _,grads=net.gradients(r,x,t,label,weights=factor)
            net.update(grads,lr=task['lr'])
        if task['save_weights']:
            _save_network(weights_path,net,fingerprint,checkpoints)
    value,baseline=evaluate(net,evaluation_data(name,p,d,rep,task['ntest'],T,t0),T,t0)
    if not np.isfinite(value):
        raise FloatingPointError(f'Nonfinite loss: {key}')
    timed=_time_errors(net,task)
    row=dict(model=name,p=p,d=d,width=width,depth=task['depth'],parameters=net.parameter_count,
        replicate=rep,target=task['target'],steps=task['steps'],batch=task['batch'],learning_rate=task['lr'],
        T=T,t0=t0,test_n=task['ntest'],SM_loss=value,SM_loss_per_dim=value/d,
        baseline_SM=baseline,relative_to_baseline=value/baseline if baseline>0 else None,
        seconds=time.perf_counter()-started)
    for item in timed:
        assert abs(item['MSE']/d-item['MSE_per_dim'])<1e-12
        assert abs(item['scaled_MSE_per_dim']-(-np.expm1(-item['t']))**2*item['MSE_per_dim'])<1e-12
    result=dict(schema=CODE_SCHEMA,training_fingerprint=fingerprint,evaluation_signature=evaluation_signature,
                row=row,time_rows=timed,checkpoints=checkpoints,status=status)
    _atomic_json(result_path,result)
    return result


def summarize(rows, keys, metrics):
    groups = {}
    for r in rows:
        groups.setdefault(tuple((r[k] for k in keys)), []).append(r)
    result = []
    for key, sub in sorted(groups.items()):
        r = dict(zip(keys, key))
        for metric in metrics:
            r.update({metric + '_' + k: v for k, v in mc_summary([a[metric] for a in sub]).items()})
        result.append(r)
    return result

def run_capacity(a):
    models=[name.strip() for name in a.models.split(',') if name.strip()]
    dims=parse_dims(a.dims); widths=parse_widths(a.widths)
    if not 0<a.t0<a.T or a.lr<=0 or not np.isfinite(a.lr):
        raise ValueError('0<t0<T ')
    if min(a.steps,a.batch,a.ntest,a.ntime,a.workers,a.depth)<1 or a.repeats<2:
        raise ValueError('repeats>=2。')
    if len(set(models))!=len(models) or any(name not in NAMES for name in models):
        raise ValueError('G3,G4 and A1')
    a.output=Path(a.output); a.output.mkdir(parents=True,exist_ok=True)
    config={key:str(value) if isinstance(value,Path) else value for key,value in vars(a).items()}
    config['dims']=','.join(f'{p}:{d}' for p,d in dims)
    config['widths']=','.join(map(str,widths))
    diagnostic_dims,diagnostic_widths,grid=diagnostic_settings(config)
    if not a.resume and (a.output/'config.json').exists():
        raise FileExistsError('outputs --resume。')
    tasks=[]
    for model in models:
        for p,d in dims:
            for width in widths:
                for rep in range(a.repeats):
                    tasks.append(dict(model=model,p=p,d=d,width=width,replicate=rep,
                        steps=a.steps,T=a.T,t0=a.t0,ntest=a.ntest,batch=a.batch,target=a.target,
                        lr=a.lr,depth=a.depth,ntime=a.ntime,
                        time_grid=grid if (p,d) in diagnostic_dims and width in diagnostic_widths else [],
                        output=str(a.output),resume=a.resume,save_weights=a.save_weights))
    for task in tasks:
        path=a.output/'replicates'/f"{task['model']}_p{task['p']}_d{task['d']}_w{task['width']}_r{task['replicate']}.json"
        if a.resume and path.exists():
            cached=json.loads(path.read_text(encoding='utf-8'))
            if isinstance(cached,dict) and cached.get('training_fingerprint')!=_training_fingerprint(task):
                raise ValueError(f'{path.name} parameters diff')
    _atomic_json(a.output/'config.json',config)
    print(f"{a.target}: {len(models)} models × {len(dims)} dims × {len(widths)} widths × {a.repeats} repeats = {len(tasks)} networks；diagnostic time {len(diagnostic_dims)} dim、{len(diagnostic_widths)} width。",flush=True)
    started=time.perf_counter(); rows=[]; timed=[]; checkpoints=[]
    counts=dict(trained=0,evaluated=0,cached=0)

    if a.workers==1:
        results=map(fit_one,tasks)
        pool=None
    else:
        pool=ProcessPoolExecutor(max_workers=a.workers,mp_context=mp.get_context('spawn'))
        futures=[pool.submit(fit_one,task) for task in tasks]
        results=(future.result() for future in as_completed(futures))
    try:
        for done,result in enumerate(results,1):
            rows.append(result['row']); timed.extend(result['time_rows']); checkpoints.extend(result['checkpoints'])
            counts[result['status']]+=1
            if done<=3 or done%10==0 or done==len(tasks):
                print(f"{a.target}: {done}/{len(tasks)} 完成；训练 {counts['trained']}，补算 {counts['evaluated']}，缓存 {counts['cached']}；耗时 {time.perf_counter()-started:.1f}s",flush=True)
    finally:
        if pool is not None:
            pool.shutdown(wait=True,cancel_futures=True)
    rows.sort(key=lambda row:(row['model'],row['p'],row['d'],row['width'],row['replicate']))
    timed.sort(key=lambda row:(row['model'],row['p'],row['d'],row['width'],row['replicate'],row['t']))
    checkpoints.sort(key=lambda row:(row['model'],row['p'],row['d'],row['width'],row['replicate'],row['step']))
    write_csv(a.output/'capacity_raw.csv',rows)
    summary=summarize(rows,['model','p','d','width','depth','parameters','target'],
        ['SM_loss','SM_loss_per_dim','baseline_SM'])
    write_csv(a.output/'capacity_summary.csv',summary)
    write_csv(a.output/'training_checkpoints.csv',checkpoints)
    if timed:
        write_csv(a.output/'time_raw.csv',timed)
        write_csv(a.output/'time_summary.csv',summarize(timed,['model','p','d','width','target','t'],
            ['MSE','MSE_per_dim','scaled_MSE_per_dim']))
    else:
        for filename in ('time_raw.csv','time_summary.csv'):
            (a.output/filename).unlink(missing_ok=True)
    paired=[]
    for model in models:
        for p,d in dims:
            reference={row['replicate']:row['SM_loss_per_dim'] for row in rows
                if (row['model'],row['p'],row['d'],row['width'])==(model,p,d,max(widths))}
            for width in widths:
                sub=[row for row in rows if (row['model'],row['p'],row['d'],row['width'])==(model,p,d,width)]
                paired.append(dict(model=model,p=p,d=d,width=width,reference_width=max(widths),
                    **mc_summary([row['SM_loss_per_dim']-reference[row['replicate']] for row in sub])))
    write_csv(a.output/'paired_width_differences.csv',paired)
    elapsed=time.perf_counter()-started
    _atomic_json(a.output/'run_metadata.json',dict(fitted_cells=len(summary),replicates=len(rows),target=a.target,
        new_fits_this_invocation=counts['trained'],diagnostics_refreshed=counts['evaluated'],cached_fits=counts['cached'],
        elapsed_seconds_this_invocation=elapsed,recorded_fit_seconds_sum=sum(row['seconds'] for row in rows),
        python=platform.python_version(),numpy=np.__version__,time_rows=len(timed)))
    return a.output



# =====================================================================
# check

def gradient_check(p, d, width=8, depth=2):
    rng = np.random.default_rng(482 + p + d)
    net = ScoreNet(rng, p, d, width, depth=depth)
    net.params[-2] = rng.normal(size=net.params[-2].shape) * 0.1
    x = rng.uniform(size=(9, p))
    r = rng.normal(size=(9, d))
    t = rng.uniform(0.05, 10, size=9)
    target = rng.normal(size=(9, d))
    weights = rng.uniform(0.1, 2, size=9)
    loss, g = net.gradients(r, x, t, target, weights)
    worst = 0.0
    for j in range(len(net.params)):
        for k in range(3):
            idx = tuple((rng.integers(s) for s in net.params[j].shape))
            old = net.params[j][idx]
            h = 1e-06
            net.params[j][idx] = old + h
            lp = 0.5 * np.mean(weights[:, None] * (net.forward(r, x, t) - target) ** 2)
            net.params[j][idx] = old - h
            lm = 0.5 * np.mean(weights[:, None] * (net.forward(r, x, t) - target) ** 2)
            net.params[j][idx] = old
            worst = max(worst, abs((lp - lm) / (2 * h) - g[j][idx]))
    assert worst < 3e-07, (p, d, worst)
    expected = (depth - 1) * width * width + (p + 2 * d + depth + 1) * width + d
    assert net.parameter_count == expected, (net.parameter_count, expected)
    return worst

def run_checks(output):
    rng = np.random.default_rng(781)
    gradients = {f'{p}:{d}:L{L}': gradient_check(p, d, depth=L) for p, d in ((1, 1), (7, 3), (3, 7), (20, 20)) for L in (1, 2, 3)}
    scores = {}
    moments = {}
    projection_z = {}
    for name in NAMES:
        for p, d in ((1, 1), (7, 3), (3, 7), (20, 20), (50, 100), (100, 50), (50, 50), (50, 80)):
            x = rng.uniform(size=(5, p))
            m = mean(name, x, d)
            y = rng.normal(size=(5, d))
            t = np.linspace(0.05, 10, 5)
            exact = score(name, y, x, t, m)
            err = 0.0
            for j in sorted({0, d // 2, d - 1}):
                yp = y.copy()
                ym = y.copy()
                yp[:, j] += 1e-05
                ym[:, j] -= 1e-05
                fd = (log_density(name, yp, x, t, m) - log_density(name, ym, x, t, m)) / 2e-05
                err = max(err, float(np.max(np.abs(fd - exact[:, j]))))
            assert err < 2e-06, (name, p, d, err)
            scores[f'{name}:{p}:{d}'] = err
            net = ScoreNet(rng, p, d, 16)
            tt = rng.uniform(0.05, 10, 5)
            rr, g, mm, _ = forward_sample(name, x, d, tt, rng)
            for _ in range(3):
                loss, gg = net.gradients(rr, x, tt, g)
                net.update(gg)
            assert np.isfinite(loss) and net.forward(rr, x, tt).shape == (5, d)
        p, d = (7, 5)
        n = 60000
        xf = rng.uniform(size=(1, p))
        x = np.repeat(xf, n, axis=0)
        base, v, w = parameters(name, xf, d)
        m = mean(name, xf, d)[0]
        y = sample(name, x, d, rng)
        variance = v[0] if w is None else v[0] + 4 * 0.9 ** 2 * w[0] * (1 - w[0])
        z = np.max(np.abs(y.mean(axis=0) - m) / np.sqrt(variance / n))
        assert z < 6, (name, z)
        relative_var = np.max(np.abs(y.var(axis=0) - variance) / variance)
        assert relative_var < 0.04, (name, relative_var)
        moments[name] = dict(max_mean_z=float(z), max_relative_variance_error=float(relative_var))
        n = 80000
        x = rng.uniform(size=(n, 7))
        tt = rng.uniform(0.05, 10, n)
        r, g, m, _ = forward_sample(name, x, 5, tt, rng)
        truth = score(name, r + m, x, tt, m)
        cand = -r
        residual = 0.5 * np.sum((cand - g) ** 2 - (cand - truth) ** 2 - (truth - g) ** 2, axis=1)
        se = residual.std(ddof=1) / np.sqrt(n)
        z = abs(residual.mean()) / se
        assert z < 6, (name, 'DSM identity', z)
        projection_z[name] = float(z)
    directions = np.eye(3)
    aa = rng.normal(size=(100, 3))
    assert sliced_w2_squared(aa, aa, directions) == 0.0

    class Exact:
        d = 3
        T = 10.0

        def forward(self, r, x, t):
            m = mean('A1', x, self.d)
            return score('A1', r + m, x, t, m)
    xx = rng.uniform(size=(1, 7))
    n = 16000
    draw = generate(Exact(), lambda x: mean('A1', x, 3), xx, rng, ngen=n, K=1000)[0]
    mu = mean('A1', xx, 3)[0]
    vt = 1 - 0.5 * np.exp(-0.05)
    sampler_mean_z = float(np.max(abs(draw.mean(axis=0) - mu) / np.sqrt(vt / n)))
    sampler_variance_error = float(np.max(abs(draw.var(axis=0) - vt) / vt))
    assert sampler_mean_z < 6 and sampler_variance_error < 0.06
    from numpy.polynomial.legendre import leggauss
    nodes, weights = leggauss(128)
    u = (nodes + 1) / 2
    weights = weights / 2
    t0, T = (0.05, 10.0)
    uniform = t0 + (T - t0) * u
    logarithmic = t0 * np.exp(np.log(T / t0) * u)

    def integrand(t):
        vt = 1 - 0.5 * np.exp(-t)
        return (vt - 1) ** 2 / vt
    direct = float(weights @ integrand(uniform))
    importance = float(weights @ (integrand(logarithmic) * logarithmic * np.log(T / t0) / (T - t0)))
    assert abs(direct - importance) < 1e-12
    ast.parse(Path(__file__).read_text(encoding='utf-8'))
    gradients['50:80:w256:L2'] = gradient_check(50,80,width=256,depth=2)
    properties = model_property_checks()
    out = dict(status='passed', network_gradient_errors=gradients, score_gradient_errors=scores, conditional_moment_checks=moments, DSM_projection_identity_z=projection_z, exact_gaussian_sampler=dict(mean_z=sampler_mean_z, relative_variance_error=sampler_variance_error), importance_integral_error=abs(direct - importance), model_property_checks=properties, dimensions_checked=[[1, 1], [7, 3], [3, 7], [20, 20], [50, 100], [100, 50], [50, 50], [50, 80]])
    output.parent.mkdir(exist_ok=True, parents=True)
    output.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out, indent=2))
    return out



def read_rows(path):
    with Path(path).open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in row.items():
            if key not in ('model', 'target'):
                try:
                    row[key] = float(value)
                except (ValueError, TypeError):
                    pass
    return rows


def configure_plot_style():
    style = PLOT_CONFIG
    plt.rcParams.update({
        'font.family': style['font_family'], 'font.size': style['font_size'],
        'axes.labelsize': style['label_size'], 'axes.titlesize': style['title_size'],
        'xtick.labelsize': style['tick_size'], 'ytick.labelsize': style['tick_size'],
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': style['grid'], 'pdf.fonttype': 42, 'pdf.use14corefonts': False,
    })


def save_figure(fig, figdir, name):
    figdir = Path(figdir)
    figdir.mkdir(parents=True, exist_ok=True)
    for extension in PLOT_CONFIG['formats']:
        options = {'facecolor': 'white'}
        if extension.lower() != 'pdf':
            options['dpi'] = PLOT_CONFIG['dpi']
        if extension.lower() in ('jpg', 'jpeg'):
            options['pil_kwargs'] = {'quality': 95}
        fig.savefig(figdir / f'{name}.{extension}', **options)
    plt.close(fig)


def apply_axes(ax,cfg,ylim=None,ticks=None,base2=False):
    if cfg['xscale']=='log' and base2:
        ax.set_xscale('log',base=2)
    else:
        ax.set_xscale(cfg['xscale'])
    ax.set_yscale(cfg['yscale']); ax.set_xlabel(cfg['xlabel'])
    chosen=cfg.get('xticks') if cfg.get('xticks') is not None else ticks
    if chosen is not None:
        ax.set_xticks(chosen,[f'{float(value):g}' for value in chosen])
    if cfg.get('xlim') is not None:
        ax.set_xlim(*cfg['xlim'])
    elif base2 and ticks:
        low,high=min(ticks),max(ticks)
        if low==high:
            ax.set_xlim(low/1.1,high*1.1)
        elif cfg['xscale']=='log':
            pad=2**cfg.get('xpad_log2',.08)
            ax.set_xlim(low/pad,high*pad)
        else:
            pad=.02*(high-low); ax.set_xlim(low-pad,high+pad)
    if ylim is not None:
        ax.set_ylim(*ylim)



def apply_legend(ax, cfg):
    ax.legend(title=cfg.get('legend_title'), loc=cfg['legend_loc'],
              fontsize=cfg['legend_fontsize'], title_fontsize=cfg['legend_title_fontsize'],
              ncol=cfg['legend_ncol'], frameon=cfg['legend_frame'])

def plot_capacity(lookup, dims, models, widths, figdir):
    cfg = PLOT_CONFIG['capacity']

    metrics = (
        ('SM_loss', 'ylabel_top', 'ylim_top'),
        ('SM_loss_per_dim', 'ylabel_bottom', 'ylim_bottom'),
    )

    for model in models:
        for p in sorted({p for p, d in dims}):
            for key, ylabel_key, ylim_key in metrics:
                # 每次只创建一个坐标轴，即一张独立图片
                fig, ax = plt.subplots(
                    figsize=cfg['figsize'], layout='constrained'
                )

                ds = sorted(d for pp, d in dims if pp == p)

                for d, color, marker in zip(
                    ds,
                    itertools.cycle(PLOT_CONFIG['colors']),
                    itertools.cycle(PLOT_CONFIG['markers']),
                ):
                    sub = [lookup[model, p, d, width] for width in widths]
                    av = np.array([item[key + '_mean'] for item in sub])
                    lo = np.array([item[key + '_CI95_low'] for item in sub])
                    hi = np.array([item[key + '_CI95_high'] for item in sub])

                    ax.errorbar(
                        widths, av,
                        yerr=np.vstack([av - lo, hi - av]),
                        fmt=marker + '-',
                        color=color,
                        markersize=PLOT_CONFIG['markersize'],
                        capsize=PLOT_CONFIG['capsize'],
                        linewidth=PLOT_CONFIG['linewidth'],
                        label=cfg['legend_template'].format(p=p, d=d),
                    )

                apply_axes(
                    ax, cfg, cfg[ylim_key], ticks=widths, base2=True
                )
                ax.set_ylabel(cfg[ylabel_key])
                ax.set_title(cfg['title_template'].format(
                    model=_display_model(model),
                    p=p,
                    description=PLOT_CONFIG['model_titles'].get(
                        _display_model(model), model
                    ),
                ))
                apply_legend(ax, cfg)

                save_figure(
                    fig, figdir,
                    f'score_capacity_{_display_model(model)}_p{p}_{key}'
                )


def plot_time(time_rows, models, widths, dims, figdir):
    cfg = PLOT_CONFIG['time']

    if not time_rows:
        print('error。', flush=True)
        return
    metrics = (
        ('MSE_per_dim', 'ylabel_top', 'ylim_top'),
        ('scaled_MSE_per_dim', 'ylabel_bottom', 'ylim_bottom'),
    )

    for p, d in dims:
        for model in models:
            selected = [
                item for item in time_rows
                if item['model'] == model
                and int(item['p']) == p
                and int(item['d']) == d
            ]
            if not selected:
                continue

            for key, ylabel_key, ylim_key in metrics:
                fig, ax = plt.subplots(
                    figsize=cfg['figsize'], layout='constrained'
                )

                for width, color, marker in zip(
                    widths,
                    itertools.cycle(PLOT_CONFIG['colors']),
                    itertools.cycle(PLOT_CONFIG['markers']),
                ):
                    sub = sorted(
                        [
                            item for item in selected
                            if int(item['width']) == width
                        ],
                        key=lambda item: item['t'],
                    )
                    if not sub:
                        continue

                    av = np.array([item[key + '_mean'] for item in sub])
                    lo = np.array([item[key + '_CI95_low'] for item in sub])
                    hi = np.array([item[key + '_CI95_high'] for item in sub])

                    ax.errorbar(
                        [item['t'] for item in sub], av,
                        yerr=np.vstack([av - lo, hi - av]),
                        fmt=marker + '-',
                        color=color,
                        markersize=PLOT_CONFIG['markersize'],
                        capsize=PLOT_CONFIG['capsize'],
                        linewidth=PLOT_CONFIG['linewidth'],
                        label=cfg['legend_template'].format(width=width),
                    )

                apply_axes(ax, cfg, cfg[ylim_key])
                ax.set_ylabel(cfg[ylabel_key])
                ax.set_title(cfg['title_template'].format(
                    model=_display_model(model), p=p, d=d
                ))
                if ax.get_legend_handles_labels()[0]:
                    apply_legend(ax, cfg)

                save_figure(
                    fig, figdir,
                    f'score_time_{_display_model(model)}_p{p}_d{d}_{key}'
                )


def plot_training(checkpoints, dims, models, widths, figdir):
    cfg = PLOT_CONFIG['training']

    for model in models:
        for p, d in dims:
            fig, ax = plt.subplots(
                figsize=cfg['figsize'], layout='constrained'
            )

            for width, color, marker in zip(
                widths,
                itertools.cycle(PLOT_CONFIG['colors']),
                itertools.cycle(PLOT_CONFIG['markers']),
            ):
                sub = [
                    item for item in checkpoints
                    if item['model'] == model
                    and int(item['p']) == p
                    and int(item['d']) == d
                    and int(item['width']) == width
                ]
                if not sub:
                    continue

                updates = sorted({item['step'] for item in sub})
                av = [
                    np.mean([
                        item['validation_loss_per_dim']
                        for item in sub if item['step'] == step
                    ])
                    for step in updates
                ]

                ax.plot(
                    updates, av, marker + '-',
                    markersize=PLOT_CONFIG['markersize'],
                    color=color,
                    lw=PLOT_CONFIG['linewidth'],
                    label=cfg['legend_label'].format(width=width),
                )

            apply_axes(ax, cfg, cfg['ylim'])
            ax.set_ylabel(cfg['ylabel'])
            ax.set_title(cfg['title_template'].format(
                model=_display_model(model), p=p, d=d
            ))
            if ax.get_legend_handles_labels()[0]:
                apply_legend(ax, cfg)

            save_figure(
                fig, figdir,
                f'training_{_display_model(model)}_p{p}_d{d}'
            )


def capacity_table(lookup, dims, models, widths, label, caption):
    out = [r'\begin{table}[pos=!htbp]', r'\footnotesize\centering',
           r'\caption{'+caption+'}', r'\label{'+label+'}',
           r'\setlength{\tabcolsep}{3pt}', r'\renewcommand{\arraystretch}{1.12}',
           r'\begin{tabularx}{\textwidth}{@{}crr *{'+str(len(models))+r'}{>{\centering\arraybackslash}Xrr}@{}}',
           r'\toprule', r'& & & '+' & '.join(r'\multicolumn{3}{c}{$\mathrm{'+_display_model(m)+r'}$}' for m in models)+r'\\',
           ''.join(r'\cmidrule(lr){'+str(4+3*j)+'-'+str(6+3*j)+'}' for j in range(len(models))),
           r'$(p,d)$ & $w$ & $P$ & '+' & '.join([r'Mean $\pm$ SD & MCSE & Mean/$d$']*len(models))+r'\\',
           r'\midrule']
    for i, (p,d) in enumerate(dims):
        if i:
            out.append(r'\midrule')
        for j, width in enumerate(widths):
            lead = r'\multirow{'+str(len(widths))+r'}{*}{$('+str(p)+','+str(d)+r')$}' if j == 0 else ''
            line = lead+f" & {width} & {int(lookup[models[0],p,d,width]['parameters'])}"
            for model in models:
                row = lookup[model,p,d,width]
                line += (f" & ${row['SM_loss_mean']:.4f} \\pm {row['SM_loss_SD']:.4f}$"
                         f" & {row['SM_loss_MCSE']:.4f} & {row['SM_loss_per_dim_mean']:.4f}")
            out.append(line+r'\\')
    out.extend([r'\bottomrule', r'\end{tabularx}', r'\end{table}'])
    return '\n'.join(out)+'\n'


def supplemental_tables(lookup, raw, texdir, dsm_input=None, stress_input=None):
    counts = {'network_fits_DSM': 0, 'network_fits_stress50': 0}
    if dsm_input is not None and (Path(dsm_input)/'capacity_summary.csv').exists():
        dsm_input = Path(dsm_input)
        summaries = read_rows(dsm_input/'capacity_summary.csv')
        dsm_raw = read_rows(dsm_input/'capacity_raw.csv')
        counts['network_fits_DSM'] = len(dsm_raw)
        contrasts = []
        for row in summaries:
            key = (row['model'],int(row['p']),int(row['d']),int(row['width']))
            if key not in lookup:
                continue
            a = {z['replicate']:z['SM_loss_per_dim'] for z in dsm_raw
                 if (z['model'],int(z['p']),int(z['d']),int(z['width'])) == key}
            b = {z['replicate']:z['SM_loss_per_dim'] for z in raw
                 if (z['model'],int(z['p']),int(z['d']),int(z['width'])) == key}
            if set(a) != set(b):
                raise ValueError('DSM 与 oracle 的重复编号必须一一对应。')
            contrasts.append(dict(model=key[0],p=key[1],d=key[2],width=key[3],
                oracle_per_dim=lookup[key]['SM_loss_per_dim_mean'],dsm_per_dim=row['SM_loss_per_dim_mean'],
                **mc_summary([a[k]-b[k] for k in sorted(a)])))
        if contrasts:
            write_csv(texdir.parent/'results'/'paired_objective_comparison.csv', contrasts)
            lines = [r'\begin{table}[pos=!htbp]',r'\centering\small',
                r'\caption{Independent test score loss per response coordinate for exact-score regression and DSM training.}',
                r'\label{tab:score_dsm}',r'\begin{tabular}{lrrrrrr}',r'\toprule',
                r'Model & $p$ & $d$ & $w$ & Exact-score & DSM & Paired MCSE\\',r'\midrule']
            for r in contrasts:
                lines.append(f"{r['model']} & {r['p']} & {r['d']} & {r['width']} & {r['oracle_per_dim']:.5f} & {r['dsm_per_dim']:.5f} & {r['MCSE']:.5f}"+r'\\')
            lines.extend([r'\bottomrule',r'\end{tabular}',r'\end{table}'])
            (texdir/'table_dsm_control.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if stress_input is not None and (Path(stress_input)/'capacity_summary.csv').exists():
        stress_input = Path(stress_input)
        rows = read_rows(stress_input/'capacity_summary.csv')
        cfg = json.loads((stress_input/'config.json').read_text(encoding='utf-8'))
        counts['network_fits_stress50'] = len(read_rows(stress_input/'capacity_raw.csv'))
        local = {(r['model'],int(r['p']),int(r['d']),int(r['width'])):r for r in rows}
        caption = ('Additional score-approximation results with '
                   f"$R={cfg['repeats']}$ independent fits per model--dimension--width setting.")
        (texdir/'table_score_stress50.tex').write_text(capacity_table(local,
            parse_dims(cfg['dims']),cfg['models'].split(','),parse_widths(cfg['widths']),
            'tab:score_stress',caption),encoding='utf-8')
    return counts


def generate_report(inp,report_dir,dsm_input=None,stress_input=None):
    inp=Path(inp);report_dir=Path(report_dir)
    cfg=json.loads((inp/'config.json').read_text(encoding='utf-8'))
    dims=parse_dims(cfg['dims']);models=cfg['models'].split(',');widths=parse_widths(cfg['widths'])
    rows=read_rows(inp/'capacity_summary.csv');raw=read_rows(inp/'capacity_raw.csv')
    lookup={(row['model'],int(row['p']),int(row['d']),int(row['width'])):row for row in rows}
    expected=len(models)*len(dims)*len(widths)*int(cfg['repeats'])
    if len(lookup)!=len(models)*len(dims)*len(widths) or len(raw)!=expected:
        raise ValueError('error')
    if len({(r['model'],r['p'],r['d'],r['width'],r['replicate']) for r in raw})!=expected:
        raise ValueError('error')
    if any(abs(r['SM_loss']/r['d']-r['SM_loss_per_dim'])>1e-12 for r in raw):
        raise ValueError('error')
    figdir=report_dir/'figures';texdir=report_dir/'latex';out=report_dir/'results'
    for directory in (figdir,texdir,out):
        directory.mkdir(parents=True,exist_ok=True)
    times=read_rows(inp/'time_summary.csv') if (inp/'time_summary.csv').exists() else []
    available={(r['model'],int(r['p']),int(r['d']),int(r['width']),float(r['t'])):r for r in times}
    grid=[float(t) for t in str(cfg.get('time_grid','0.05,0.1,0.2,0.5,1,2,5,10')).split(',') if t.strip() and float(cfg['t0'])<=float(t)<=float(cfg['T'])]
    missing=[]; incomplete=[]
    for model in models:
        for p,d in dims:
            for width in widths:
                for tv in grid:
                    key=(model,p,d,width,tv)
                    if key not in available:
                        missing.append(dict(model=model,p=p,d=d,width=width,t=tv))
                    elif int(available[key]['MSE_replicates'])!=int(cfg['repeats']):
                        incomplete.append(dict(model=model,p=p,d=d,width=width,t=tv))
    coverage=dict(complete_all_dimensions_and_widths=not missing and not incomplete,
        expected_cells=len(models)*len(dims)*len(widths)*len(grid),available_cells=len(available),
        missing=missing,incomplete=incomplete)
    _atomic_json(out/'diagnostic_coverage.json',coverage)
    if missing or incomplete:
        print(f'time_miss {len(missing)} set，{len(incomplete)} error；look diagnostic_coverage.json。',flush=True)
    configure_plot_style();plot_capacity(lookup,dims,models,widths,figdir)
    plot_time(times,models,widths,dims,figdir)
    plot_training(read_rows(inp/'training_checkpoints.csv'),dims,models,widths,figdir)
    main_setting=TABLE_CONFIG['main_widths']
    main_widths=sorted({min(widths),max(widths)}) if main_setting=='endpoints' else (widths if main_setting=='all' else parse_widths(main_setting))
    if not set(main_widths).issubset(widths):
        raise ValueError('table_width。')
    caption=(f"Integrated test score loss with $R={cfg['repeats']}$ independent fits per setting. "
        r'Mean, SD, and MCSE refer to $\widehat{\mathcal L}_{\mathrm{SM}}$; Mean/$d$ reports the per-coordinate mean. '
        f"All networks use ${cfg['depth']}$ hidden layers and ${cfg['steps']}$ updates.")
    main_table=(_all_widths_table(lookup,dims,models,main_widths,'tab:score_capacity',caption)
                if len(dims)*len(main_widths)>28 else
                capacity_table(lookup,dims,models,main_widths,'tab:score_capacity',caption))
    (texdir/'table_score_main.tex').write_text(main_table,encoding='utf-8')
    if TABLE_CONFIG['split_main_by_p']:
        for p in sorted({p for p,d in dims}):
            local_dims=[(pp,d) for pp,d in dims if pp==p]
            (texdir/f'table_score_p{p}.tex').write_text(capacity_table(lookup,local_dims,models,main_widths,
                f'tab:score_p{p}',caption.replace('Integrated test score loss with',f'Integrated test score loss at $p={p}$, with',1)),encoding='utf-8')
    (texdir/'table_score_all_widths.tex').write_text(_all_widths_table(lookup,dims,models,widths),encoding='utf-8')
    comparisons=[]
    for p,d in dims:
        for model in models:
            small=lookup[model,p,d,min(widths)];large=lookup[model,p,d,max(widths)]
            denominator=small['SM_loss_mean']
            comparisons.append(dict(model=model,p=p,d=d,small_mean=denominator,large_mean=large['SM_loss_mean'],
                reduction_percent=100*(1-large['SM_loss_mean']/denominator) if denominator>0 else None))
    counts=supplemental_tables(lookup,raw,texdir,dsm_input,stress_input)
    analysis=dict(capacity_comparisons=comparisons,network_fits_main=len(raw),
        time_diagnostic_coverage_complete=coverage['complete_all_dimensions_and_widths'],**counts)
    _atomic_json(out/'analysis.json',analysis)
    print(f'all plots：{figdir.resolve()}\nall LaTeX：{texdir.resolve()}',flush=True)
    return analysis




# =====================================================================
def parse_dims(text):
    try:
        dims=[tuple(map(int,item.strip().split(':'))) for item in str(text).split(',') if item.strip()]
        if not dims or any(len(item)!=2 for item in dims):
            raise ValueError
        for p,d in dims:
            validate_dimensions(p,d)
        if len(set(dims))!=len(dims):
            raise ValueError
    except (ValueError,TypeError) as exc:
        raise ValueError('dims p:d error') from exc
    return dims



def parse_widths(text):
    values=[int(value.strip()) for value in str(text).split(',') if value.strip()]
    if not values or min(values)<1 or len(set(values))!=len(values):
        raise ValueError('widths error。')
    return sorted(values)



def capacity_settings(args,target=None,dims=None,widths=None,output=None,time_dim=None):
    keys=('models','dims','widths','repeats','steps','batch','ntest','ntime','workers','depth',
          'target','lr','T','t0','time_dim','time_widths','time_grid','save_weights','resume')
    values={key:getattr(args,key) for key in keys}
    for key,value in [('target',target),('dims',dims),('widths',widths),('time_dim',time_dim)]:
        if value is not None:
            values[key]=value
    values['output']=Path(output) if output is not None else args.output/'results'/values['target']
    return SimpleNamespace(**values)



def run_learning(args):
    validate_dimensions(args.p,args.d)
    if args.n<1 or args.repeats<2 or args.steps<1 or args.width<1 or args.ntest<1:
        raise ValueError('n、steps、width、ntest >0，repeats >=2')
    if not 0<args.t0<args.T or args.K<1 or args.ngen<2:
        raise ValueError('0<t0<T、K>=1、ngen>=2。')
    rows=[]
    for replicate in range(args.repeats):
        row=learning_one(args.model,args.p,args.d,args.n,replicate,args.steps,args.width,
                         args.T,args.t0,args.K,args.ngen,args.ntest)
        rows.append(row)
        print(f'conditial_study：{replicate+1}/{args.repeats}，sliced W2={row["sliced_W2"]:.6f}',flush=True)
    folder=args.output/'learning'
    write_csv(folder/'learning_raw.csv',rows)
    summary=summarize(rows,['model','p','d','n','width'],
        ['SM_loss','SM_loss_per_dim','sliced_W2','Gaussian_sliced_W2',
         'marginal_coverage90','Gaussian_marginal_coverage90'])
    write_csv(folder/'learning_summary.csv',summary)
    (folder/'config.json').write_text(json.dumps({key:str(value) if isinstance(value,Path) else value
        for key,value in vars(args).items()},indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(f'conditial_study_results：{folder.resolve()}',flush=True)


def make_parser():
    cfg=EXPERIMENT_CONFIG
    parser=argparse.ArgumentParser(description='定理2：全维度、全宽度误差诊断',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mode',choices=('capacity','all','quick','pilot','plot','check','learning'),default=cfg['mode'])
    for key in ('models','dims','widths','time_dim','time_widths','time_grid','target','stress_dims','stress_widths','dsm_dims','dsm_widths'):
        parser.add_argument('--'+key.replace('_','-'),default=cfg[key])
    for key in ('repeats','steps','batch','ntest','ntime','workers','depth'):
        parser.add_argument('--'+key,type=int,default=cfg[key])
    for key in ('lr','T','t0'):
        parser.add_argument('--'+key,type=float,default=cfg[key])
    parser.add_argument('--output',type=Path,default=Path(cfg['output']))
    parser.add_argument('--input',type=Path,help='plot 模式读取的已有结果目录，内含 config.json')
    parser.add_argument('--dsm-input',type=Path);parser.add_argument('--stress-input',type=Path)
    parser.add_argument('--resume',dest='resume',action='store_true')
    parser.add_argument('--no-resume',dest='resume',action='store_false')
    parser.set_defaults(resume=cfg['resume'])
    parser.add_argument('--no-save-weights',dest='save_weights',action='store_false')
    parser.set_defaults(save_weights=cfg['save_weights'])
    parser.add_argument('--with-dsm',action='store_true',default=cfg['run_dsm'])
    parser.add_argument('--with-stress',action='store_true',default=cfg['run_stress'])
    parser.add_argument('--skip-dsm',action='store_true');parser.add_argument('--skip-stress',action='store_true')
    parser.add_argument('--skip-checks',action='store_true')
    parser.add_argument('--p',type=int,default=5);parser.add_argument('--d',type=int,default=5)
    parser.add_argument('--model',choices=NAMES,default='G4');parser.add_argument('--n',type=int,default=1024)
    parser.add_argument('--width',type=int,default=64);parser.add_argument('--K',type=int,default=512)
    parser.add_argument('--ngen',type=int,default=512)
    return parser



def main(argv=None):
    args=make_parser().parse_args(argv)
    planned=dict(models=args.models,dims=args.dims,widths=args.widths,repeats=args.repeats,
                 steps=args.steps,batch=args.batch,ntest=args.ntest,ntime=args.ntime)
    if args.target not in ('oracle','dsm'):
        raise ValueError('target 应为 oracle 或 dsm；它是训练标签选择，不是维度选择。')
    if args.mode in ('quick','pilot') and args.output==Path(EXPERIMENT_CONFIG['output']):
        args.output=ROOT/('theorem2_'+args.mode+'_output_v2')
    if args.mode=='quick':
        args.models='G3,G4';args.dims='7:3,3:7';args.widths='8,16,32'
        args.repeats=2;args.steps=20;args.batch=32;args.ntest=256;args.ntime=128
        args.time_dim='all';args.time_widths='all';args.workers=min(args.workers,2)
        PLOT_CONFIG['dpi']=150;PLOT_CONFIG['formats']=('pdf','png')
        print('check',flush=True)
    if args.mode=='pilot':
        dims=parse_dims(args.dims);p=max(p for p,d in dims);d=max(d for p,d in dims)
        args.dims=f'{p}:{d}';args.widths=str(max(parse_widths(args.widths)))
        args.repeats=2;args.time_dim='all';args.time_widths='all'
        PLOT_CONFIG['dpi']=150;PLOT_CONFIG['formats']=('pdf','png')
        print('little_expriments',flush=True)
    args.output=args.output.expanduser().resolve();args.output.mkdir(parents=True,exist_ok=True)
    if args.mode=='check':
        run_checks(args.output/'validation.json');return
    if args.mode=='plot':
        inp=args.input if args.input is not None else args.output/'results'/args.target
        dsm=args.dsm_input;stress=args.stress_input
        generate_report(inp,args.output,dsm,stress);return
    if args.mode=='learning':
        run_learning(args);return
    if args.mode in ('capacity','all','quick') and not args.skip_checks:
        run_checks(args.output/'validation.json')
    if args.mode=='all':
        args.target='oracle'
    main_input=run_capacity(capacity_settings(args))
    dsm_input=stress_input=None
    if args.mode=='all' and args.with_dsm and not args.skip_dsm:
        dsm_input=run_capacity(capacity_settings(args,target='dsm',dims=args.dsm_dims,widths=args.dsm_widths,
            time_dim='all',output=args.output/'results'/'dsm'))
    if args.mode=='all' and args.with_stress and not args.skip_stress:
        stress_input=run_capacity(capacity_settings(args,target='oracle',dims=args.stress_dims,
            widths=args.stress_widths,time_dim='all',output=args.output/'results'/'stress'))
    analysis=generate_report(main_input,args.output,dsm_input,stress_input)
    if args.mode=='pilot':
        raw=read_rows(main_input/'capacity_raw.csv')
        per_fit=float(np.mean([row['seconds'] for row in raw]))
        fits=len(planned['models'].split(','))*len(parse_dims(planned['dims']))*len(parse_widths(planned['widths']))*planned['repeats']
        budget=dict(planned_networks=fits,pilot_mean_fit_seconds=per_fit,workers=args.workers,
            rough_worker_hours=fits*per_fit/max(1,args.workers)/3600,
            note='time')
        _atomic_json(args.output/'budget_estimate.json',budget)
        print(json.dumps(budget,ensure_ascii=False,indent=2),flush=True)
    print(f'finish {analysis["network_fits_main"]} networks;outputs：{args.output}',flush=True)





def _display_model(name):
    return {'A2':'G3', 'A3':'G4'}.get(name,name)



def diagnostic_settings(config):
    dims=parse_dims(config['dims']); widths=parse_widths(config['widths'])
    dim_text=str(config.get('time_dim','all')).strip().lower()
    selected_dims=dims if dim_text=='all' else ([] if dim_text=='none' else parse_dims(dim_text))
    width_text=str(config.get('time_widths','all')).strip().lower()
    selected_widths=widths if width_text=='all' else ([] if width_text=='none' else parse_widths(width_text))
    if not set(selected_dims).issubset(dims) or not set(selected_widths).issubset(widths):
        raise ValueError('time,width')
    grid=sorted(set(float(t.strip()) for t in str(config.get('time_grid','0.05,0.1,0.2,0.5,1,2,5,10')).split(',') if t.strip()))
    if any(not np.isfinite(t) or t<=0 for t in grid):
        raise ValueError('time_grid>0')
    grid=[t for t in grid if float(config['t0'])<=t<=float(config['T'])]
    if selected_dims and selected_widths and not grid:
        raise ValueError('revision time_grid。')
    return selected_dims,selected_widths,grid



def _atomic_json(path,payload):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    temporary.replace(path)



def _training_fingerprint(task):
    keys=('model','p','d','width','replicate','steps','batch','target','lr','depth','T','t0')
    payload={key:task[key] for key in keys}
    payload['training_version']='residual_relu_log_uniform_v1'
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode('utf-8')).hexdigest()



def _save_network(path,net,fingerprint,checkpoints):
    temporary=Path(str(path)+'.tmp')
    arrays={f'parameter_{i}':value for i,value in enumerate(net.params)}
    arrays['fingerprint']=np.array(fingerprint)
    arrays['checkpoints_json']=np.array(json.dumps(checkpoints))
    with temporary.open('wb') as handle:
        np.savez_compressed(handle,**arrays)
    temporary.replace(path)



def _time_errors(net,task):
    rows=[]
    for j,tv in enumerate(task['time_grid']):
        r,x,t,truth=evaluation_data(task['model'],task['p'],task['d'],task['replicate'],
            task['ntime'],task['T'],task['t0'],stream=220+j,time_value=tv)
        mse=float(np.mean(np.sum((prediction(net,r,x,t)-truth)**2,axis=1)))
        rows.append(dict(model=task['model'],p=task['p'],d=task['d'],width=task['width'],
            replicate=task['replicate'],target=task['target'],t=tv,test_n=task['ntime'],
            MSE=mse,MSE_per_dim=mse/task['d'],scaled_MSE_per_dim=(-np.expm1(-tv))**2*mse/task['d']))
    return rows



def _all_widths_table(lookup,dims,models,widths,label='tab:score_capacity_full',caption='Integrated test score loss for all dimension and width configurations.'):
    columns='crr'+'rrr'*len(models)
    header=[r'\toprule',r'$(p,d)$ & $w$ & $P$ & '+' & '.join(
        [rf'\multicolumn{{3}}{{c}}{{{_display_model(model)}}}' for model in models])+r'\\',
        ' & & & '+' & '.join([r'Mean $\pm$ SD & MCSE & Mean/$d$']*len(models))+r'\\',r'\midrule']
    lines=[r'\begingroup\footnotesize',r'\setlength{\tabcolsep}{3pt}',
        r'\begin{longtable}{@{}'+columns+r'@{}}',
        r'\caption{'+caption+r'}\label{'+label+r'}\\']
    lines+=header+[r'\endfirsthead']+header+[r'\endhead',r'\bottomrule',r'\endfoot']
    for p,d in dims:
        for width in widths:
            line=f"$({p},{d})$ & {width} & {int(lookup[models[0],p,d,width]['parameters'])}"
            for model in models:
                item=lookup[model,p,d,width]
                line+=f" & ${item['SM_loss_mean']:.4f} \\pm {item['SM_loss_SD']:.4f}$ & {item['SM_loss_MCSE']:.4f} & {item['SM_loss_per_dim_mean']:.4f}"
            lines.append(line+r'\\')
    return '\n'.join(lines+[r'\end{longtable}',r'\endgroup'])+'\n'



def model_property_checks():
    rng=np.random.default_rng(91273)
    worst=0.0
    for p,d in ((10,10),(10,80),(20,50),(50,80)):
        b=projection(p,d)
        # Cov(X)=I_p/12，故 Var[sqrt(12)*b_j^T(X-1/2)]=||b_j||^2=1。
        worst=max(worst,float(np.max(np.abs(np.sum(b*b,axis=1)-1))))
        x=rng.uniform(size=(100,p))
        base,var,_=parameters('G3',x,d)
        assert np.all((var>=.25)&(var<=.75))
        base,component_var,w=parameters('G4',x,d)
        assert np.all((w>=.2)&(w<=.8))
        assert np.allclose(mean('G4',x,d),base+.9*(2*w-1),rtol=0,atol=1e-14)
        total_var=component_var+4*.9**2*w*(1-w)
        assert np.all((total_var>=.8784-1e-12)&(total_var<=1.17+1e-12))
    assert worst<1e-12
    return dict(projection_variance_max_error=worst,Gaussian_variance_bounds=[.25,.75],
        mixture_weight_bounds=[.2,.8],mixture_marginal_variance_bounds=[.8784,1.17],
        note='score,DSM,etc ')


if __name__=='__main__':
    mp.freeze_support()
    main()

