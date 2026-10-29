"""Supervision from measured candidates; gradients never enter the codec."""
import math
from statistics import NormalDist
import numpy as np
import torch
import torch.nn.functional as F

def oracle_weights(weights):
    if weights is None: return np.full(3,.25)
    value=np.asarray(weights,float)
    if value.shape!=(3,) or not np.isfinite(value).all() or not (value>=0).all():
        raise ValueError('nonnegative finite per-task oracle weights required')
    return value


def targets(rate,damage,mask,tolerance,budget,weights=None):
    rate=np.asarray(rate,float); damage=np.asarray(damage,float); mask=np.asarray(mask,bool)
    if rate.ndim!=1 or not len(rate) or damage.shape!=(len(rate),3) or mask.shape!=(3,) or not mask.any():
        raise ValueError('active measured task required')
    valid=np.isfinite(damage[:,mask]).all(1)&np.isfinite(rate)&(rate>0)&(rate<=budget)
    valid &= (damage[:,mask]<=np.asarray(tolerance)[mask]).all(1)
    if not valid[0] or abs(rate[0]-1)>1e-6 or np.max(np.abs(damage[0,mask]))>1e-6:
        raise ValueError('identity anchor missing')
    score=rate+(.25*np.nan_to_num(damage[:,mask]).mean(1) if weights is None else
                (oracle_weights(weights)[None,mask]*np.nan_to_num(damage[:,mask])).mean(1))
    return int(np.argmin(np.where(valid,score,np.inf)))


def targets_per_task(rate,damage,mask,tolerance,budget,weights=None):
    """Independent rate-aware labels; inactive tasks carry the sentinel -1."""
    active=np.asarray(mask,bool)
    if active.shape!=(3,) or not active.any(): raise ValueError('active measured task required')
    weights=oracle_weights(weights)
    labels=np.full(3,-1,np.int64)
    for task in np.flatnonzero(active):
        labels[task]=targets(rate,damage,np.arange(3)==task,tolerance,budget,weights)
    return labels


def relative_damage(rows,floor=None):
    """Express damage in identity-cost units without amplifying nearly perfect clips."""
    floor=np.array([1.,.25,.25]) if floor is None else np.asarray(floor,float)
    if floor.shape!=(3,) or not np.isfinite(floor).all() or not (floor>0).all():
        raise ValueError('positive finite per-task damage floor required')
    result=[]
    for r in rows:
        if 'cost' not in r: raise ValueError('measured cost required for relative damage')
        cost=np.asarray(r['cost'],float); mask=np.asarray(r['mask'],bool)
        absolute=r.get('damage_absolute',r['damage'])
        damage=np.asarray(absolute,float)
        if cost.ndim!=2 or cost.shape[1]!=3 or not len(cost) or damage.shape!=cost.shape or mask.shape!=(3,) or not mask.any():
            raise ValueError('invalid relative damage table')
        if not np.isfinite(cost[0,mask]).all(): raise ValueError('finite identity cost required for active tasks')
        if not np.isfinite(damage[:,mask]).all(): raise ValueError('finite damage required for active tasks')
        if np.any(np.abs(damage[0,mask])>1e-6): raise ValueError('identity anchor missing')
        reference=np.maximum(cost[0],floor)
        normalized=damage/reference
        normalized[0]=np.where(np.isfinite(damage[0]),0.,np.nan)
        clean=lambda table:[[float(v) if np.isfinite(v) else None for v in row] for row in table]
        result.append(dict(r,damage_absolute=clean(damage),damage=clean(normalized),
                           damage_reference=[float(v) if np.isfinite(v) else None for v in reference]))
    return result


def masked_loss(prediction,target,mask,scale=None):
    active=mask[:,None,:].expand_as(prediction)
    if not active.any(): raise ValueError('empty supervised task mask')
    if not torch.isfinite(target[active]).all(): raise ValueError('nonfinite active task target')
    if scale is None: return (prediction[active]-target[active]).square().mean()
    scale=torch.as_tensor(scale,device=prediction.device,dtype=prediction.dtype)
    if scale.shape!=(3,) or not (scale>0).all(): raise ValueError('positive task scales required')
    # Dimensionless residuals avoid fitting AR CE and track counts on one scale.
    # Huber limits outlier dominance without clipping the measured risk targets.
    return F.smooth_l1_loss((prediction/scale)[active],(target/scale)[active])

def pinball_loss(prediction,target,mask,quantile=.9,scale=None):
    """Masked quantile regression, with optional dimensionless task scales."""
    if not 0<quantile<1: raise ValueError('quantile outside (0,1)')
    active=mask[:,None,:].expand_as(prediction)
    if not active.any(): raise ValueError('empty supervised task mask')
    if not torch.isfinite(target[active]).all(): raise ValueError('nonfinite active task target')
    residual=target[active]-prediction[active]
    if scale is not None:
        scale=torch.as_tensor(scale,device=prediction.device,dtype=prediction.dtype)
        if scale.shape!=(3,) or not torch.isfinite(scale).all() or not (scale>0).all():
            raise ValueError('positive finite task scales required')
        residual=residual/scale.expand_as(prediction)[active]
    return torch.maximum(quantile*residual,(quantile-1)*residual).mean()


def _calibration_residuals(predicted,actual,rate_pred,rate_actual,mask,coverage):
    p,a=np.asarray(predicted,float),np.asarray(actual,float)
    mask=np.asarray(mask,bool)
    if p.shape!=a.shape or p.ndim!=3 or p.shape[2]!=3 or not p.shape[1] or mask.shape!=(len(p),3) or not 0<coverage<1:
        raise ValueError('invalid calibration table')
    rp,ra=np.asarray(rate_pred,float),np.asarray(rate_actual,float)
    if rp.shape!=p.shape[:2] or ra.shape!=rp.shape:
        raise ValueError('invalid calibrated rates')
    r=ra-rp
    if not np.isfinite(r).all(): raise ValueError('invalid calibrated rates')
    residual=a-p
    for k in range(3):
        if not np.isfinite(residual[mask[:,k],:,k]).all(): raise ValueError('invalid calibrated risks')
    return residual,r,mask


def calibrate(predicted,actual,rate_pred,rate_actual,mask,coverage,mode='quantile'):
    if mode=='mean_ucb': return calibrate_mean_ucb(predicted,actual,rate_pred,rate_actual,mask,coverage)
    if mode!='quantile': raise ValueError('unknown calibration mode')
    residual,r,mask=_calibration_residuals(predicted,actual,rate_pred,rate_actual,mask,coverage)
    def upper(values):
        # Finite-sample one-sided order statistic. This is empirical calibration,
        # not a guarantee under distribution shift or correlated codec conditions.
        if not len(values): return 1e6
        return max(0.,float(np.sort(values)[min(len(values)-1,math.ceil((len(values)+1)*coverage)-1)]))
    risk=[[upper(residual[mask[:,k],j,k]) for k in range(3)] for j in range(residual.shape[1])]
    risk[0]=[0.]*3
    rates=[upper(r[:,j]) for j in range(residual.shape[1])]; rates[0]=0.
    return dict(risk=risk,rate=rates,coverage=coverage,count=len(residual),scope='empirical one-sided CAL residuals')


def calibrate_mean_ucb(predicted,actual,rate_pred,rate_actual,mask,coverage):
    """One-sided normal UCB on clipped mean residuals, not per-row coverage."""
    residual,r,mask=_calibration_residuals(predicted,actual,rate_pred,rate_actual,mask,coverage)
    z=NormalDist().inv_cdf(coverage)
    def upper(values):
        if not len(values): return 1e6
        values=np.clip(values,-1.,1.)
        sem=float(values.std(ddof=1)/np.sqrt(len(values))) if len(values)>1 else 0.
        return max(0.,float(values.mean())+z*sem)
    risk=[[upper(residual[mask[:,k],j,k]) for k in range(3)] for j in range(residual.shape[1])]
    rates=[upper(r[:,j]) for j in range(residual.shape[1])]
    risk[0]=[0.]*3; rates[0]=0.
    return dict(risk=risk,rate=rates,coverage=coverage,count=len(residual),mode='mean_ucb',
                residual_clip=[-1.,1.],scope='one-sided mean CAL residual UCB; clipped residuals; correlated rows, no individual coverage guarantee')

def select_action(risk,rate,calibration,mask,tolerance,budget):
    risk=np.asarray(risk,float).copy(); rate=np.asarray(rate,float).copy(); mask=np.asarray(mask,bool)
    if risk.shape!=(len(rate),3) or mask.shape!=(3,) or not mask.any(): raise ValueError('invalid selection inputs')
    bounds=np.asarray(calibration['risk'])+risk; rate_bound=rate+np.asarray(calibration['rate'])
    bounds[0]=0; rate_bound[0]=1; risk[0]=0; rate[0]=1
    feasible=(bounds[:,mask]<=np.asarray(tolerance)[mask]).all(1)&(rate_bound<=budget)
    feasible &= np.isfinite(bounds[:,mask]).all(1)&np.isfinite(rate_bound)
    score=rate+.25*risk[:,mask].mean(1)
    return int(np.argmin(np.where(feasible,score,np.inf))) if feasible.any() else 0

def select_direct_action(out,mask,tolerance,budget,weights=None,domain_idx=None):
    """Rank predicted-feasible actions by task logits, then rate/risk, then index.

    A supplied domain selects that task's logits; every active task still gates
    feasibility. The known identity risk/rate overrides its learned predictions.
    """
    risk=np.asarray(out['risk'],float).copy(); rate=np.asarray(out['rate'],float).copy()
    mask=np.asarray(mask,bool); tolerance=np.asarray(tolerance,float)
    if rate.ndim!=1 or not len(rate) or risk.shape!=(len(rate),3) or mask.shape!=(3,) or not mask.any() or tolerance.shape!=(3,):
        raise ValueError('invalid direct selection inputs')
    weights=oracle_weights(weights)
    if domain_idx is not None and (not isinstance(domain_idx,(int,np.integer)) or domain_idx not in (0,1,2) or not mask[domain_idx]):
        raise ValueError('direct domain must be an active task')
    if 'logits_task' in out:
        task=np.asarray(out['logits_task'],float)
        if task.shape!=risk.shape: raise ValueError('invalid task logits')
        if domain_idx is not None:
            logits=task[:,domain_idx]
        else:
            active_weights=weights[mask]
            logits=(task[:,mask]*active_weights).sum(1)/active_weights.sum() if active_weights.any() else np.zeros(len(rate))
    else:
        logits=np.asarray(out['logits'],float)
    if logits.shape!=rate.shape: raise ValueError('invalid direct logits')
    risk[0]=0.; rate[0]=1.
    feasible=np.isfinite(risk[:,mask]).all(1)&np.isfinite(rate)&np.isfinite(logits)&(rate>0)&(rate<=budget)
    feasible &= (risk[:,mask]<=tolerance[mask]).all(1)
    candidates=np.flatnonzero(feasible)
    if not len(candidates): return 0
    cand_logits=logits[candidates]-np.max(logits[candidates])
    prob=np.exp(cand_logits); prob=prob/np.sum(prob)
    objective=rate[candidates]+(weights[None,mask]*np.maximum(risk[candidates][:,mask],0.)).mean(1)-.25*prob
    order=np.lexsort((candidates,-logits[candidates],objective))
    return int(candidates[order[0]])


def calibrate_domains(rows,predictions,coverage,mode='quantile'):
    domains={}
    for domain in sorted({r['domain'] for r in rows}):
        ids=[i for i,r in enumerate(rows) if r['domain']==domain]
        selected=[rows[i] for i in ids]; pred=[predictions[i] for i in ids]
        value=calibrate(np.array([p['risk'] for p in pred]),np.array([r['damage'] for r in selected],float),
                        np.array([p['rate'] for p in pred]),np.array([r['rate'] for r in selected]),
                        np.array([r['mask'] for r in selected],bool),coverage,mode=mode)
        value['source_count']=len({r.get('id',str(i)) for i,r in enumerate(selected)})
        domains[domain]=value
    return dict(domains=domains,count=len(rows),coverage=coverage,
                scope='domain-specific empirical CAL residuals; correlated QP/window rows; no coverage guarantee')

def domain_calibration(calibration,domain):
    if 'domains' not in calibration: return calibration
    if domain not in calibration['domains']: raise ValueError('missing domain calibration')
    return calibration['domains'][domain]

def train_controller(model,rows,inputs,cfg,device):
    task_logits=getattr(model,'task_logits',False)
    if 'task_logits' in cfg and bool(cfg['task_logits'])!=task_logits:
        raise ValueError('controller task-logit configuration mismatch')
    loss_mode=cfg.get('risk_loss')
    if loss_mode not in (None,'huber','pinball'): raise ValueError('unknown controller risk loss')
    task_weight=float(cfg.get('task_ce_weight',1. if task_logits else 0.))
    if not math.isfinite(task_weight) or task_weight<0: raise ValueError('nonnegative task CE weight required')
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg['lr']); history=[]
    model.to(device).train()
    groups={}
    for i,r in enumerate(rows):
        groups.setdefault(inputs[r['id']][0].shape[0],[]).append(i)
    c=torch.tensor([r['condition'] for r in rows],device=device,dtype=torch.float32)
    rate=torch.tensor([r['rate'] for r in rows],device=device,dtype=torch.float32)
    damage=torch.tensor([[[float('nan') if v is None else v for v in a] for a in r['damage']] for r in rows],device=device,dtype=torch.float32)
    mask=torch.tensor([r['mask'] for r in rows],device=device,dtype=torch.bool)
    labels=torch.tensor([targets(r['rate'],np.array(r['damage'],float),r['mask'],cfg['damage_tolerance'],cfg['budget_ratio'],cfg.get('oracle_damage_weight')) for r in rows],device=device)
    labels_task=None
    if task_logits and task_weight:
        labels_task=torch.as_tensor(np.stack([targets_per_task(r['rate'],r['damage'],r['mask'],cfg['damage_tolerance'],cfg['budget_ratio'],cfg.get('oracle_damage_weight')) for r in rows]),device=device)
    def risk_loss(prediction,target,active):
        if loss_mode=='pinball':
            return pinball_loss(prediction,target,active,cfg.get('risk_quantile',.9),cfg.get('risk_scale'))
        scale=cfg.get('risk_scale')
        if loss_mode=='huber' and scale is None: scale=[1.]*3
        return masked_loss(prediction,target,active,scale)
    for epoch in range(cfg['epochs']):
        batches=[]; values=[]
        for group in groups.values():
            indices=torch.tensor(group)[torch.randperm(len(group))]
            batches.extend(indices.split(16))
        for index in torch.randperm(len(batches)).tolist():
            ids=batches[index].tolist(); batch=batches[index].to(device)
            x=torch.stack([inputs[rows[i]['id']][0] for i in ids]).to(device)
            m=torch.stack([inputs[rows[i]['id']][1] for i in ids]).to(device)
            out=model(x,m,c[batch])
            loss=F.cross_entropy(out['logits'],labels[batch])+4*risk_loss(out['risk'],damage[batch],mask[batch])+4*F.mse_loss(out['rate'],rate[batch])
            if labels_task is not None:
                loss=loss+task_weight*F.cross_entropy(out['logits_task'].permute(0,2,1)[mask[batch]],labels_task[batch][mask[batch]])
            if not torch.isfinite(loss): raise ValueError('nonfinite controller loss')
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5); optimizer.step()
            values.append(float(loss.detach()))
        history.append(dict(epoch=epoch+1,loss=float(np.mean(values))))
    model.eval(); return history
