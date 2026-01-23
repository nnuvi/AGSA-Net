import torch

import numpy as np

from metrics import AvgrageMeter, accuracy
from utils import device

def train_epoch(model, model_name, train_loader, criterion, optimizer):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data, batch_target) in enumerate(train_loader):
        batch_data = batch_data.to(device)
        batch_target = batch_target.to(device)

        optimizer.zero_grad()
        if 'unmix' in model_name:
            re_unmix_nonlinear, re_unmix, batch_pred, embed_dim, depth, num_heads = model(batch_data)

            band = re_unmix.shape[1]//2  # 2 represents the number of layer
            output_linear = re_unmix[:,0:band] + re_unmix[:,band:band*2]
            re_unmix = re_unmix_nonlinear + output_linear

            eps = 1e-7

            norm_pred = torch.norm(re_unmix, dim=1, p=2) + eps
            norm_true = torch.norm(batch_data, dim=1, p=2) + eps

            # Calculate cosine similarity
            cos_sim = torch.sum(batch_data * re_unmix, dim=1) / (norm_pred * norm_true)
            cos_sim = torch.clamp(cos_sim, min=-1.0+eps, max=1.0-eps)

            sad_loss = torch.mean(torch.acos(cos_sim))

            alpha = 0.8
            loss = alpha*criterion(batch_pred, batch_target) + (1-alpha)*sad_loss
        else:
            batch_pred = model(batch_data)
            loss = criterion(batch_pred, batch_target)
        loss.backward()
        optimizer.step()

        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())
    return top1.avg, objs.avg, tar, pre, embed_dim, depth, num_heads

def valid_epoch(model, model_name, valid_loader, criterion, optimizer):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data, batch_target) in enumerate(valid_loader):
        batch_data = batch_data.to(device)
        batch_target = batch_target.to(device)

        if 'unmix' in model_name:
            re_unmix_nonlinear, re_unmix, batch_pred, embed_dim, depth, num_heads = model(batch_data)

            band = re_unmix.shape[1]//2
            output_linear = re_unmix[:,0:band] + re_unmix[:,band:band*2]
            re_unmix = re_unmix_nonlinear + output_linear

            eps = 1e-7

            norm_pred = torch.norm(re_unmix, dim=1, p=2) + eps
            norm_true = torch.norm(batch_data, dim=1, p=2) + eps

            # Calculate cosine similarity
            cos_sim = torch.sum(batch_data * re_unmix, dim=1) / (norm_pred * norm_true)
            cos_sim = torch.clamp(cos_sim, min=-1.0+eps, max=1.0-eps)

            sad_loss = torch.mean(torch.acos(cos_sim))

            loss = criterion(batch_pred, batch_target) + sad_loss
        else:
            batch_pred = model(batch_data)
            loss = criterion(batch_pred, batch_target)

        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())
    return tar, pre

def test_epoch(model, model_name, test_loader):
    pre = np.array([])
    for batch_idx, (batch_data, batch_target) in enumerate(test_loader):
        batch_data = batch_data.to(device)
        batch_target = batch_target.to(device)

        if 'unmix' in model_name:
            re_unmix_nonlinear, re_unmix, batch_pred, embed_dim, depth, num_heads = model(batch_data)
        else:
            batch_pred = model(batch_data)

        _, pred = batch_pred.topk(1, 1, True, True)
        pp = pred.squeeze()
        pre = np.append(pre, pp.data.cpu().numpy())

    return pre