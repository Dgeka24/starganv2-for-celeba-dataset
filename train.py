import os
from os.path import join as ospj
import time
import datetime
from munch import Munch

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import build_model
from checkpoint import CheckpointIO
from datasets.datasets import InputFetcher


class Solver(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.nets = build_model(args)
        for name, module in self.nets.items():
            setattr(self, name, module)

        if args.mode == 'train':
            self.optims = Munch()
            for net in self.nets.keys():
                self.optims[net] = torch.optim.Adam(
                    params=self.nets[net].parameters(),
                    lr=args.f_lr if net == 'mapping_network' else args.lr,
                    betas=[args.beta1, args.beta2],
                    weight_decay=args.weight_decay)

            self.ckptios = [
                CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets.ckpt'), data_parallel=True, **self.nets),
                CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_optims.ckpt'), **self.optims)]

        self.to(self.device)
        for name, network in self.named_children():
            print('Initializing %s...' % name)
            network.apply(he_init)

    def _save_checkpoint(self, step):
        for ckptio in self.ckptios:
            ckptio.save(step)

    def _load_checkpoint(self, step):
        for ckptio in self.ckptios:
            ckptio.load(step)

    def _reset_grad(self):
        for optim in self.optims.values():
            optim.zero_grad()

    def train(self, loaders):
        args = self.args
        nets = self.nets
        optims = self.optims

        fetcher = InputFetcher(loaders.src, loaders.ref, args.latent_dim, 'train')

        if args.resume_iter > 0:
            self._load_checkpoint(args.resume_iter)

        initial_lambda_ds = args.lambda_ds

        print('Start training...')
        log_file = open('logs.txt', 'a')
        start_time = time.time()
        for i in range(args.resume_iter, args.total_iters):
            inputs = next(fetcher)
            x_real, y_org = inputs.x_src, inputs.y_src
            x_ref, x_ref2, y_trg = inputs.x_ref, inputs.x_ref2, inputs.y_ref
            z_trg, z_trg2 = inputs.z_trg, inputs.z_trg2

            d_loss, d_losses_latent = compute_d_loss(
                nets, args, x_real, y_org, y_trg, z_trg=z_trg)
            self._reset_grad()
            d_loss.backward()
            optims.discriminator.step()

            d_loss, d_losses_ref = compute_d_loss(
                nets, args, x_real, y_org, y_trg, x_ref=x_ref)
            self._reset_grad()
            d_loss.backward()
            optims.discriminator.step()

            g_loss, g_losses_latent = compute_g_loss(
                nets, args, x_real, y_org, y_trg, z_trgs=[z_trg, z_trg2])
            self._reset_grad()
            g_loss.backward()
            optims.generator.step()
            optims.mapping_network.step()
            optims.style_encoder.step()

            g_loss, g_losses_ref = compute_g_loss(
                nets, args, x_real, y_org, y_trg, x_refs=[x_ref, x_ref2])
            self._reset_grad()
            g_loss.backward()
            optims.generator.step()

            if args.lambda_ds > 0:
                args.lambda_ds -= (initial_lambda_ds / args.ds_iter)

            if (i+1) % args.print_every == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
                log = "Elapsed time [%s], Iteration [%i/%i], " % (elapsed, i+1, args.total_iters)
                all_losses = dict()
                for loss, prefix in zip([d_losses_latent, d_losses_ref, g_losses_latent, g_losses_ref],
                                        ['D/latent_', 'D/ref_', 'G/latent_', 'G/ref_']):
                    for key, value in loss.items():
                        all_losses[prefix + key] = value
                all_losses['G/lambda_ds'] = args.lambda_ds
                log += ' '.join(['%s: [%.4f]' % (key, value) for key, value in all_losses.items()])
                print(log)
                
                log_file.write(log + '\n')
                log_file.flush()
            
            if (i+1) % args.save_every == 0:
                self._save_checkpoint(step=i+1)
           
        log_file.close()



def compute_d_loss(nets, args, x_real, y_org, y_trg, z_trg=None, x_ref=None):
    assert (z_trg is None) != (x_ref is None)
    x_real.requires_grad_()
    out = nets.discriminator(x_real, y_org)
    loss_real = adv_loss(out, 1)
    loss_reg = r1_reg(out, x_real)

    with torch.no_grad():
        if z_trg is not None:
            s_trg = nets.mapping_network(z_trg, y_trg)
        else:
            s_trg = nets.style_encoder(x_ref, y_trg)

        x_fake = nets.generator(x_real, s_trg)
    out = nets.discriminator(x_fake, y_trg)
    loss_fake = adv_loss(out, 0)

    loss = loss_real + loss_fake + args.lambda_reg * loss_reg
    return loss, Munch(real=loss_real.item(),
                       fake=loss_fake.item(),
                       reg=loss_reg.item())


def compute_g_loss(nets, args, x_real, y_org, y_trg, z_trgs=None, x_refs=None):
    assert (z_trgs is None) != (x_refs is None)
    if z_trgs is not None:
        z_trg, z_trg2 = z_trgs
    if x_refs is not None:
        x_ref, x_ref2 = x_refs

    if z_trgs is not None:
        s_trg = nets.mapping_network(z_trg, y_trg)
    else:
        s_trg = nets.style_encoder(x_ref, y_trg)

    x_fake = nets.generator(x_real, s_trg)
    out = nets.discriminator(x_fake, y_trg)
    loss_adv = adv_loss(out, 1)

    s_pred = nets.style_encoder(x_fake, y_trg)
    loss_sty = torch.mean(torch.abs(s_pred - s_trg))

    if z_trgs is not None:
        s_trg2 = nets.mapping_network(z_trg2, y_trg)
    else:
        s_trg2 = nets.style_encoder(x_ref2, y_trg)
    x_fake2 = nets.generator(x_real, s_trg2)
    x_fake2 = x_fake2.detach()
    loss_ds = torch.mean(torch.abs(x_fake - x_fake2))

    s_org = nets.style_encoder(x_real, y_org)
    x_rec = nets.generator(x_fake, s_org)
    loss_cyc = torch.mean(torch.abs(x_rec - x_real))

    loss = loss_adv + args.lambda_sty * loss_sty \
        - args.lambda_ds * loss_ds + args.lambda_cyc * loss_cyc
    return loss, Munch(adv=loss_adv.item(),
                       sty=loss_sty.item(),
                       ds=loss_ds.item(),
                       cyc=loss_cyc.item())


def moving_average(model, model_test, beta=0.999):
    for param, param_test in zip(model.parameters(), model_test.parameters()):
        param_test.data = torch.lerp(param.data, param_test.data, beta)


def adv_loss(logits, target):
    assert target in [1, 0]
    targets = torch.full_like(logits, fill_value=target)
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    return loss


def r1_reg(d_out, x_in):
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    return reg

def he_init(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)