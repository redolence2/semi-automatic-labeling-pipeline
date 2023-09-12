import argparse
import datetime
import os
import traceback

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn
from torchvision import transforms
from tqdm.autonotebook import tqdm

from val import val
from backbone import HybridNetsBackbone
from utils.utils import get_last_weights, init_weights, boolean_string, \
    save_checkpoint, DataLoaderX, Params
from hybridnets.dataset import BddDataset
from hybridnets.custom_dataset import CustomDataset
from hybridnets.autoanchor import run_anchor
from hybridnets.model import ModelWithLoss
from utils.constants import *
from collections import OrderedDict
# from torchinfo import summary
from functools import partial
import sys
import cv2

def get_args():
    parser = argparse.ArgumentParser('HybridNets: End-to-End Perception Network - DatVu')
    parser.add_argument('-p', '--project', type=str, default='bdd100k', help='Project file that contains parameters')
    parser.add_argument('-bb', '--backbone', type=str, help='Use timm to create another backbone replacing efficientnet. '
                                                            'https://github.com/rwightman/pytorch-image-models')
    parser.add_argument('-c', '--compound_coef', type=int, default=3, help='Coefficient of efficientnet backbone')
    parser.add_argument('-n', '--num_workers', type=int, default=8, help='Num_workers of dataloader')
    parser.add_argument('-b', '--batch_size', type=int, default=12, help='Number of images per batch among all devices')
    parser.add_argument('--freeze_backbone', type=boolean_string, default=False,
                        help='Freeze encoder and neck (effnet and bifpn)')
    parser.add_argument('--freeze_det', type=boolean_string, default=False,
                        help='Freeze detection head')
    parser.add_argument('--freeze_seg', type=boolean_string, default=False,
                        help='Freeze segmentation head')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--optim', type=str, default='adamw', help='Select optimizer for training, '
                                                                   'suggest using \'adamw\' until the'
                                                                   ' very final stage then switch to \'sgd\'')
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--val_interval', type=int, default=1, help='Number of epoches between valing phases')
    parser.add_argument('--save_interval', type=int, default=500, help='Number of steps between saving')
    parser.add_argument('--es_min_delta', type=float, default=0.0,
                        help='Early stopping\'s parameter: minimum change loss to qualify as an improvement')
    parser.add_argument('--es_patience', type=int, default=0,
                        help='Early stopping\'s parameter: number of epochs with no improvement after which '
                             'training will be stopped. Set to 0 to disable this technique')
    parser.add_argument('--data_path', type=str, default='datasets/', help='The root folder of dataset')
    parser.add_argument('--log_path', type=str, default='checkpoints/')
    parser.add_argument('-w', '--load_weights', type=str, default=None,
                        help='Whether to load weights from a checkpoint, set None to initialize,'
                             'set \'last\' to load last checkpoint')
    parser.add_argument('--saved_path', type=str, default='checkpoints/')
    parser.add_argument('--debug', type=boolean_string, default=False,
                        help='Whether visualize the predicted boxes of training, '
                             'the output images will be in test/, '
                             'and also only use first 500 images.')
    parser.add_argument('--cal_map', type=boolean_string, default=True,
                        help='Calculate mAP in validation')
    parser.add_argument('-v', '--verbose', type=boolean_string, default=True,
                        help='Whether to print results per class when valing')
    parser.add_argument('--plots', type=boolean_string, default=True,
                        help='Whether to plot confusion matrix when valing')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='Number of GPUs to be used (0 to use CPU)')
    parser.add_argument('--conf_thres', type=float, default=0.001,
                        help='Confidence threshold in NMS')
    parser.add_argument('--iou_thres', type=float, default=0.6,
                        help='IoU threshold in NMS')
    parser.add_argument('--amp', type=boolean_string, default=False,
                        help='Automatic Mixed Precision training')

    parser.add_argument('--warm_lr', type=float, default=1e-6)
    parser.add_argument('--poly_end_lr', type=float, default=1e-6)
    parser.add_argument('--poly_decay_power', type=float, default=0.9)
    parser.add_argument('--warmup_epoch', type=int, default=8)

    parser.add_argument('--freeze_instance', type=boolean_string, default=False,
                        help='Freeze instance head')
    parser.add_argument('--lr_scheduler', type=str, default='poly')
    parser.add_argument('--vis_input', type=boolean_string, default=False)



    args = parser.parse_args()
    return args

def train(opt):
    experient_name = "_stage1_lr5e4_"
    
    torch.backends.cudnn.benchmark = True
    print("\nCUDNN VERSION: {}\n".format(torch.backends.cudnn.version()))
    params = Params(f'projects/{opt.project}.yml')

    if opt.num_gpus == 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    else:
        torch.manual_seed(42)

    opt.saved_path = opt.saved_path + f'/{opt.project}/'
    opt.log_path = opt.log_path + f'/{opt.project}/tensorboard/'
    os.makedirs(opt.log_path, exist_ok=True)
    os.makedirs(opt.saved_path, exist_ok=True)

    seg_mode = MULTILABEL_MODE if params.seg_multilabel else MULTICLASS_MODE if len(params.seg_list) > 1 else BINARY_MODE

    train_dataset = BddDataset(
        params=params,
        is_train=True,
        inputsize=params.model['image_size'],
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=params.mean, std=params.std
            )
        ]),
        seg_mode=seg_mode,
        debug=opt.debug
    )

    training_generator = DataLoaderX(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=params.pin_memory,
        collate_fn=BddDataset.collate_fn
    )

    valid_dataset = BddDataset(
        params=params,
        is_train=False,
        inputsize=params.model['image_size'],
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=params.mean, std=params.std
            )
        ]),
        seg_mode=seg_mode,
        debug=opt.debug
    )

    val_generator = DataLoaderX(
        valid_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=params.pin_memory,
        collate_fn=BddDataset.collate_fn
    )

    if params.need_autoanchor:
        params.anchors_scales, params.anchors_ratios = run_anchor(None, train_dataset)

    model = HybridNetsBackbone(num_classes=len(params.obj_list), compound_coef=opt.compound_coef,
                               ratios=eval(params.anchors_ratios), scales=eval(params.anchors_scales),
                               seg_classes=len(params.seg_list), backbone_name=opt.backbone,
                               seg_mode=seg_mode)

    # load last weights
    ckpt = {}
    # last_step = None
    if opt.load_weights:
        if opt.load_weights.endswith('.pth'):
            weights_path = opt.load_weights
        else:
            weights_path = get_last_weights(opt.saved_path)
        # try:
        #     last_step = int(os.path.basename(weights_path).split('_')[-1].split('.')[0])
        # except:
        #     last_step = 0

        try:
            ckpt = torch.load(weights_path)
            # new_weight = OrderedDict((k[6:], v) for k, v in ckpt['model'].items())
            model.load_state_dict(ckpt.get('model', ckpt), strict=False)
        except RuntimeError as e:
            print(f'[Warning] Ignoring {e}')
            print(
                '[Warning] Don\'t panic if you see this, this might be because you load a pretrained weights with different number of classes. The rest of the weights should be loaded already.')
    else:
        print('[Info] initializing weights...')
        init_weights(model)

    print('[Info] Successfully!!!')

    if opt.freeze_backbone:
        model.encoder.requires_grad_(False)
        model.bifpn.requires_grad_(False)
        print('[Info] freezed backbone')

    # if opt.freeze_det:
    #     model.regressor.requires_grad_(False)
    #     model.classifier.requires_grad_(False)
    #     model.anchors.requires_grad_(False)
    #     print('[Info] freezed detection head')

    if opt.freeze_seg:
        model.bifpndecoder.requires_grad_(False)
        model.segmentation_head.requires_grad_(False)
        print('[Info] freezed segmentation head')

    if opt.freeze_instance:
        model.bifpndecoder_embedding.requires_grad_(False)
        model.embedding_head.requires_grad_(False)
        print('[Info] freezed instance head')
    #summary(model, (1, 3, 384, 640), device='cpu')

    writer = SummaryWriter(opt.log_path + f'/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}_' + experient_name + '/')

    # wrap the model with loss function, to reduce the memory usage on gpu0 and speedup
    model = ModelWithLoss(model, debug=opt.debug)

    model = model.to(memory_format=torch.channels_last)

    if opt.num_gpus > 0:
        model = model.cuda()

    if opt.optim == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), opt.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), opt.lr, momentum=0.9, nesterov=True)
    # print(ckpt)
    scaler = torch.cuda.amp.GradScaler(enabled=opt.amp)
    # if opt.load_weights is not None and ckpt.get('optimizer', None):
        # scaler.load_state_dict(ckpt['scaler'])
        # optimizer.load_state_dict(ckpt['optimizer'])

    def _compute_warm_lr(glb_step, warmup_steps, init_lr, warmup_init_lr):
        factor = torch.pow(torch.tensor(init_lr / warmup_init_lr), torch.tensor(1.0 / warmup_steps))
        warmup_lr = warmup_init_lr * torch.pow(factor, torch.tensor(glb_step))
        return warmup_lr / opt.lr
    
    def _custom_polynomial_decay(glb_step, lr, end_lr, decay_step, power):
        global_step = min(glb_step, decay_step)
        decayed_lr = (lr - end_lr) * torch.pow(torch.tensor((1 - global_step / decay_step)), torch.tensor(power)) + end_lr
        return decayed_lr / opt.lr

    def _lr_warm_or_poly_decay(glb_step, warmup_steps, init_lr, warmup_init_lr, lr, end_lr, decay_step, power):
        if glb_step < warmup_steps:
            return _compute_warm_lr(glb_step, warmup_steps, init_lr, warmup_init_lr)
        else:
            return _custom_polynomial_decay(glb_step, lr, end_lr, decay_step, power)
    
    print(f"len(training_generator)---------{len(training_generator)}")
    partial_lr_warm_or_poly_decay = partial(_lr_warm_or_poly_decay, warmup_steps=opt.warmup_epoch * len(training_generator), init_lr=opt.lr,  warmup_init_lr=opt.warm_lr, lr=opt.lr, end_lr=opt.poly_end_lr, decay_step=opt.num_epochs * len(training_generator), power=opt.poly_decay_power)

    if opt.lr_scheduler == 'poly':
        print(f"using polynomial decay------------------")
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=partial_lr_warm_or_poly_decay, verbose=False)
    elif opt.lr_scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    else:
        print("scheduler is not support")
        sys.exit()
    

    epoch = 0
    best_loss = 1e5
    best_epoch = 0
    last_step = ckpt['step'] if opt.load_weights is not None and ckpt.get('step', None) else 0
    best_fitness = ckpt['best_fitness'] if opt.load_weights is not None and ckpt.get('best_fitness', None) else 0
    step = max(0, last_step)
    model.train()

    num_iter_per_epoch = len(training_generator)
    try:
        for epoch in range(opt.num_epochs):
            # last_epoch = step // num_iter_per_epoch
            # if epoch < last_epoch:
            #     continue

            epoch_loss = []
            progress_bar = tqdm(training_generator, ascii=True)
            for iter, data in enumerate(progress_bar):
                # if iter < step - last_epoch * num_iter_per_epoch:
                #     progress_bar.update()
                #     continue
                try:
                    imgs = data['img']
                    annot = data['annot']
                    seg_annot = data['segmentation']
                    inst_annot = data['instance']

                    if opt.num_gpus == 1:
                        # if only one gpu, just send it to cuda:0
                        imgs = imgs.to(device="cuda", memory_format=torch.channels_last)
                        annot = annot.cuda()
                        seg_annot = seg_annot.cuda()
                        inst_annot = inst_annot.cuda()
                    
                    if opt.vis_input == True:
                        imgs_vis = imgs.clone().cpu().numpy()
                        annot_vis = annot.clone().cpu().numpy()
                        seg_annot_vis = seg_annot.clone().cpu().numpy()
                        inst_annot_vis = inst_annot.clone().cpu().numpy()

                        print(f" imgs_vis : {imgs_vis.shape}, annot_vis : {annot_vis.shape}, seg_annot_vis : {seg_annot_vis.shape}, inst_annot_vis : {inst_annot_vis.shape}")

                        debug_path = "/home/wan/ZT_2T/work/git_hybridNets/HybridNets/out/debug"
                        imgs_vis_path = os.path.join(debug_path, "imgs_vis.jpg")
                        seg_annot_vis_path = os.path.join(debug_path, "seg_annot_vis.jpg")
                        inst_annot_vis_path = os.path.join(debug_path, "inst_annot_vis.jpg")
                        cv2.imwrite(imgs_vis_path, np.transpose(np.squeeze(imgs_vis), (1,2,0)))
                        cv2.imwrite(seg_annot_vis_path, np.squeeze(seg_annot_vis))
                        cv2.imwrite(inst_annot_vis_path, np.squeeze(inst_annot_vis))
                        
                        
                        sys.exit()
                        

                    optimizer.zero_grad(set_to_none=True)
                    with torch.cuda.amp.autocast(enabled=opt.amp):
                        # cls_loss, reg_loss, seg_loss, regression, classification, anchors, segmentation = model(imgs, annot, seg_annot, obj_list=params.obj_list)
                        seg_loss, inst_loss = model(imgs, annot, seg_annot, inst_annot, obj_list=params.obj_list)
                        # cls_loss = cls_loss.mean() if not opt.freeze_det else torch.tensor(0, device="cuda")
                        # reg_loss = reg_loss.mean() if not opt.freeze_det else torch.tensor(0, device="cuda")

                        seg_loss = seg_loss.mean() if not opt.freeze_seg else torch.tensor(0, device="cuda")
                        inst_loss = inst_loss.mean() if not opt.freeze_instance else torch.tensor(0, device="cuda")

                        # loss = cls_loss + reg_loss + seg_loss
                        loss = inst_loss + seg_loss
                        
                    if loss == 0 or not torch.isfinite(loss):
                        continue

                    scaler.scale(loss).backward()

                    # Don't have to clip grad norm, since our gradients didn't explode anywhere in the training phases
                    # This worsens the metrics
                    # scaler.unscale_(optimizer)
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                    
                    scaler.step(optimizer)
                    scaler.update()

                    epoch_loss.append(float(loss))

                    # progress_bar.set_description(
                    #     'Step: {}. Epoch: {}/{}. Iteration: {}/{}. Cls loss: {:.5f}. Reg loss: {:.5f}. Seg loss: {:.5f}. Total loss: {:.5f}'.format(
                    #         step, epoch, opt.num_epochs, iter + 1, num_iter_per_epoch, cls_loss.item(),
                    #         reg_loss.item(), seg_loss.item(), loss.item()))

                    progress_bar.set_description(
                        'Step: {}. Epoch: {}/{}. Iteration: {}/{}. Seg loss: {:.5f}. Inst loss: {:.5f}. Total loss: {:.5f}'.format(step, epoch, opt.num_epochs, iter + 1, num_iter_per_epoch, seg_loss.item(), inst_loss.item(), loss.item()))

                    writer.add_scalars('Loss', {'train': loss}, step)
                    # writer.add_scalars('Regression_loss', {'train': reg_loss}, step)
                    # writer.add_scalars('Classfication_loss', {'train': cls_loss}, step)
                    writer.add_scalars('Segmentation_loss', {'train': seg_loss}, step)
                    writer.add_scalars('instance_loss', {'train': inst_loss}, step)

                    # log learning_rate
                    current_lr = optimizer.param_groups[0]['lr']
                    writer.add_scalar('learning_rate', current_lr, step)

                    step += 1

                    if opt.lr_scheduler == 'poly':
                        scheduler.step()

                    if step % opt.save_interval == 0 and step > 0:
                        save_checkpoint(model, opt.saved_path, f'hybridnets-d{opt.compound_coef}_{epoch}_{step}_' + experient_name + '.pth')
                        print('checkpoint...')

                except Exception as e:
                    print('[Error]', traceback.format_exc())
                    print(e)
                    continue

                # if opt.lr_scheduler == 'poly':
                #         scheduler.step()
                # current_lr = optimizer.param_groups[0]['lr']
                # writer.add_scalar('learning_rate', current_lr, step)
                # step += 1

            
            if opt.lr_scheduler == 'plateau':
                scheduler.step(np.mean(epoch_loss))

            # if epoch % opt.val_interval == 0:
            #     best_fitness, best_loss, best_epoch = val(model, val_generator, params, opt, seg_mode, is_training=True,
            #                                               optimizer=optimizer, scaler=scaler, writer=writer, epoch=epoch, step=step, 
            #                                               best_fitness=best_fitness, best_loss=best_loss, best_epoch=best_epoch)
    except KeyboardInterrupt:
        save_checkpoint(model, opt.saved_path, f'hybridnets-d{opt.compound_coef}_{epoch}_{step}_' + experient_name + '.pth')
    finally:
        writer.close()


if __name__ == '__main__':
    opt = get_args()
    train(opt)
