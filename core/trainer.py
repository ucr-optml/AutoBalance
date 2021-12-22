import torch.nn.functional as F
from utils.metrics import topk_corrects
import torch
from torch.autograd import grad
import numpy as np

from core.utils import gather_flat_grad,neumann_hyperstep_preconditioner
from core.utils import get_trainable_hyper_params,assign_hyper_gradient

def train_epoch(
    cur_epoch, model, args,
    low_loader, low_criterion , low_optimizer, low_params=None,
    up_loader=None, up_optimizer=None, up_criterion=None, up_params=None,
    ):
    """Performs one epoch of bilevel optimization."""
    # Enable training mode
    num_classes=args["num_classes"]
    group_size=args["group_size"]
    ARCH_EPOCH=args["up_configs"]["start_epoch"]
    ARCH_END=args["up_configs"]["end_epoch"]
    ARCH_EPOCH_INTERVAL=args["up_configs"]["epoch_interval"]
    ARCH_INTERVAL=args["up_configs"]["iter_interval"]
    ARCH_TRAIN_SAMPLE=args["up_configs"]["train_batches"]
    ARCH_VAL_SAMPLE=args["up_configs"]["val_batches"]
    device=args["device"]
    is_up=(cur_epoch >= ARCH_EPOCH) and (cur_epoch <= ARCH_END) and \
        ((cur_epoch+1) % ARCH_EPOCH_INTERVAL) == 0
    
    if is_up:
        print('lower lr: ',low_optimizer.param_groups[0]['lr'],'  upper lr: ',up_optimizer.param_groups[0]['lr'])
        up_iter = iter(up_loader)
        low_iter_alt=iter(low_loader)
    else:
        print('lr: ',low_optimizer.param_groups[0]['lr'])
    
    model.train()
    total_correct=0.
    total_sample=0.
    total_loss=0.
    num_weights, num_hypers = sum(p.numel() for p in model.parameters()), 2*((num_classes-1)//group_size)+1
    use_reg=True

    d_train_loss_d_w = torch.zeros(num_weights,device=device)

    for cur_iter, (low_data, low_targets) in enumerate(low_loader):
        #print(cur_iter)
        # Transfer the data to the current GPU device
        # if cur_iter%5==0:
        #     print(cur_iter,len(low_loader))
        low_data, low_targets = low_data.to(device=device, non_blocking=True), low_targets.to(device=device, non_blocking=True)
        # Update architecture
        if is_up:
            model.train()
            up_optimizer.zero_grad()
            if cur_iter%ARCH_INTERVAL==0:
                for _ in range(ARCH_TRAIN_SAMPLE):
                    try:
                        low_data_alt, low_targets_alt = next(low_iter_alt)
                    except StopIteration:
                        low_iter_alt = iter(low_loader)
                        low_data_alt, low_targets_alt = next(low_iter_alt) 
                    low_data_alt, low_targets_alt = low_data_alt.to(device=device, non_blocking=True), low_targets_alt.to(device=device, non_blocking=True)
                    low_optimizer.zero_grad()
                    low_preds=model(low_data_alt)
                    low_loss=low_criterion(low_preds,low_targets_alt,low_params,group_size=group_size) 
                    d_train_loss_d_w+=gather_flat_grad(grad(low_loss,model.parameters(),create_graph=True))
                    #print(cur_iter_alt)
                d_train_loss_d_w/=ARCH_TRAIN_SAMPLE
                d_val_loss_d_theta = torch.zeros(num_weights,device=device)
                #d_val_loss_d_theta, direct_grad = torch.zeros(num_weights).cuda(), torch.zeros(num_hypers).cuda()

                for _ in range(ARCH_VAL_SAMPLE):
                    try:
                        up_data, up_targets = next(up_iter)
                    except StopIteration:
                        up_iter = iter(up_loader)
                        up_data, up_targets = next(up_iter) 
                #for _,(up_data,up_targets) in enumerate(up_loader):
                    up_data, up_targets = up_data.to(device=device, non_blocking=True), up_targets.to(device=device, non_blocking=True)
                    model.zero_grad()
                    low_optimizer.zero_grad()
                    up_preds = model(up_data)
                    up_loss = up_criterion(up_preds,up_targets,up_params,group_size=group_size)
                    d_val_loss_d_theta += gather_flat_grad(grad(up_loss, model.parameters(), retain_graph=use_reg))
                    # if use_reg:
                    #     direct_grad+=gather_flat_grad(grad(up_loss, get_trainable_hyper_params(up_params), allow_unused=True))
                    #     direct_grad[direct_grad != direct_grad] = 0
                d_val_loss_d_theta/=ARCH_VAL_SAMPLE
                #direct_grad/=ARCH_VAL_SAMPLE
                preconditioner = d_val_loss_d_theta
                
                preconditioner = neumann_hyperstep_preconditioner(d_val_loss_d_theta, d_train_loss_d_w, 1.0,
                                                                5, model)
                indirect_grad = gather_flat_grad(
                    grad(d_train_loss_d_w, get_trainable_hyper_params(up_params), grad_outputs=preconditioner.view(-1),allow_unused=True))
                hyper_grad=-indirect_grad#+direct_grad
                up_optimizer.zero_grad()
                assign_hyper_gradient(up_params,hyper_grad,num_classes)
                up_optimizer.step()
                d_train_loss_d_w = torch.zeros(num_weights,device=device)
        
        if is_up:
            try:
                up_data, up_targets = next(up_iter)
            except StopIteration:
                up_iter = iter(up_loader)
                up_data, up_targets = next(up_iter) 
            up_data, up_targets = up_data.to(device=device, non_blocking=True), up_targets.to(device=device, non_blocking=True)
            up_preds=model(up_data)
            up_loss=up_criterion(up_preds,up_targets,up_params,group_size=group_size)
            up_optimizer.zero_grad()
            up_loss.backward()
            up_optimizer.step()


        # Perform the forward pass
        low_preds = model(low_data)

        # Compute the loss
        loss = low_criterion(low_preds, low_targets, low_params,group_size=group_size)
        # Perform the backward pass
        low_optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm(model.parameters(), 5.0)
        low_optimizer.step()

        # Compute the errors
        mb_size = low_data.size(0)
        ks = [1] 
        top1_correct = topk_corrects(low_preds, low_targets, ks)[0]
        
        # Copy the stats from GPU to CPU (sync point)
        loss = loss.item()
        top1_correct = top1_correct.item()
        total_correct+=top1_correct
        total_sample+=mb_size
        total_loss+=loss*mb_size
    # Log epoch stats
    print(f'Epoch {cur_epoch} :  Loss = {total_loss/total_sample}   ACC = {total_correct/total_sample*100.}')



# @torch.no_grad()
# def eval_epoch(data_loader, model, criterion, cur_epoch, text, args, params=None):
#     num_classes=args["num_classes"]
#     group_size=args["group_size"]
#     model.eval()
#     correct=0.
#     total=0.
#     loss=0.
#     class_correct=np.zeros(num_classes,dtype=float)
#     class_total=np.zeros(num_classes,dtype=float)
#     confusion_matrix = torch.zeros(num_classes, num_classes).cuda()
#     for cur_iter, (data, targets) in enumerate(data_loader):
#         if cur_iter%5==0:
#             print(cur_iter,len(data_loader))
#         data, targets = data.cuda(), targets.cuda(non_blocking=True)
#         logits = model(data)
         
#         preds = logits.data.max(1)[1]
#         mb_size = data.size(0)
#         loss+=criterion(logits,targets,params,group_size=group_size).item()*mb_size

#         total+=mb_size
#         correct+=preds.eq(targets.data.view_as(preds)).sum().item()


        
#         for i in range(num_classes):
#             indexes=np.where(targets.cpu().numpy()==i)[0]
#             class_total[i]+=indexes.size
#             class_correct[i]+=preds[indexes].eq(targets[indexes].data.view_as(preds[indexes])).sum().item()

#     text=f'{text}: Epoch {cur_epoch} :  Loss = {loss/total}   ACC = {correct/total*100.} Balanced ACC = {np.mean(class_correct/class_total*100.)} \n Class wise = {class_correct/class_total*100.}'
#     print(text)
#     return text, loss/total, 100.-correct/total*100., 100.-float(np.mean(class_correct/class_total*100.))

@torch.no_grad()
def eval_epoch(data_loader, model, criterion, cur_epoch, text, args, params=None):
    num_classes=args["num_classes"]
    group_size=args["group_size"]
    model.eval()
    correct=0.
    total=0.
    loss=0.
    class_correct=np.zeros(num_classes,dtype=float)
    class_total=np.zeros(num_classes,dtype=float)
    confusion_matrix = torch.zeros(num_classes, num_classes).cuda()
    for cur_iter, (data, targets) in enumerate(data_loader):
        # if cur_iter%5==0:
        #     print(cur_iter,len(data_loader))
        data, targets = data.cuda(), targets.cuda(non_blocking=True)
        logits = model(data)
         
        preds = logits.data.max(1)[1]
        for t, p in zip(targets.view(-1), preds.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1
        mb_size = data.size(0)
        loss+=criterion(logits,targets,params,group_size=group_size).item()*mb_size
    class_correct=confusion_matrix.diag().cpu().numpy()
    class_total=confusion_matrix.sum(1).cpu().numpy()
    total=confusion_matrix.sum().cpu().numpy()
    correct=class_correct.sum()

    text=f'{text}: Epoch {cur_epoch} :  Loss = {loss/total}   ACC = {correct/total*100.} Balanced ACC = {np.mean(class_correct/class_total*100.)} \n Class wise = {class_correct/class_total*100.}'
    print(text)
    return text, loss/total, 100.-correct/total*100., 100.-float(np.mean(class_correct/class_total*100.)), (100.-class_correct/class_total*100.).tolist()

