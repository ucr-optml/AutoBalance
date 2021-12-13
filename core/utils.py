from torch._C import device
import torch.nn.functional as F
from utils.metrics import topk_corrects
import torch
from torch.autograd import grad
import numpy as np
from numpy.lib.scimath import log
from scipy import interpolate

def gather_flat_grad(loss_grad):
    #cnt = 0
    # for g in loss_grad:
    #    g_vector = g.contiguous().view(-1) if cnt == 0 else torch.cat([g_vector, g.contiguous().view(-1)])
    #    cnt = 1
    # g_vector
    return torch.cat([p.contiguous().view(-1) for p in loss_grad if not p is None])


def neumann_hyperstep_preconditioner(d_val_loss_d_theta, d_train_loss_d_w, elementary_lr, num_neumann_terms, model):
    preconditioner = d_val_loss_d_theta.detach()
    counter = preconditioner
    # Do the fixed point iteration to approximate the vector-inverseHessian product
    i = 0
    while i < num_neumann_terms:  # for i in range(num_neumann_terms):
        old_counter = counter
        # This increments counter to counter * (I - hessian) = counter - counter * hessian
        #gradient=grad(d_train_loss_d_w, model.parameters(), grad_outputs=counter.view(-1), retain_graph=True)
        # print(gradient)
        # print(d_train_loss_d_w)
        hessian_term = gather_flat_grad(
            grad(d_train_loss_d_w, model.parameters(), grad_outputs=counter.view(-1), retain_graph=True))
        counter = old_counter - elementary_lr * hessian_term
        preconditioner = preconditioner + counter
        i += 1
    return elementary_lr * preconditioner


def loss_adjust_cross_entropy(logits, targets, params, group_size=1):
    dy = params[0]
    ly = params[1]
    if group_size != 1:
        new_dy = dy.repeat_interleave(group_size)
        new_ly = ly.repeat_interleave(group_size)
        x = logits*F.sigmoid(new_dy)+new_ly
    else:
        x = logits*F.sigmoid(dy)+ly
    if len(params) == 3:
        wy = params[2]
        loss = F.cross_entropy(x, targets, weight=wy)
    else:
        loss = F.cross_entropy(x, targets)
    return loss

def loss_adjust_cross_entropy_cdt(logits, targets, params, group_size=1):
    dy = params[0]
    ly = params[1]
    if group_size != 1:
        new_dy = dy.repeat_interleave(group_size)
        new_ly = ly.repeat_interleave(group_size)
        x = logits*new_dy+new_ly
    else:
        x = logits*dy+ly
    if len(params) == 3:
        wy = params[2]
        loss = F.cross_entropy(x, targets, weight=wy)
    else:
        loss = F.cross_entropy(x, targets)
    return loss


def cdt_cross_entropy(logits, targets, params, group_size=1):
    dy = params[0]
    ly = params[1]
    if group_size != 1:
        new_dy = dy.repeat_interleave(group_size)
        new_ly = ly.repeat_interleave(group_size)
        x = logits*new_dy+new_ly
    else:
        x = logits*dy+ly
    if len(params) == 3:
        wy = params[2]
        loss = F.cross_entropy(x, targets, weight=wy)
    else:
        loss = F.cross_entropy(x, targets)
    return loss


def loss_adjust_dy(logits, targets, params, group_size=1):
    dy = params[0]
    ly = params[1]
    x = torch.transpose(torch.transpose(logits, 0, 1) *
                        F.sigmoid(dy[targets]), 0, 1)+ly
    loss = F.cross_entropy(x, targets)
    return loss


def cross_entropy(logits, targets, params, group_size=1):
    if len(params) == 3:
        return F.cross_entropy(logits, targets, weight=params[2])
    else:
        return F.cross_entropy(logits, targets)


def logit_adjust_ly(logits, params):
    dy = params[0]
    ly = params[1]
    x = logits*dy-ly
    return x


def get_trainable_hyper_params(params):
    return[param for param in params if param.requires_grad]


def assign_hyper_gradient(params, gradient, num_classes):
    i = 0
    for para in params:
        if para.requires_grad:
            num = para.nelement()
            grad = gradient[i:i+num].clone()
            torch.reshape(grad, para.shape)
            para.grad = grad
            i += num
            # para.grad=gradient[i:i+num].clone()
            # para.grad=gradient[i:i+num_classes].clone()
            # i+=num_classes


def get_LA_params(num_train_samples, tau, group_size, device):
    pi = num_train_samples/np.sum(num_train_samples)
    pi = tau*log(pi)
    if group_size!=1:
        pi=[pi[i] for i in range(group_size//2,len(num_train_samples),group_size)]
    print('Google pi: ', pi)
    pi = torch.tensor(pi, dtype=torch.float32, device=device)
    return pi


def get_CDT_params(num_train_samples, gamma, device):
    return torch.tensor((np.array(num_train_samples)/np.max(np.array(num_train_samples)))**gamma, dtype=torch.float32, device=device)


def get_init_dy(args, num_train_samples):
    num_classes = args["num_classes"]
    device = args["device"]
    dy_init = args["up_configs"]["dy_init"]
    group_size= args["group_size"]

    if dy_init == 'Ones':
        dy = torch.ones([((num_classes-1)//group_size)+1],
                        dtype=torch.float32, device=device)
    elif dy_init == 'CDT':
        gamma = args["up_configs"]["dy_CDT_gamma"]
        dy = get_CDT_params(num_train_samples, gamma, device=device)
    elif dy_init == 'Retrain':
        dy = args["result"]["dy"][-1]
        if num_classes//group_size!=len(dy):
            group_size=num_classes//len(dy)
            x=range(group_size//2,num_classes,group_size)
            inperp_func=interpolate.interp1d(x,dy,fill_value="extrapolate",kind="linear")
            dy=inperp_func(range(0,num_classes,1))
        dy = torch.tensor(dy, dtype=torch.float32, device=device)
    else:
        file = open(dy_init, mode='r')
        dy = file.readline().replace(
            '[', '').replace(']', '').replace('\n', '').split()
        print(dy)
        dy = np.array([float(a) for a in dy])
        dy = torch.tensor(dy, dtype=torch.float32, device=device)
    dy.requires_grad = args["up_configs"]["dy"]
    return dy


def get_init_ly(args, num_train_samples):
    num_classes = args["num_classes"]
    ly_init = args["up_configs"]["ly_init"]
    device = args["device"]
    group_size= args["group_size"]

    if ly_init == 'Zeros':
        ly = torch.zeros([((num_classes-1)//group_size)+1],
                         dtype=torch.float32, device=device)
    elif ly_init == 'LA':
        ly = get_LA_params(num_train_samples,
                           args["up_configs"]["ly_LA_tau"], args["group_size"], device)
    elif ly_init == 'Retrain':
        ly = args["result"]["ly"][-1]
        if num_classes//group_size!=len(ly):
            group_size=num_classes//len(ly)
            x=range(group_size//2,num_classes,group_size)
            inperp_func=interpolate.interp1d(x,ly,fill_value="extrapolate",kind="linear")
            ly=inperp_func(range(0,num_classes,1))
        ly = torch.tensor(ly, dtype=torch.float32, device=device)
    else:
        file = open(ly_init, mode='r')
        ly = file.readline().replace(
            '[', '').replace(']', '').replace('\n', '').split()
        ly = np.array([float(a) for a in ly])
        ly = torch.tensor(ly, dtype=torch.float32,device=device)
    ly.requires_grad = args["up_configs"]["ly"]
    return ly

def get_train_w(args, num_train_samples):
    num_classes = args["num_classes"]
    wy_init = args["up_configs"]["wy_init"]
    device = args["device"]
    group_size= args["group_size"]

    if wy_init == 'Ones':
        w_train = torch.ones([num_classes], dtype=torch.float32, device=device)
        # w_val=torch.ones([num_classes],dtype=torch.float32,device=device)
    elif wy_init == 'Pi':
        w_train = np.sum(num_train_samples)/num_train_samples
        w_train = w_train/np.linalg.norm(w_train)
        w_train = w_train/np.median(w_train)
        w_train = torch.tensor(w_train, dtype=torch.float32, device=device)
    elif wy_init == 'Retrain':
        w_train = args["result"]["w_train"][-1]
        if num_classes//group_size!=len(w_train):
            group_size=num_classes//len(w_train)
            x=range(group_size//2,num_classes,group_size)
            inperp_func=interpolate.interp1d(x,w_train,fill_value="extrapolate",kind="linear")
            w_train=inperp_func(range(0,num_classes,1))
        w_train = torch.tensor(w_train, dtype=torch.float32, device=device)
    w_train.requires_grad = args["up_configs"]["wy"]
    return w_train

def get_val_w(args, num_val_samples):
    device = args["device"]
    num_classes = args["num_classes"]
    if args["balance_val"]:
        w_val = torch.ones([num_classes], dtype=torch.float32, device=device)
    else:
        w_val=np.sum(num_val_samples)/num_val_samples
        w_val=w_val/np.linalg.norm(w_val)
    w_val=torch.tensor(w_val,dtype=torch.float32, device=device)
    w_val.requires_grad=False
    return w_val
