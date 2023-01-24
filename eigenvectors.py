import os
from argparse import ArgumentParser
from time import sleep, strftime
import json

import cupy as cp
import numpy as np
import torch
from scipy.sparse.linalg import LinearOperator, eigsh, lobpcg
from torch import autograd, nn
from torch.nn import functional as F
from tqdm import tqdm

from dinopl import utils as U
from losslandscape import ParamProjector, load_data, load_model

device = torch.device(U.pick_single_gpu())

class HVP(LinearOperator):
    def __init__(self, student:nn.Module, teacher:nn.Module, loss_func, dataloader, device):

        self.student = student
        self.teacher = teacher
        self.loss_func = loss_func
        self.dataloader = dataloader
        self.device = device

        vec = U.module_to_vector(teacher).cpu().numpy()
        super().__init__(dtype=vec.dtype, shape=(len(vec), len(vec)))
        self.student.requires_grad_(True)

    @torch.no_grad()
    def hvp(self, mat:torch.Tensor) -> torch.Tensor:
        # view vec as parameter list without copy
        parammat = [U.vector_as_params(vec.squeeze(), self.teacher) for vec in mat.T] 

        # store training mode and switch to eval
        s_mode, t_mode = self.student.training, self.teacher.training
        self.student.eval(), self.teacher.eval()

        numel = 0
        outmat = torch.zeros_like(mat)
        for inputs, _ in tqdm(self.dataloader, postfix=f'k={len(parammat)}'):
            numel += inputs.shape[0]
            inputs = inputs.to(self.device) if self.device is not None else inputs

            if self.teacher is not self.student:
                teacher_out = self.teacher(inputs)

            with torch.enable_grad():
                student_out = self.student(inputs)
                if self.teacher is self.student:
                    teacher_out = student_out.detach()

                loss = self.loss_func(student_out, teacher_out)
                grads = autograd.grad(loss, self.student.parameters(), create_graph=True)

            for col, paramvec in enumerate(parammat):
                Hv = autograd.grad(grads, self.student.parameters(), paramvec, retain_graph=True)
                outmat[:, col] += torch.cat([Hv_i.reshape(-1) for Hv_i in Hv])

        # restore previous mode
        self.student.train(t_mode), self.teacher.train(s_mode)
        return outmat.div_(numel)

    def _matmat(self, mat):
        if isinstance(mat, torch.Tensor):
            input_dev = mat.device
            return self.hvp(mat.to(self.device)).to(input_dev)
        if isinstance(mat, np.ndarray):
            mat = torch.from_numpy(mat).to(device=self.device)
            return self.hvp(mat).cpu().numpy()
        if isinstance(mat, cp.ndarray):
            mat = torch.as_tensor(mat, device=self.device) # view without copy if on same device
            return cp.asarray(self.hvp(mat)) # view without copy if on same device


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('ckpt', type=str)
    parser.add_argument('--k', type=int, default=2)
    parser.add_argument('--runname', type=str, default=strftime('%Y-%m-%d--%H-%M-%S'))

    args = vars(parser.parse_args())
    
    # Prepare directories for logging and storing
    while True:
        args['dir'] = os.path.join(os.environ['DINO_RESULTS'], 'eigenvalues', args['runname'])
        try:
            os.makedirs(args['dir'], exist_ok=False)
        except OSError:
            sleep(1)
            args['runname'] = strftime('%Y-%m-%d--%H-%M-%S')
            continue
        break

    # Store args to directory
    with open(os.path.join(args['dir'], 'args.json'), 'w') as f:
        s = json.dumps(args, indent=2)
        f.write(s)


    model = load_model(args['ckpt']).to(device=device)
    train_dl, _ = load_data(args['ckpt'], 128, 4, True)

    model.return_dict = False
    #def loss_func(student_out, teacher_out):
    #    log_preds= F.log_softmax(student_out['logits'], dim=-1)
    #    targs = F.softmax(teacher_out['logits'], dim=-1)
    #    return U.cross_entropy(log_preds, targs).sum()
    def loss_func(predictions, targets):
        log_preds = F.log_softmax(predictions, dim=-1)
        targs = F.softmax(targets, dim=-1)
        return U.cross_entropy(log_preds, targs).sum()
        

    hessian = HVP(model, model, loss_func, train_dl, device=device)

    n = sum([p.numel() for p in model.parameters()])
    X = np.random.randn(n, args['k'])
    tol = np.finfo(X.dtype).eps * 2
    eigvals, eigvecs, lambdas, rnorms = lobpcg(hessian, X, tol=tol, maxiter=20, largest=True,
                                                retLambdaHistory=True, retResidualNormsHistory=True, verbosityLevel=0)

    torch.save(torch.as_tensor(eigvals).cpu(), os.path.join(args['dir'], 'eigvals.pt'))
    torch.save(torch.as_tensor(eigvecs).cpu(), os.path.join(args['dir'], 'eigvecs.pt'))
    torch.save(torch.as_tensor(lambdas).cpu(), os.path.join(args['dir'], 'lambdas.pt'))
    torch.save(torch.as_tensor(rnorms).cpu(), os.path.join(args['dir'], 'rnorms.pt'))
