from model import dataset
import torch
from torch.autograd import grad
import numpy as np
from model.model import VanillaPDETransformer
from collections import defaultdict
from bcics.boundary_conditions import DirichletBC
from bcics.initial_conditions import IC

def gen_testdata():
    data = np.load("./data/Burgers.npz")
    t, x, exact = data["t"], data["x"], data["usol"].T
    xx, tt = np.meshgrid(x, t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    y = exact.flatten()[:, None]
    return X, y


def pde(x, y):
    dy_x = grad(y, x[:, 0], grad_outputs=torch.ones_like(y), retain_graph=True, create_graph=True)
    dy_t = grad(y, x[:, -1], grad_outputs=torch.ones_like(y), retain_graph=True, create_graph=True)
    dy_xx = grad(dy_x, x[:, 0], grad_outputs=torch.ones_like(y), retain_graph=True, create_graph=True)
    return dy_t + y * dy_x - 0.01 / np.pi * dy_xx


geom = [(-1, 1), (0, 0.99)]

bc_1 = DirichletBC(geom, boundary_dim=0, boundary_point=-1, time_dim=True, func=lambda x: 0)
bc_2 = DirichletBC(geom, boundary_dim=0, boundary_point=1, time_dim=True, func=lambda x: 0)
ic = IC(geom, lambda x: -torch.sin(np.pi * x[:, 0]))



config = defaultdict(lambda: None,
                            num_feats=2,
                            pos_dim=2,
                            n_targets=1,
                            n_hidden=128,
                            num_feat_layers=2,
                            num_encoder_layers=4,
                            n_head=8,
                            # pred_len=0,
                            dim_feedforward=256,
                            attention_type='fourier_zero_sum',  # no softmax
                            xavier_init=1e-4,
                            diagonal_weight=1e-2,
                            symmetric_init=False,
                            layer_norm=True,
                            attn_norm=False,
                            batch_norm=True,
                            spacial_residual=False,
                            return_attn_weight=False,
                            seq_len=None,
                            activation='silu',
                            decoder_type='pointwise',
                            # freq_dim=64,
                            num_regressor_layers=2,
                            # fourier_modes=16,
                            spacial_dim=2,
                            spacial_fc=False,
                            dropout=0.,)

net = VanillaPDETransformer(**config)

model = dde.Model(data, net)

model.compile("adam", lr=1e-3)
losshistory, train_state = model.train(iterations=20000, display_every=10)
model.compile("L-BFGS")
losshistory, train_state = model.train(iterations=1000, display_every=100)
dde.saveplot(losshistory, train_state, issave=True, isplot=True)

X, y_true = gen_testdata()
y_pred = model.predict(X)
f = model.predict(X, operator=pde)
print("Mean residual:", np.mean(np.absolute(f)))
print("L2 relative error:", dde.metrics.l2_relative_error(y_true, y_pred))
np.savetxt("test.dat", np.hstack((X, y_true, y_pred)))