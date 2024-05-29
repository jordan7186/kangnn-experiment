import random
import torch
import torch.nn as nn
import numpy as np

from torch_geometric.datasets import Planetoid, WebKB
import torch_geometric.transforms as T
from torch_geometric.nn import GCN, GAT, GraphSAGE, GIN, MLP
from utils import random_disassortative_splits
from models import KANGNN, KANonly

# Log training time
import time

seed = 0

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_KANGNN(feat, adj, label, mask, model, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(feat, adj)
    pred, true = out[mask], label[mask]
    loss = criterion(pred, true)
    acc = int((pred.argmax(dim=-1) == true).sum()) / int(mask.sum())
    loss.backward()
    optimizer.step()
    return acc, loss.item()


@torch.no_grad()
def eval_KANGNN(feat, adj, model):
    model.eval()
    with torch.no_grad():
        pred = model(feat, adj)
    pred = pred.argmax(dim=-1)
    return pred


def train_KAN(feat, label, mask, model, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(feat)
    pred, true = out[mask], label[mask]
    loss = criterion(pred, true)
    acc = int((pred.argmax(dim=-1) == true).sum()) / int(mask.sum())
    loss.backward()
    optimizer.step()
    return acc, loss.item()


@torch.no_grad()
def eval_KAN(feat, model):
    model.eval()
    with torch.no_grad():
        pred = model(feat)
    pred = pred.argmax(dim=-1)
    return pred


def train(feat, edge_index, label, mask, model, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(feat, edge_index)
    pred, true = out[mask], label[mask]
    loss = criterion(pred, true)
    acc = int((pred.argmax(dim=-1) == true).sum()) / int(mask.sum())
    loss.backward()
    optimizer.step()
    return acc, loss.item()


@torch.no_grad()
def eval(feat, edge_index, model):
    model.eval()
    pred = model(feat, edge_index)
    pred = pred.argmax(dim=-1)
    return pred


def train_MLP(mlp, mlp_optimizer, feat, label, mask, criterion):
    mlp.train()
    mlp_optimizer.zero_grad()
    out = mlp(feat)
    pred, true = out[mask], label[mask]
    loss = criterion(pred, true)
    acc = int((pred.argmax(dim=-1) == true).sum()) / int(mask.sum())
    loss.backward()
    mlp_optimizer.step()
    return acc, loss.item()


@torch.no_grad()
def eval_MLP(feat, mlp):
    mlp.eval()
    pred = mlp(feat).argmax(dim=-1)
    return pred


# For GNN experiments
def run_GNN_experiment(
    dataset_name: str,
    model_type: str,
    hidden_size: int,
    n_layers: int,
    lr: float,
    epochs: int,
    patience: int = 100,
    device="cuda",
):
    path = "./data/"
    transform = T.Compose([T.NormalizeFeatures(), T.GCNNorm(), T.ToSparseTensor()])

    if dataset_name in {"Cora", "Citeseer", "Pubmed"}:
        dataset = Planetoid(path, dataset_name, transform=transform)[0]
    elif dataset_name in {"Cornell", "Texas", "Wisconsin", "Washington"}:
        dataset = WebKB(path, dataset_name, transform=transform)[0]

    in_feat = dataset.num_features
    out_feat = max(dataset.y) + 1

    if model_type == "GCN":
        model = GCN(in_feat, hidden_size, n_layers).to(device)
    elif model_type == "GAT":
        model = GAT(in_feat, hidden_size, n_layers).to(device)
    elif model_type == "GraphSAGE":
        model = GraphSAGE(in_feat, hidden_size, n_layers).to(device)
    elif model_type == "GIN":
        model = GIN(in_feat, hidden_size, n_layers).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    adj = dataset.adj_t.to(device)
    feat = dataset.x.float().to(device)
    label = dataset.y.to(device)

    trn_mask, val_mask, tst_mask = random_disassortative_splits(label, out_feat)
    trn_mask, val_mask, tst_mask = (
        trn_mask.to(device),
        val_mask.to(device),
        tst_mask.to(device),
    )

    num_params = sum(p.numel() for p in model.parameters())
    # Early stopping
    best_val_acc = 0.0
    best_tst_acc = 0.0
    patience_cnt = 0
    # Log training time
    start_time = time.time()
    # Log best epoch
    best_epoch = 0
    for i in range(epochs):
        train(feat, adj, label, trn_mask, model, optimizer, criterion)
        pred = eval(feat, adj, model)
        val_acc = int((pred[val_mask] == label[val_mask]).sum()) / int(val_mask.sum())
        tst_acc = int((pred[tst_mask] == label[tst_mask]).sum()) / int(tst_mask.sum())

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_tst_acc = tst_acc
            best_epoch = i
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt == patience:
                break

    return best_val_acc, best_tst_acc, num_params, time.time() - start_time, best_epoch


# For KAN + GNN experiments
def run_KANGNN_experiment(
    dataset_name: str,
    hidden_size: int,
    kan_layers: int,
    mp_layers: int,
    input_embed: bool,
    lr: float,
    epochs: int,
    patience: int = 100,
    device="cuda",
):
    assert kan_layers in {1, 2}
    assert mp_layers in {1, 2, 3}

    path = "./data/"
    transform = T.Compose([T.NormalizeFeatures(), T.GCNNorm(), T.ToSparseTensor()])

    if dataset_name in {"Cora", "Citeseer", "Pubmed"}:
        dataset = Planetoid(path, dataset_name, transform=transform)[0]
    elif dataset_name in {"Cornell", "Texas", "Wisconsin", "Washington"}:
        dataset = WebKB(path, dataset_name, transform=transform)[0]

    in_feat = dataset.num_features
    out_feat = max(dataset.y) + 1

    model = KANGNN(
        in_feat=in_feat,
        hidden_feat=hidden_size,
        out_feat=out_feat,
        use_bias=False,
        kan_layers=kan_layers,
        mp_layers=mp_layers,
        input_embed=input_embed,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    adj = dataset.adj_t.to(device)
    feat = dataset.x.float().to(device)
    label = dataset.y.to(device)

    trn_mask, val_mask, tst_mask = random_disassortative_splits(label, out_feat)
    trn_mask, val_mask, tst_mask = (
        trn_mask.to(device),
        val_mask.to(device),
        tst_mask.to(device),
    )

    num_params = sum(p.numel() for p in model.parameters())

    # Early stopping
    best_val_acc = 0.0
    best_tst_acc = 0.0
    # Log training time
    start_time = time.time()
    # Log best epoch
    best_epoch = 0
    patience_cnt = 0
    for i in range(epochs):
        train_KANGNN(feat, adj, label, trn_mask, model, optimizer, criterion)
        pred = eval_KANGNN(feat, adj, model)
        val_acc = int((pred[val_mask] == label[val_mask]).sum()) / int(val_mask.sum())
        tst_acc = int((pred[tst_mask] == label[tst_mask]).sum()) / int(tst_mask.sum())

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_tst_acc = tst_acc
            best_epoch = i
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt == patience:
                break

    return best_val_acc, best_tst_acc, num_params, time.time() - start_time, best_epoch


# For KAN + GNN experiments
def run_KAN_experiment(
    dataset_name: str,
    hidden_size: int,
    kan_layers: int,
    input_embed: bool,
    lr: float,
    epochs: int,
    patience: int = 100,
    device="cuda",
):
    assert kan_layers in {1, 2}

    path = "./data/"
    transform = T.Compose([T.NormalizeFeatures(), T.GCNNorm(), T.ToSparseTensor()])

    if dataset_name in {"Cora", "Citeseer", "Pubmed"}:
        dataset = Planetoid(path, dataset_name, transform=transform)[0]
    elif dataset_name in {"Cornell", "Texas", "Wisconsin", "Washington"}:
        dataset = WebKB(path, dataset_name, transform=transform)[0]

    in_feat = dataset.num_features
    out_feat = max(dataset.y) + 1

    model = KANonly(
        in_feat=in_feat,
        hidden_feat=hidden_size,
        out_feat=out_feat,
        use_bias=False,
        kan_layers=kan_layers,
        input_embed=input_embed,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    feat = dataset.x.float().to(device)
    label = dataset.y.to(device)

    trn_mask, val_mask, tst_mask = random_disassortative_splits(label, out_feat)
    trn_mask, val_mask, tst_mask = (
        trn_mask.to(device),
        val_mask.to(device),
        tst_mask.to(device),
    )

    num_params = sum(p.numel() for p in model.parameters())

    # Early stopping
    best_val_acc = 0.0
    best_tst_acc = 0.0
    # Log training time
    start_time = time.time()
    # Log best epoch
    best_epoch = 0
    patience_cnt = 0
    for i in range(epochs):
        train_KAN(feat, label, trn_mask, model, optimizer, criterion)
        pred = eval_KAN(feat, model)
        val_acc = int((pred[val_mask] == label[val_mask]).sum()) / int(val_mask.sum())
        tst_acc = int((pred[tst_mask] == label[tst_mask]).sum()) / int(tst_mask.sum())

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_tst_acc = tst_acc
            best_epoch = i
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt == patience:
                break

    return best_val_acc, best_tst_acc, num_params, time.time() - start_time, best_epoch


# For MLP experiments
def run_MLP_experiment(
    dataset_name: str,
    hidden_size: int,
    n_layers: int,
    lr: float,
    epochs: int,
    patience: int = 100,
    device="cuda",
):
    path = "./data/"
    transform = T.Compose([T.NormalizeFeatures(), T.GCNNorm(), T.ToSparseTensor()])

    if dataset_name in {"Cora", "Citeseer", "Pubmed"}:
        dataset = Planetoid(path, dataset_name, transform=transform)[0]
    elif dataset_name in {"Cornell", "Texas", "Wisconsin", "Washington"}:
        dataset = WebKB(path, dataset_name, transform=transform)[0]

    in_feat = dataset.num_features
    out_feat = max(dataset.y) + 1

    model = MLP(
        in_channels=in_feat,
        hidden_channels=hidden_size,
        out_channels=out_feat,
        num_layers=n_layers,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    feat = dataset.x.float().to(device)
    label = dataset.y.to(device)

    trn_mask, val_mask, tst_mask = random_disassortative_splits(label, out_feat)
    trn_mask, val_mask, tst_mask = (
        trn_mask.to(device),
        val_mask.to(device),
        tst_mask.to(device),
    )

    num_params = sum(p.numel() for p in model.parameters())
    # Early stopping
    best_val_acc = 0.0
    best_tst_acc = 0.0
    patience_cnt = 0
    # Log training time
    start_time = time.time()
    # Log best epoch
    best_epoch = 0
    for i in range(epochs):
        train_MLP(
            mlp=model,
            mlp_optimizer=optimizer,
            feat=feat,
            label=label,
            mask=trn_mask,
            criterion=criterion,
        )
        pred = eval_MLP(feat=feat, mlp=model)
        val_acc = int((pred[val_mask] == label[val_mask]).sum()) / int(val_mask.sum())
        tst_acc = int((pred[tst_mask] == label[tst_mask]).sum()) / int(tst_mask.sum())

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_tst_acc = tst_acc
            best_epoch = i
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt == patience:
                break

    return best_val_acc, best_tst_acc, num_params, time.time() - start_time, best_epoch


# import wandb
from itertools import product

if __name__ == "__main__":
    dataset_list = [
        "Cora",
        "Citeseer",
        "Pubmed",
        "Cornell",
        "Texas",
        "Wisconsin",
    ]

    for dataset_name in dataset_list:
        # GNN experiments first
        model_name_list = ["GCN", "GAT", "GraphSAGE", "GIN"]
        hidden_size_list = [16, 32, 64]
        n_layers_list = [1, 2, 3]
        lr_list = [0.1, 0.01, 0.001, 0.0001]
        epochs = 300

        for model_name, hidden_size, n_layers, lr in product(
            model_name_list, hidden_size_list, n_layers_list, lr_list
        ):
            val_acc, tst_acc, num_params, train_time, best_epoch = run_GNN_experiment(
                dataset_name,
                model_name,
                hidden_size,
                n_layers,
                lr,
                epochs,
                device=device,
            )
            # Log locally to a .txt file
            with open("results_GNN.txt", "a") as f:
                # Config
                f.write(
                    f"Dataset name: {dataset_name}, Model name: {model_name}, Hidden size: {hidden_size}, Num. layers: {n_layers}, lr: {lr}\n"
                )
                # Results
                f.write(
                    f"Validation accuracy: {val_acc:.6}, Test accuracy: {tst_acc:.6}, Num. parameters: {num_params}, Train time: {train_time:.2f}, Best epoch: {best_epoch}\n"
                )

        # KAN + GNN experiments next
        hidden_size_list = [16, 32, 64]
        kan_layers_list = [1, 2]
        mp_layers_list = [1, 2, 3]
        input_embed_list = [True, False]
        lr_list = [0.1, 0.01, 0.001, 0.0001]
        epochs = 1000

        for hidden_size, kan_layers, mp_layers, input_embed, lr in product(
            hidden_size_list, kan_layers_list, mp_layers_list, input_embed_list, lr_list
        ):
            val_acc, tst_acc, num_params, train_time, best_epoch = (
                run_KANGNN_experiment(
                    dataset_name,
                    hidden_size,
                    kan_layers,
                    mp_layers,
                    input_embed,
                    lr,
                    epochs,
                    device=device,
                )
            )
            # Log locally to a .txt file
            with open("results_KANGNN.txt", "a") as f:
                # Config
                f.write(
                    f"Dataset name: {dataset_name}, Model name: KANGNN, Hidden size: {hidden_size}, Num. KAN layers: {kan_layers}, MP layers: {mp_layers}, Input embed: {input_embed}, lr: {lr}\n"
                )
                # Results
                f.write(
                    f"Validation accuracy: {val_acc:.6}, Test accuracy: {tst_acc:.6}, Num. parameters: {num_params}, Train time: {train_time:.2f}, Best epoch: {best_epoch}\n"
                )

        # KAN experiments last
        hidden_size_list = [16, 32, 64]
        kan_layers_list = [1, 2]
        input_embed_list = [True, False]
        lr_list = [0.1, 0.01, 0.001, 0.0001]
        epochs = 1000

        for hidden_size, kan_layers, input_embed, lr in product(
            hidden_size_list, kan_layers_list, input_embed_list, lr_list
        ):
            val_acc, tst_acc, num_params, train_time, best_epoch = run_KAN_experiment(
                dataset_name,
                hidden_size,
                kan_layers,
                input_embed,
                lr,
                epochs,
                device=device,
            )
            # Log locally to a .txt file
            with open("results.txt", "a") as f:
                # Config
                f.write(
                    f"Dataset name: {dataset_name}, Model name: KAN, Hidden size: {hidden_size}, Num. KAN layers: {kan_layers}, Input embed: {input_embed}, lr: {lr}\n"
                )
                # Results
                f.write(
                    f"Validation accuracy: {val_acc:.6}, Test accuracy: {tst_acc:.6}, Num. parameters: {num_params}, Train time: {train_time:.2f}, Best epoch: {best_epoch}\n"
                )

        # MLP experiments
        hidden_size_list = [16, 32, 64]
        n_layers_list = [1, 2, 3]
        lr_list = [0.1, 0.01, 0.001, 0.0001]
        epochs = 1000

        for hidden_size, n_layers, lr in product(
            hidden_size_list, n_layers_list, lr_list
        ):
            val_acc, tst_acc, num_params, train_time, best_epoch = run_MLP_experiment(
                dataset_name,
                hidden_size,
                n_layers,
                lr,
                epochs,
                device=device,
            )

            # Log locally to a .txt file
            with open("results_MLP.txt", "a") as f:
                # Config
                f.write(
                    f"Dataset name: {dataset_name}, Model name: MLP, Hidden size: {hidden_size}, Num. layers: {n_layers}, lr: {lr}\n"
                )
                # Results
                f.write(
                    f"Validation accuracy: {val_acc:.6}, Test accuracy: {tst_acc:.6}, Num. parameters: {num_params}, Train time: {train_time:.2f}, Best epoch: {best_epoch}\n"
                )
