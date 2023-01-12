import os

os.environ["DINO_DATA"] = "datasets"
os.environ["DINO_RESULTS"] = "results"

import torch
import shutil
from losslandscape import load_model
import argparse
from finetuning_utils.utils import get_dataloaders, get_optimizer
from tqdm import tqdm


class FullModel(torch.nn.Module):
    def __init__(self, hid_dim=512, out_size=10):
        super().__init__()
        # dummy placeholder
        self.model = load_model(
            "saves/22abl14o/epoch=58-step=11564-probe_student=0.524.ckpt:teacher.init"
        ).enc
        self.head = torch.nn.Linear(hid_dim, out_size)

    def forward(self, x):
        return self.head(self.model(x))


def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj][0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--aug", type=str, default="none")
    parser.add_argument("--optimizer", type=str, default="sgd")
    parser.add_argument("--dataset", type=str, default="cifar-10")
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--log_every", type=int, default=10)

    args = parser.parse_args()

    random_model = load_model(
        "saves/22abl14o/epoch=58-step=11564-probe_student=0.524.ckpt:teacher.init"
    ).enc
    student_alpha_0_best = load_model(
        "saves/22abl14o/epoch=58-step=11564-probe_student=0.524.ckpt:student"
    ).enc
    student_alpha_0_last = load_model("saves/22abl14o/last.ckpt:student").enc
    student_alpha_1_best = load_model(
        "saves/3mtlpc13/epoch=96-step=19012-probe_student=0.446.ckpt:student"
    ).enc
    student_alpha_1_last = load_model("saves/3mtlpc13/last.ckpt:student").enc

    num_classes = {
        "cifar-10": 10,
        "cifar-100": 100,
        "stl-10": 10,
        "tiny-imagenet": 200,
    }

    loss_fn = torch.nn.CrossEntropyLoss()

    models = {
        namestr(x, globals()): x
        for x in [
            random_model,
            student_alpha_0_best,
            student_alpha_0_last,
            student_alpha_1_best,
            student_alpha_1_last,
        ]
    }

    for model_name, model in models.items():

        name = f"aug_{args.aug}_optimizer_{args.optimizer}_dataset_{args.dataset}_model_name_{model_name}"

        shutil.rmtree(f"results/{name}", ignore_errors=True)

        if not os.path.exists(f"results/{name}"):
            os.makedirs(f"results/{name}")

        full_model = FullModel(out_size=num_classes[args.dataset])
        full_model.model.load_state_dict(model.state_dict())
        full_model.cuda()

        results = {
            "inner_train_loss": [],
            "inner_train_acc": [],
            "train_loss": [],
            "train_acc": [],
            "test_loss": [],
            "test_acc": [],
        }

        train_loader, test_loader = get_dataloaders(args.dataset, args.aug)

        opt = get_optimizer(full_model, args.optimizer)
        step = -1

        for epoch in tqdm(range(args.n_epochs), desc=name + "_" + model_name):
            total_loss = 0
            total_acc = 0
            total_cnt = 0

            full_model.train()
            for batch in train_loader:
                step += 1
                x, y = batch[0].cuda(), batch[1].cuda()

                pred = full_model(x)

                loss = loss_fn(pred, y)

                opt.zero_grad()
                loss.backward()
                opt.step()

                total_loss += loss.item() * x.shape[0]
                total_acc += (torch.argmax(pred, dim=-1) == y).float().sum().item()
                total_cnt += x.shape[0]

                if step % args.log_every == 0:
                    results["inner_train_loss"].append(loss.item())
                    results["inner_train_acc"].append(
                        (torch.argmax(pred, dim=-1) == y).float().mean().item()
                    )

            results["train_loss"].append(total_loss / total_cnt)
            results["train_acc"].append(total_acc / total_cnt)

            total_loss = 0
            total_acc = 0
            total_cnt = 0

            full_model.eval()
            for batch in test_loader:
                x, y = batch[0].cuda(), batch[1].cuda()

                with torch.no_grad():
                    pred = full_model(x)
                    loss = loss_fn(pred, y)

                total_loss += loss.item() * x.shape[0]
                total_acc += (torch.argmax(pred, dim=-1) == y).float().sum().item()
                total_cnt += x.shape[0]

            results["test_loss"].append(total_loss / total_cnt)
            results["test_acc"].append(total_acc / total_cnt)

        with open(f"results/{name}/train_loss.txt", "w") as f:
            f.write(str(results["train_loss"]))

        with open(f"results/{name}/train_acc.txt", "w") as f:
            f.write(str(results["train_acc"]))

        with open(f"results/{name}/inner_train_loss.txt", "w") as f:
            f.write(str(results["inner_train_loss"]))

        with open(f"results/{name}/inner_train_acc.txt", "w") as f:
            f.write(str(results["inner_train_acc"]))

        with open(f"results/{name}/test_loss.txt", "w") as f:
            f.write(str(results["test_loss"]))

        with open(f"results/{name}/test_acc.txt", "w") as f:
            f.write(str(results["test_acc"]))
