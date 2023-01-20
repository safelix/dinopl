import os


os.environ["DINO_DATA"] = "datasets"
os.environ["DINO_RESULTS"] = "results"

from dino import *
from tqdm import tqdm

from configuration import Configuration
import argparse


def add_parameter_to_model(model, parameter):
    cnt = 0
    for p in model.parameters():
        size = p.numel()
        p.data = p.data + parameter[cnt : cnt + size].reshape(p.data.shape).cuda()
        cnt += size


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--network", type=str, default="resnet", choices=["resnet", "vgg11"]
    )
    parser.add_argument(
        "--init_loc",
        type=str,
        default="teacher",
        choices=[
            "teacher",
            "student_0_best",
            "student_0_last",
            "student_1_best",
            "student_1_last",
        ],
    )
    parser.add_argument("--norm_limit", type=int, default=60)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--num_points", type=int, default=100)
    parser.add_argument("--opt_direction", action="store_true")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    args.name = (
        args.network
        + "_"
        + args.init_loc
        + "_norm_"
        + str(args.norm_limit)
        + "_points_"
        + str(args.num_points)
        + "_opt_direction_"
        + str(args.opt_direction)
        + "_seed_"
        + str(args.seed)
    )

    # 1. Get defaults from parser
    config = Configuration.get_default()
    config = Configuration.from_json("configs/cifar10_distillation_v2.json", config)

    config.dataset = "cifar10"
    config.enc = args.network

    config.mc_spec = create_mc_spec(config)

    DSet = get_dataset(config)
    self_trfm = transforms.Compose(
        [  # self-training
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.ToTensor(),
            transforms.Normalize(DSet.mean, DSet.std),
        ]
    )

    if config.float64:  # convert inputs to float64 if needed
        self_trfm = transforms.Compose(
            [self_trfm, transforms.ConvertImageDtype(torch.float64)]
        )

    eval_trfm = transforms.Compose(
        [transforms.Resize(size=config.mc_spec[0]["out_size"]), self_trfm]  # evaluation
    )
    mc = MultiCrop(config.mc_spec, per_crop_transform=self_trfm)

    # Data Setup.
    dino_train_set = DSet(root="datasets", train=True, transform=mc)

    dl_args = dict(
        num_workers=config.n_workers, pin_memory=False if config.force_cpu else True
    )
    dino_train_dl = DataLoader(
        dataset=dino_train_set,
        batch_size=config.bs_train,
        shuffle=True,
        generator=None,
        **dl_args,
    )

    # Model Setup.
    enc = get_encoder(config)()
    config.embed_dim = enc.embed_dim
    head = DINOHead(
        config.embed_dim,
        config.out_dim,
        hidden_dims=config.hid_dims,
        l2bot_dim=config.l2bot_dim,
        l2bot_cfg=config.l2bot_cfg,
        use_bn=config.mlp_bn,
        act_fn=config.mlp_act,
    )
    model = DINOModel(enc, head)

    student, teacher = init_student_teacher(config=config, model=model)

    # DINO Setup
    dino = DINO(
        mc_spec=config.mc_spec,
        student=student,
        teacher=teacher,
        s_mode=config.s_mode,
        t_mode=config.t_mode,
        t_mom=Schedule.parse(config.t_mom),
        t_update_every=config.t_update_every,
        t_bn_mode=config.t_bn_mode,
        t_eval=config.t_eval,
        t_cmom=Schedule.parse(config.t_cmom),
        s_cmom=Schedule.parse(config.s_cmom),
        t_temp=Schedule.parse(config.t_temp),
        s_temp=Schedule.parse(config.s_temp),
        loss=config.loss,
        loss_pairing=config.loss_pairing,
        opt=create_optimizer(config),
        opt_lr=Schedule.parse(config.opt_lr),
        opt_wd=Schedule.parse(config.opt_wd),
        wn_freeze_epochs=config.wn_freeze_epochs,
    )

    dino.cuda()
    dino.eval()

    from losslandscape import load_model

    if config.enc == "resnet":
        initial_teacher = load_model(
            "saves/2b1fre3w/epoch=56-step=11172-probe_student=0.445.ckpt:teacher"
        )
        student_alpha_0_best = load_model(
            "saves/2b1fre3w/epoch=56-step=11172-probe_student=0.445.ckpt:student"
        )
        student_alpha_0_last = load_model("saves/2b1fre3w/last.ckpt:student")
        student_alpha_1_best = load_model(
            "saves/34hao21o/epoch=80-step=15876-probe_student=0.397.ckpt:student"
        )
        student_alpha_1_last = load_model("saves/34hao21o/last.ckpt:student")
    elif config.enc == "vgg11":
        initial_teacher = load_model(
            "saves/22abl14o/epoch=58-step=11564-probe_student=0.524.ckpt:teacher"
        )
        student_alpha_0_best = load_model(
            "saves/22abl14o/epoch=58-step=11564-probe_student=0.524.ckpt:student"
        )
        student_alpha_0_last = load_model("saves/22abl14o/last.ckpt:student")
        student_alpha_1_best = load_model(
            "saves/3mtlpc13/epoch=96-step=19012-probe_student=0.446.ckpt:student"
        )
        student_alpha_1_last = load_model("saves/3mtlpc13/last.ckpt:student")

    if args.init_loc == "teacher":
        teacher.load_state_dict(initial_teacher.state_dict())
    elif args.init_loc == "student_0_best":
        student.load_state_dict(student_alpha_0_best.state_dict())
    elif args.init_loc == "student_0_last":
        student.load_state_dict(student_alpha_0_last.state_dict())
    elif args.init_loc == "student_1_best":
        student.load_state_dict(student_alpha_1_best.state_dict())
    elif args.init_loc == "student_1_last":
        student.load_state_dict(student_alpha_1_last.state_dict())

    if not args.opt_direction:
        # fix random seed
        torch.manual_seed(args.seed)

        direction = torch.randn(sum([p.numel() for p in dino.student.parameters()]))
        direction = direction / torch.norm(direction)
    else:
        assert args.init_loc == "teacher"
        raise NotImplementedError("TODO")
        weights_teacher = torch.cat([p.reshape(-1).cpu() for p in teacher.parameters()])
        weights_student = torch.cat([p.reshape(-1).cpu() for p in student.parameters()])

    muls = torch.linspace(args.norm_limit * (-1), args.norm_limit, args.num_points)

    added_directions = direction[None, :] * muls[:, None]

    values = []

    for added_direction in tqdm(added_directions):
        student.load_state_dict(teacher.state_dict())
        add_parameter_to_model(student, added_direction)

        kls = 0
        total = 0

        with torch.no_grad():
            for batch in dino_train_dl:
                batch = torch.stack(batch[0]).cuda(), batch[1].cuda()
                out = dino.validation_step(batch, None)
                kls += out["KL"] * batch[1].shape[0]

                total += batch[1].shape[0]

        res = kls / total

        values.append(res.item())

    with open(f"{args.results_dir}/{args.name}.txt", "w") as f:
        f.write(str(values))
