from lightning_modules import CIFAR10DataModule, LitModel, cb_list
from lightning.pytorch import Trainer, seed_everything
from argparse import ArgumentParser
from models import VGG


seed_everything(42, workers=True)


def main(args):
    dataset = CIFAR10DataModule(batch_size=args.batch)

    net = VGG('VGG19')

    model = LitModel(model=net, lr=args.lr)

    trainer = Trainer(
        max_epochs=args.epoch,
        enable_model_summary=False,
        callbacks=cb_list
    )

    trainer.fit(model, dataset)


if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    main(args)
