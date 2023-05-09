from Models import VGG
from lightning.pytorch import Trainer, seed_everything
from LitModules import LitModel, CIFARDataModule
from argparse import ArgumentParser

seed_everything(42, workers=True)

def main(args):
    dataset = CIFARDataModule(batch_size=args.batch)

    net = VGG('VGG11')

    model = LitModel(net, lr=1e-4)

    trainer = Trainer(max_epochs=args.epoch)

    trainer.fit(model, dataset)

if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    main(args)
