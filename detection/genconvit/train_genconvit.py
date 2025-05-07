import argparse
import torch
import os

from models.genconvit import GenConViT
from utils.trainer import Trainer
from utils.evaluator import Evaluator
from utils.deepfake_dataset import get_dataloader
from utils.logger import get_logger
from utils.transforms import GenConViTTransforms


def main():
    parser = argparse.ArgumentParser(description='Prepare deepfake dataset: extract frames and detect faces.')
    parser.add_argument('--train_dir', type=str, required=True, help='Directory containing training images')
    parser.add_argument('--val_dir', type=str, required=True, help='Directory containing validation images')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save the model and logs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs for training')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device for face detection ("cpu" or "cuda")')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # load data
    train_transforms = GenConViTTransforms(aug=True)
    val_transforms = GenConViTTransforms(aug=False)
    train_dataloader = get_dataloader(args.train_dir, train_transforms, batch_size=args.batch_size, shuffle=True)
    val_dataloader = get_dataloader(args.val_dir, val_transforms, batch_size=args.batch_size, shuffle=False)
    print('Finished loading data.')

    # create model
    device = torch.device(args.device)
    model = GenConViT('./genconvit_weights/genconvit_ed_inference.pth',
                    './genconvit_weights/genconvit_vae_inference.pth')
    
    print('Finish creating model.')

    # create trainer and evaluator
    trainer = Trainer(model, train_dataloader, lr=args.lr, epochs=args.epochs, device=device)
    evaluator = Evaluator(model, val_dataloader, device=device)

    # create logger
    logger = get_logger(os.path.join(args.save_dir, 'train.log'))

    best_auc = float('-inf')

    # training
    for epoch in range(args.epochs):
        trainer.train()
        auc = evaluator.evaluate_auc()
        logger.info(f'[Epoch {epoch + 1} / {args.epochs}]: Validation AUC: {auc:.4f}')
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), os.path.join(args.save_dir, f'best_model.pth'))
            logger.info(f'Best validation model has been saved to {args.save_dir}')

if __name__ == '__main__':
    main()