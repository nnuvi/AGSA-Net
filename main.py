import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn

from scipy.io import savemat

from dataset import prepare_dataset
from metrics import NonZeroClipper, output_metric, print_all_matrics
from model import AGSANet
from train import train_epoch, valid_epoch, test_epoch
from utils import set_seed

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42, help='number of seed')
    parser.add_argument('--dataset', choices=['Indian', 'Berlin', 'Augsburg', 'Houston'], default='Indian', help='dataset to use')
    parser.add_argument('--flag_test', choices=['test', 'train'], default='train', help='testing mark')
    parser.add_argument('--model_name', choices=['conv2d_unmix'], default='conv2d_unmix', help='AGSANet')
    parser.add_argument("--depth", type=int, default=3, help="Depth of the model")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument('--batch_size', type=int, default=64, help='number of batch size')
    parser.add_argument('--test_freq', type=int, default=5, help='number of evaluation')
    parser.add_argument('--patches', type=int, default=7, help='number of patches')
    parser.add_argument('--epoches', type=int, default=500, help='epoch number')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.9, help='gamma')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight_decay')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    set_seed(args.seed)
    ## prepare dataset
    ( label_train_loader, label_test_loader, label_true_loader,
      band, height, width,
      num_classes, label, total_pos_true, class_names)  = prepare_dataset(args.dataset, args.patches, args.batch_size)

    # create model
    if args.model_name == 'conv2d_unmix':
        model = AGSANet(band, num_classes, args.patches, args.depth, args.num_heads, args.model_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
    else:
        raise KeyError("{} model is unknown.".format(args.model_name))
    model = model.to(device)
    print("Model Name: {}".format(args.model_name))

    # criterion
    criterion = nn.CrossEntropyLoss().to(device)
    # Set the optimizer
    if 'unmix' in args.model_name:
        params = map(id, model.unmix_decoder.parameters())
        ignored_params = list(set(params))
        base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
        optimizer = torch.optim.Adam([{'params': base_params},{'params': model.unmix_decoder.parameters(), 'lr': 3e-4}],
                                    lr = args.learning_rate, weight_decay = args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    apply_nonegative = NonZeroClipper()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epoches//10, gamma=args.gamma)
    #-------------------------------------------------------------------------------
    if args.flag_test == 'test':
        model.eval()
        model.load_state_dict(torch.load(f'./result/{args.dataset}/model.pkl'))

        print("Calculating test metrics...")

        tar_v, pre_v = valid_epoch(model, args.model_name, label_test_loader, criterion, optimizer)
        OA2, AA_mean2, Kappa2, AA2 = output_metric(tar_v, pre_v)
        print("Final Test Result:")
        print("OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(OA2, AA_mean2, Kappa2))
        print("Class-wise AA:", AA2)

        pre_u = test_epoch(model, args.model_name, label_true_loader)

        out_dir = './output'
        os.makedirs(out_dir, exist_ok=True)

        prediction_matrix = np.zeros((height, width), dtype=float)
        print_all_matrics(pre_u, label, total_pos_true, prediction_matrix, args.dataset, num_classes, class_names, out_dir)
    else:
        print("start training")
        tic = time.time()
        min_val_obj, best_OA = 0.5, 0
        for epoch in range(args.epoches):
            # scheduler.step()

            # train model
            model.train()
            train_acc, train_obj, tar_t, pre_t, embed_dim, depth, num_heads = train_epoch(model, args.model_name, label_train_loader, criterion, optimizer)
            OA1, AA_mean1, Kappa1, AA1 = output_metric(tar_t, pre_t)
            print("Epoch: {:03d} train_loss: {:.4f} train_acc: {:.4f}"
                            .format(epoch+1, train_obj, train_acc))

            if 'unmix' in args.model_name: # regularize unmix decoder
                model.unmix_decoder.apply(apply_nonegative)

            if (epoch % args.test_freq == 0) | (epoch == args.epoches - 1):
                model.eval()
                tar_v, pre_v = valid_epoch(model, args.model_name, label_test_loader, criterion, optimizer)
                OA2, AA_mean2, Kappa2, AA2 = output_metric(tar_v, pre_v)
                print("OA: {:.4f} AA: {:.4f} Kappa: {:.4f}"
                            .format(OA2, AA_mean2, Kappa2))
                print("*************************")

                if OA2 > min_val_obj and epoch > 10:
                    save_path = f'./result/{args.dataset}'
                    os.makedirs(save_path, exist_ok=True)
                    model_save_path = (
                        f"{save_path}/{args.dataset}_{args.model_name}_p{args.patches}_b{args.batch_size}_"
                        f"ed{embed_dim}_d{depth}_nh{num_heads}_"
                        f"{OA2*100:.2f}_epoch{epoch}.pkl"
                    )

                    torch.save(model.state_dict(), model_save_path)

                    min_val_obj = OA2
                    best_epoch = epoch
                    best_OA = OA2
                    best_AA = AA_mean2
                    best_Kappa = Kappa2
                    best_each_AA = AA2
            scheduler.step()

        toc = time.time()
        print("Running Time: {:.2f}".format(toc-tic))
        print("**************************************************")
        if best_OA == 0:
            save_path = f'./result/{args.dataset}'
            os.makedirs(save_path, exist_ok=True)
            model_save_path = (
                    f"{save_path}/{args.dataset}_{args.model_name}_p{args.patches}_b{args.batch_size}_"
                    f"ed{embed_dim}_d{depth}_nh{num_heads}_"
                    f"{OA2*100:.2f}_epoch{epoch}.pkl"
            )
            torch.save(model.state_dict(), model_save_path)

    if args.flag_test == 'train':
        print("Final result:")
        print("OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(OA2, AA_mean2, Kappa2))
        print(AA2)
        print("**************************************************")
        print("Best Epoch: {:03d} | Best OA: {:.4f} | Best AA: {:.4f} | Best Kappa: {:.4f}".format(best_epoch, best_OA, best_AA, best_Kappa))
        print(best_each_AA)
