"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
"""

import math
from tqdm import tqdm
import numpy as np
from scipy import stats
import torch
import random
import io
import matplotlib as mpl
mpl.use('Agg')  # No display
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
import pickle
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('run/out_16_512_10_5lr_dropout2')

class ScatterPlots:
    def __init__(self, r, mae, y):
        self.r = r
        self.mae = mae
        self.y = y

    def plot2(self,fig):
        with io.BytesIO() as buff:
            fig.savefig(buff, format='png')
            buff.seek(0)
            im = plt.imread(buff)
            return im

    def tensorboard_plot(self,img_batch,epoch):
        img_batch = img_batch[:, :, :3]      # convert to 3 channels
        img_batch = np.transpose(img_batch, [2, 0, 1])
        img_batch = img_batch[None,:, :, :] # convert to 1 batch (None), replace with batch_size here...
        writer.add_images('image_batch', img_batch, 0)
        writer.add_figure('epoch_{:d}'.format(epoch), plt.gcf(), 0)
        writer.close()


class TrainerConfig:
    # optimization parameters
    start_epoch = 0
    max_epochs = 2
    batch_size = 100
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights

    # checkpoint settings
    ckpt_path = None
    num_workers = 0 # for DataLoader


    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:

    def __init__(self, model, train_dataset, val_dataset, test_dataset, config, output_dir):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.output_dir = output_dir

        # take over whatever gpus are on the system
        self.device = 'mps'
        self.model = torch.nn.DataParallel(self.model).to(self.device)
        self.model = self.model.to(self.device)

        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)
            self.model = self.model.to(self.device)

    def save_checkpoint(self):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        # logger.info("saving %s", self.config.ckpt_path)
        torch.save(raw_model.state_dict(), self.config.ckpt_path)

    def train(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)
        scaler = MinMaxScaler()

        #def run_epoch():
        def run_epoch(trainloader,validloader,testloader):
            train_loss=0.0
            val_loss=0.0
            test_loss=0.0

            model.train
            y_k1 = []
            y_k2 = []
            for _, d_it in tqdm(enumerate(trainloader), total=len(trainloader)):
                x, y = d_it

                # place data on the correct device
                x = x.to(self.device)

                y = y.to(self.device)

                # backprop and update the parameters
                optimizer.zero_grad()   # clear gradients for next train

                # forward the model
                output, loss = model(x, y)
                #loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                #train_loss += loss.item()

                # backprop and update the parameters
                # optimizer.zero_grad()   # clear gradients for next train
                loss.backward()         # backpropagation, compute gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                optimizer.step()         # apply gradients
                
                loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                train_loss += loss.item()

                # decay the learning rate based on our progress
                if config.lr_decay:
                    self.tokens += (y >= 0).sum() # number of tokens processed this step (i.e. label is not -100)
                    if self.tokens < config.warmup_tokens:
                        # linear warmup
                        lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                    else:
                        # cosine learning rate decay
                        progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                        lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

                    lr = config.learning_rate * lr_mult
                else:
                    lr = config.learning_rate

                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                    param_group['momentum'] = 0.2

                # report progress
                k1 = y.detach().cpu().numpy()
                k2 = output.detach().cpu().numpy()
                y_k1.append(k1)
                y_k2.append(k2)

            try:
                k1 = np.concatenate(y_k1).ravel()
                k2 = np.concatenate(y_k2).ravel()
            except:
                pass

            r2 = np.corrcoef(k1, k2.flatten())[1,0]
            print('\nEpoch %d: r-coefficient train: %.5f' % (epoch, r2))
            writer.add_scalar("r-coefficient train", r2, epoch)
            writer.add_scalar("train loss", np.mean(train_loss), epoch)

            if epoch % 5 == 0:
                y_mes = []
                y_pred = []
                test_loss = 0.0
                model.eval()     # Optional when not using Model Specific layer
                for _, d_it in tqdm(enumerate(testloader), total=len(testloader)):
                    x, y = d_it

                    # place data on the correct device
                    x = x.to(self.device)

                    y = y.to(self.device)
                    output, loss = model(x, y)
                    test_loss += loss.item()
                    y1 = output.detach().cpu().numpy()
                    y2 = y.detach().cpu().numpy()
                    y_mes.append(y2)
                    y_pred.append(y1)

                try:
                    y_mes = np.concatenate(y_mes).ravel()
                    y_pred = np.concatenate(y_pred).ravel()
                except:
                    pass
                r = (np.corrcoef(y_mes, y_pred)[1,0])
                r2 = np.corrcoef(y_mes, y_pred)
                mae = np.mean(np.abs(y_mes - y_pred))
                rmse = np.sqrt(((y_mes - y_pred) ** 2).mean())
                print(r2)

                print('\nr-coefficient test: %.2f, mae: %.2f, rmse: %.2f' % (r, mae, rmse))
                writer.add_scalar("r-coefficient test", r, epoch)
                writer.add_scalar("mae", mae, epoch)
                writer.add_scalar("rmse", rmse, epoch)
                writer.add_scalar("test loss", np.mean(test_loss), epoch)
                fp = open("data/y_pred_y_mes.obj","wb")
                pickle.dump([y_pred,y_mes],fp)
                #writer.add_scalars(f'loss/check_info', {
                #                    'test loss': np.mean(test_loss),'train loss': np.mean(train_loss),}, epoch)
            if epoch % 100 == 0:
                self.save_checkpoint()  # save model

            return train_loss, val_loss, test_loss

        best_loss = float('inf')
        self.tokens = 0 # counter used for learning rate decay

        trainloader = DataLoader(self.train_dataset, shuffle=False, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)


        validloader = DataLoader(self.val_dataset, shuffle=False, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)

        testloader = DataLoader(self.test_dataset, shuffle=False, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)

        for epoch in range(config.start_epoch, config.max_epochs):
            train_loss, val_loss, test_loss  = run_epoch(trainloader,validloader,testloader)

            # display results on Tensorboard
            writer.add_scalar("valid loss", val_loss, epoch)
