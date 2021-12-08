import os
import time
import shutil
import pickle

import torch
import torch.nn.functional as F

from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tensorboard_logger import configure, log_value

import pandas as pd

from model import RecurrentAttention
from stop_model import StopRecurrentAttention
from utils import AverageMeter


class Trainer:
    """A Recurrent Attention Model trainer.

    All hyperparameters are provided by the user in the
    config file.
    """

    def __init__(self, config, data_loader):
        """
        Construct a new Trainer instance.

        Args:
            config: object containing command line arguments.
            data_loader: A data iterator.
        """
        self.config = config

        if config.use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # glimpse network params
        self.patch_size = config.patch_size
        self.glimpse_scale = config.glimpse_scale
        self.num_patches = config.num_patches
        self.loc_hidden = config.loc_hidden
        self.glimpse_hidden = config.glimpse_hidden

        # core network params
        self.num_glimpses = config.num_glimpses
        self.hidden_size = config.hidden_size
        self.include_stop = config.include_stop

        # reinforce params
        self.std = config.std
        self.M = config.M

        # data params
        if config.is_train:
            self.train_loader = data_loader[0]
            self.valid_loader = data_loader[1]
            self.num_train = len(self.train_loader.sampler.indices)
            self.num_valid = len(self.valid_loader.sampler.indices)
        else:
            self.test_loader = data_loader
            self.num_test = len(self.test_loader.dataset)
        self.num_classes = config.num_classes
        self.num_channels = config.num_channels

        # training params
        self.epochs = config.epochs
        self.start_epoch = 0
        self.momentum = config.momentum
        self.lr = config.init_lr
        self.hesitation_penalty = config.hesitation_penalty

        # misc params
        self.best = config.best
        self.logs_dir = config.logs_dir
        self.best_valid_acc = 0.0
        self.counter = 0
        self.lr_patience = config.lr_patience
        self.train_patience = config.train_patience
        self.use_tensorboard = config.use_tensorboard
        self.resume = config.resume
        self.print_freq = config.print_freq
        self.plot_freq = config.plot_freq
        self.model_name = config.model_name
        self.model_dir = config.model_dir
        self.plot_dir = config.plot_dir
        

        # configure tensorboard logging
        if self.use_tensorboard:
            tensorboard_dir = self.logs_dir + self.model_name
            print("[*] Saving tensorboard logs to {}".format(tensorboard_dir))
            if not os.path.exists(tensorboard_dir):
                os.makedirs(tensorboard_dir)
            configure(tensorboard_dir)

        # build RAM model
        if self.include_stop:
          self.model = StopRecurrentAttention(
            self.patch_size,
            self.num_patches,
            self.glimpse_scale,
            self.num_channels,
            self.loc_hidden,
            self.glimpse_hidden,
            self.std,
            self.hidden_size,
            self.num_classes,
        )
        else:
          self.model = RecurrentAttention(
            self.patch_size,
            self.num_patches,
            self.glimpse_scale,
            self.num_channels,
            self.loc_hidden,
            self.glimpse_hidden,
            self.std,
            self.hidden_size,
            self.num_classes,
        )
        self.model.to(self.device)

        # initialize optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.init_lr
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, "min", patience=self.lr_patience
        )

    def reset(self):
        h_t = torch.zeros(
            self.batch_size,
            self.hidden_size,
            dtype=torch.float,
            device=self.device,
            requires_grad=True,
        )
        l_t = torch.zeros(
            self.batch_size,
            2,
            dtype=torch.float,
            device=self.device,
            requires_grad=True,
        )
        l_t.requires_grad = True
        if not self.include_stop:
          return h_t, l_t
          
        s_t = torch.ones(
            self.batch_size,
            1,
            dtype=torch.float,
            device=self.device,
            requires_grad=True,
        )

        return h_t, l_t, s_t

    def train(self):
        """Train the model on the training set.

        A checkpoint of the model is saved after each epoch
        and if the validation accuracy is improved upon,
        a separate ckpt is created for use on the test set.
        """
        # load the most recent checkpoint
        if self.resume:
            self.load_checkpoint(best=False)

        print(
            "\n[*] Train on {} samples, validate on {} samples".format(
                self.num_train, self.num_valid
            )
        )

        for epoch in range(self.start_epoch, self.epochs):

            print(
                "\nEpoch: {}/{} - LR: {:.6f}".format(
                    epoch + 1, self.epochs, self.optimizer.param_groups[0]["lr"]
                )
            )

            # train for 1 epoch
            if self.include_stop:
              train_loss, train_acc = self.train_one_epoch_stop(epoch)
            else:
              train_loss, train_acc = self.train_one_epoch(epoch)

            # evaluate on validation set
            if self.include_stop:
              valid_loss, valid_acc = self.validate(epoch)
            else:
              valid_loss, valid_acc = self.validate(epoch)
              
            # # reduce lr if validation loss plateaus
            self.scheduler.step(-valid_acc)

            is_best = valid_acc > self.best_valid_acc
            msg1 = "train loss: {:.3f} - train acc: {:.3f} "
            msg2 = "- val loss: {:.3f} - val acc: {:.3f} - val err: {:.3f}"
            if is_best:
                self.counter = 0
                msg2 += " [*]"
            msg = msg1 + msg2
            print(
                msg.format(
                    train_loss, train_acc, valid_loss, valid_acc, 100 - valid_acc
                )
            )

            # check for improvement
            if not is_best:
                self.counter += 1
            if self.counter > self.train_patience:
                print("[!] No improvement in a while, stopping training.")
                return
            self.best_valid_acc = max(valid_acc, self.best_valid_acc)
            self.save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "model_state": self.model.state_dict(),
                    "optim_state": self.optimizer.state_dict(),
                    "best_valid_acc": self.best_valid_acc,
                },
                is_best,
            )

    def train_one_epoch_stop(self, epoch):
        """
        Train the model for 1 epoch of the training set.

        An epoch corresponds to one full pass through the entire
        training set in successive mini-batches.

        This is used by train() and should not be called manually.
        """
        self.model.train()
        batch_time = AverageMeter()
        losses = AverageMeter()
        accs = AverageMeter()

        tic = time.time()
        with tqdm(total=self.num_train) as pbar:
            for i, (x, y) in enumerate(self.train_loader):
                self.optimizer.zero_grad()

                x, y = x.to(self.device), y.to(self.device, dtype=torch.int64)

                plot = False
                if (epoch % self.plot_freq == 0) and (i == 0):
                    plot = True

                # initialize location vector and hidden state
                self.batch_size = x.shape[0]
                h_t, l_t, s_t = self.reset()

                # save images
                imgs = []
                imgs.append(x[0:9])

                # extract the glimpses
                locs = []
                l_log_pi = []
                s_log_pi = []
                baselines = []
                log_probas = []
                stop_signals = []
                for t in range(self.num_glimpses):
                    # forward pass through model
                    h_t, l_t, s_t, b_t, log_ps, l_p, s_p = self.model(x, l_t, h_t, s_t, t == self.num_glimpses - 1)

                    # store
                    locs.append(l_t[0:9])
                    baselines.append(b_t)
                    l_log_pi.append(l_p)
                    s_log_pi.append(s_p)
                    log_probas.append(log_ps)
                    stop_signals.append(s_t)


                # # last iteration
                # h_t, l_t, b_t, log_probas, p = self.model(x, l_t, h_t, last=True)
                # log_pi.append(p)
                # baselines.append(b_t)
                # locs.append(l_t[0:9])

                # convert list to tensors and reshape
                baselines = torch.stack(baselines).transpose(1, 0)
                l_log_pi = torch.stack(l_log_pi).transpose(1, 0)
                s_log_pi = torch.stack(s_log_pi).transpose(1, 0)
                log_probas = torch.stack(log_probas).transpose(1, 0) 
                stop_signals = torch.stack(stop_signals).transpose(1, 0).squeeze(2)

                #process stop signals
                up_through_stop = stop_signals
                count = torch.arange(self.batch_size)
                num_steps = torch.sum(stop_signals, dim=1).long()
                up_through_stop[count,num_steps] += 1

                #extract log_probas at first stop signal
                log_probas = log_probas[count,num_steps,:]


                #clip histories after stop signal
                baselines = baselines * up_through_stop
                l_log_pi = l_log_pi * up_through_stop
                s_log_pi = s_log_pi * up_through_stop



                # calculate reward
                predicted = torch.max(log_probas, 1)[1]
                R = (predicted.detach() == y).float()
                R = R.unsqueeze(1).repeat(1, self.num_glimpses)
                mask = (torch.arange(R.size(1), device=num_steps.device)==num_steps.unsqueeze(1))
                R = mask*R #Reward of 1 at first stop signal
                R = R - stop_signals * self.hesitation_penalty


                # compute losses for differentiable modules
                loss_action = F.nll_loss(log_probas, y)
                loss_baseline = F.mse_loss(baselines, R)

                # compute reinforce loss
                # summed over timesteps and averaged across batch
                adjusted_reward = R - baselines.detach()
                loss_reinforce = torch.sum(-l_log_pi * adjusted_reward, dim=1) + torch.sum(-s_log_pi * adjusted_reward, dim=1)
                loss_reinforce = torch.mean(loss_reinforce, dim=0)

                # sum up into a hybrid loss
                loss = loss_action + loss_baseline + loss_reinforce * 0.01

                # compute accuracy
                correct = (predicted == y).float()
                acc = 100 * (correct.sum() / len(y))

                # store
                losses.update(loss.item(), x.size()[0])
                accs.update(acc.item(), x.size()[0])

                # compute gradients and update SGD
                loss.backward()
                self.optimizer.step()

                # measure elapsed time
                toc = time.time()
                batch_time.update(toc - tic)

                pbar.set_description(
                    (
                        "{:.1f}s - loss: {:.3f} - acc: {:.3f}".format(
                            (toc - tic), loss.item(), acc.item()
                        )
                    )
                )
                pbar.update(self.batch_size)

                # dump the glimpses and locs
                if plot:
                    imgs = [g.cpu().data.numpy().squeeze() for g in imgs]
                    locs = [l.cpu().data.numpy() for l in locs]
                    pickle.dump(
                        imgs, open(self.plot_dir + "g_{}.p".format(epoch + 1), "wb")
                    )
                    pickle.dump(
                        locs, open(self.plot_dir + "l_{}.p".format(epoch + 1), "wb")
                    )

                # log to tensorboard
                if self.use_tensorboard:
                    iteration = epoch * len(self.train_loader) + i
                    log_value("train_loss", losses.avg, iteration)
                    log_value("train_acc", accs.avg, iteration)

            return losses.avg, accs.avg

    @torch.no_grad()
    def validate_stop(self, epoch):
        """Evaluate the RAM model on the validation set.
        """
        losses = AverageMeter()
        accs = AverageMeter()

        for i, (x, y) in enumerate(self.valid_loader):
            x, y = x.to(self.device), y.to(self.device, dtype=torch.int64)

            # duplicate M times
            x = x.repeat(self.M, 1, 1, 1)

            # initialize location vector and hidden state
            self.batch_size = x.shape[0]
            h_t, l_t, s_t = self.reset()

            # extract the glimpses
            l_log_pi = []
            s_log_pi = []
            baselines = []
            log_probas = []
            stop_signals = []
            for t in range(self.num_glimpses):
                # forward pass through model
                h_t, l_t, s_t, b_t, log_ps, l_p, s_p = self.model(x, l_t, h_t, s_t)

                # store
                baselines.append(b_t)
                l_log_pi.append(l_p)
                s_log_pi.append(s_p)
                log_probas.append(log_ps)
                stop_signals.append(s_t)
                

            # convert list to tensors and reshape
            baselines = torch.stack(baselines).transpose(1, 0)
            l_log_pi = torch.stack(l_log_pi).transpose(1, 0)
            s_log_pi = torch.stack(s_log_pi).transpose(1, 0)
            log_probas = torch.stack(log_probas).transpose(1, 0) 
            stop_signals = torch.stack(stop_signals).transpose(1, 0).squeeze(2)

            #process stop signals
            up_through_stop = stop_signals
            count = torch.arange(self.batch_size)
            num_steps = torch.sum(stop_signals, dim=1).long()
            up_through_stop[count,num_steps] += 1

            #extract log_probas at first stop signal
            log_probas = log_probas[count,num_steps,:]


            #clip histories after stop signal
            baselines = baselines * up_through_stop
            l_log_pi = l_log_pi * up_through_stop
            s_log_pi = s_log_pi * up_through_stop
            

            # average
            log_probas = log_probas.view(self.M, -1, log_probas.shape[-1])
            log_probas = torch.mean(log_probas, dim=0)

            baselines = baselines.contiguous().view(self.M, -1, baselines.shape[-1])
            baselines = torch.mean(baselines, dim=0)

            l_log_pi = l_log_pi.contiguous().view(self.M, -1, l_log_pi.shape[-1])
            l_log_pi = torch.mean(l_log_pi, dim=0)
            s_log_pi = s_log_pi.contiguous().view(self.M, -1, s_log_pi.shape[-1])
            s_log_pi = torch.mean(s_log_pi, dim=0)

            # calculate reward
            predicted = torch.max(log_probas, 1)[1]
            R = (predicted.detach() == y).float()
            R = R.unsqueeze(1).repeat(1, self.num_glimpses)
            mask = (torch.arange(R.size(1), device=num_steps.device)==num_steps.unsqueeze(1))
            R = mask*R
            R = R - stop_signals * self.hesitation_penalty


            # compute losses for differentiable modules
            loss_action = F.nll_loss(log_probas, y)
            loss_baseline = F.mse_loss(baselines, R)

            # compute reinforce loss
            adjusted_reward = R - baselines.detach()
            loss_reinforce = torch.sum(-l_log_pi * adjusted_reward, dim=1) + torch.sum(-s_log_pi * adjusted_reward, dim=1)
            loss_reinforce = torch.mean(loss_reinforce, dim=0)

            # sum up into a hybrid loss
            loss = loss_action + loss_baseline + loss_reinforce * 0.01

            # compute accuracy
            correct = (predicted == y).float()
            acc = 100 * (correct.sum() / len(y))

            # store
            losses.update(loss.item(), x.size()[0])
            accs.update(acc.item(), x.size()[0])

            # log to tensorboard
            if self.use_tensorboard:
                iteration = epoch * len(self.valid_loader) + i
                log_value("valid_loss", losses.avg, iteration)
                log_value("valid_acc", accs.avg, iteration)

        return losses.avg, accs.avg

    @torch.no_grad()
    def test_stop(self):
        """Test the RAM model.

        This function should only be called at the very
        end once the model has finished training.
        """
        correct = 0

        # load the best checkpoint
        self.load_checkpoint(best=self.best)

        # removed image, final_softmax, hidden layer, softmax, final_persistant_softmax
        cols = ['image_id', 'timestep', 'num_dots', 'area', 
                'num_steps', 'final_prediction', 'next_location', 'prediction', 
                'final_persistant_absolute_error', 'final_persistant_prediction',
                'absolute_error', 'stop_signal', 'stop_probability']
        # changed naming to SMALL
        filename = self.model_name + "_SMALL.csv"
        test_path = os.path.join(self.model_dir, filename)
        
        
        for i, (x, y, a) in enumerate(self.test_loader):
            df_dict = {column_name : [] for column_name in cols}

            batch_size = x.shape[0]
            df_dict['image_id'].extend(sum([[image_id]*self.num_glimpses for image_id in range(i*batch_size, (i+1)*batch_size)], []))
            df_dict['timestep'].extend(sum([list(range(self.num_glimpses)) for image_id in range(i*batch_size, (i+1)*batch_size)], []))
            # df_dict['image'].extend(x.repeat_interleave(self.num_glimpses, dim=0).cpu().tolist())
            repeat_y = y.repeat_interleave(self.num_glimpses)
            df_dict['num_dots'].extend(repeat_y.cpu().tolist())
            df_dict['area'].extend(a.repeat_interleave(self.num_glimpses).cpu().tolist())



            x, y = x.to(self.device), y.to(self.device, dtype=torch.int64)


            # initialize location vector and hidden state
            self.batch_size = x.shape[0]
            h_t, l_t, s_t = self.reset()

            # extract the glimpses
            l_ts = []
            h_ts = []
            output_log_probas = []
            stop_signals = []
            stop_log_probs = []
            for t in range(self.num_glimpses):
                # forward pass through model
                h_t, l_t, s_t, b_t, log_ps, l_p, s_p = self.model(x, l_t, h_t, s_t)

                # store
                h_ts.append(h_t)
                l_ts.append(l_t)
                output_log_probas.append(log_ps)
                stop_signals.append(s_t)
                stop_log_probs.append(s_p)

            # convert list to tensors and reshape
            output_log_probas = torch.stack(output_log_probas).transpose(1, 0) 
            h_ts = torch.stack(h_ts).transpose(1, 0)
            l_ts = torch.stack(l_ts).transpose(1, 0)
            stop_log_probs = torch.stack(stop_log_probs).transpose(1, 0)
            stop_signals = torch.stack(stop_signals).transpose(1, 0)

            stretched_output_log_probas = output_log_probas.reshape(batch_size*self.num_glimpses, -1)
            stretched_h_ts = h_ts.reshape(batch_size * self.num_glimpses, -1)
            stretched_l_ts = l_ts.reshape(batch_size * self.num_glimpses, -1)
            stretched_stop_log_probs = stop_log_probs.reshape(batch_size * self.num_glimpses, -1)
            stretched_stop_signals = stop_signals.reshape(batch_size * self.num_glimpses, -1)

            softmaxes = torch.exp(output_log_probas)
            stretched_softmaxes = softmaxes.reshape(batch_size * self.num_glimpses, -1)
            stretched_stop_probs = torch.exp(stretched_stop_log_probs)
            # df_dict['softmax'].extend(stretched_softmaxes.cpu().tolist())
            df_dict['stop_probability'].extend(stretched_stop_probs.squeeze(1).cpu().tolist())
            # df_dict['hidden_layer'].extend(stretched_h_ts.cpu().tolist())
            df_dict['next_location'].extend(stretched_l_ts.cpu().tolist())
            stop_signals = stop_signals.squeeze(2)
            df_dict['stop_signal'].extend(stretched_stop_signals.squeeze(1).cpu().tolist())


            #process stop signals
            count = torch.arange(batch_size)
            num_steps = torch.sum(stop_signals, dim=1).long()
            #print(f"num steps: {num_steps}")
            df_dict['num_steps'].extend(num_steps.repeat_interleave(self.num_glimpses).cpu().tolist())

            up_through_stop = stop_signals
            up_through_stop[count,num_steps] += 1
            final_persistant_mask = (up_through_stop == 0)
            #print(f"mask shape: {final_persistant_mask.shape}")
            #print(f"mask: {final_persistant_mask}")

            #extract output_log_probas at first stop signal
            final_softmax = softmaxes[count,num_steps,:]
            #print(f"final soft shape: {final_softmax.shape}")
            #print(f"final soft: {final_softmax}")
            # df_dict['final_softmax'].extend(final_softmax.repeat_interleave(self.num_glimpses, dim = 0).cpu().tolist())
            unsqueezed_final_persistant_mask = final_persistant_mask.unsqueeze(2)
            repeated_final_softmax = final_softmax.unsqueeze(1).repeat(1,self.num_glimpses,1)
            final_persistant_softmaxes = torch.where(unsqueezed_final_persistant_mask, repeated_final_softmax, softmaxes)
            # df_dict['final_persistant_softmax'].extend(final_persistant_softmaxes.reshape(batch_size*self.num_glimpses, -1).cpu().tolist())

            final_pred = final_softmax.data.max(1, keepdim=True)[1]
            #print(f"final pred: {final_pred}")
            df_dict['final_prediction'].extend(final_pred.repeat_interleave(self.num_glimpses).cpu().tolist())
            correct += final_pred.eq(y.data.view_as(final_pred)).cpu().sum()

            stretched_predictions = stretched_softmaxes.data.max(1, keepdim=True)[1].squeeze(1)
            df_dict['prediction'].extend(stretched_predictions.cpu().tolist())
            predictions = stretched_predictions.reshape(batch_size, self.num_glimpses)
            repeated_final_pred = final_pred.repeat(1, self.num_glimpses)
            final_persistant_predictions = torch.where(final_persistant_mask, repeated_final_pred, predictions)
            stretched_final_persistant_predictions = final_persistant_predictions.reshape(batch_size*self.num_glimpses, -1)
            #print(f"stretched_final_persistant_predictions shape: {stretched_final_persistant_predictions.shape}")
            #print(f"stretched_final_persistant_predictions: {stretched_final_persistant_predictions}")
            df_dict['final_persistant_prediction'].extend(stretched_final_persistant_predictions.squeeze(1).cpu().tolist())


            #print(f"stretched_pred/y devices: {stretched_predictions.device}, {repeat_y.device}")
            stretched_error = torch.abs(stretched_predictions - repeat_y.cuda())
            df_dict['absolute_error'].extend(stretched_error.cpu().tolist())
            #print(f"error : {df_dict['absolute_error']}")
            final_error = torch.abs(final_pred - y.unsqueeze(1))
            error = stretched_error.reshape(batch_size, self.num_glimpses)
            repeated_final_error = final_error.repeat(1, self.num_glimpses)
            # print(f"shapes: {final_persistant_mask.shape}, {repeated_final_error.shape}, {error.shape}")
            final_persistant_error = torch.where(final_persistant_mask, repeated_final_error, error.long())
            stretched_final_persistant_error = final_persistant_error.reshape(batch_size*self.num_glimpses, -1)
            #print(f"stretched_final_persistant_error shape: {stretched_final_persistant_error.shape}")
            #print(f"stretched_final_persistant_error: {stretched_final_persistant_error}")
            df_dict['final_persistant_absolute_error'].extend(stretched_final_persistant_error.squeeze(1).cpu().tolist())
            
            df = pd.DataFrame(df_dict)
            df.to_csv(test_path, mode='a', header=not os.path.exists(test_path))


        perc = (100.0 * correct) / (self.num_test)
        error = 100 - perc
        print(
            "[*] Test Acc: {}/{} ({:.2f}% - {:.2f}%)".format(
                correct, self.num_test, perc, error
            )
        )



    def save_checkpoint(self, state, is_best):
        """Saves a checkpoint of the model.

        If this model has reached the best validation accuracy thus
        far, a seperate file with the suffix `best` is created.
        """
        filename = self.model_name + "_ckpt.pth.tar"
        model_path = os.path.join(self.model_dir, filename)
        torch.save(state, model_path)
        if is_best:
            filename = self.model_name + "_model_best.pth.tar"
            shutil.copyfile(model_path, os.path.join(self.model_dir, filename))

    def load_checkpoint(self, best=False):
        """Load the best copy of a model.

        This is useful for 2 cases:
        - Resuming training with the most recent model checkpoint.
        - Loading the best validation model to evaluate on the test data.

        Args:
            best: if set to True, loads the best model.
                Use this if you want to evaluate your model
                on the test data. Else, set to False in which
                case the most recent version of the checkpoint
                is used.
        """
        print("[*] Loading model from {}".format(self.model_dir))

        filename = self.model_name + "_ckpt.pth.tar"
        if best:
            filename = self.model_name + "_model_best.pth.tar"
        model_path = os.path.join(self.model_dir, filename)
        model = torch.load(model_path)

        # load variables from checkpoint
        self.start_epoch = model["epoch"]
        self.best_valid_acc = model["best_valid_acc"]
        self.model.load_state_dict(model["model_state"])
        self.optimizer.load_state_dict(model["optim_state"])

        if best:
            print(
                "[*] Loaded {} checkpoint @ epoch {} "
                "with best valid acc of {:.3f}".format(
                    filename, model["epoch"], model["best_valid_acc"]
                )
            )
        else:
            print("[*] Loaded {} checkpoint @ epoch {}".format(filename, model["epoch"]))


    def save_checkpoint(self, state, is_best):
        """Saves a checkpoint of the model.

        If this model has reached the best validation accuracy thus
        far, a seperate file with the suffix `best` is created.
        """
        filename = self.model_name + "_ckpt.pth.tar"
        model_path = os.path.join(self.model_dir, filename)
        torch.save(state, model_path)
        if is_best:
            filename = self.model_name + "_model_best.pth.tar"
            shutil.copyfile(model_path, os.path.join(self.model_dir, filename))

    def load_checkpoint(self, best=False):
        """Load the best copy of a model.

        This is useful for 2 cases:
        - Resuming training with the most recent model checkpoint.
        - Loading the best validation model to evaluate on the test data.

        Args:
            best: if set to True, loads the best model.
                Use this if you want to evaluate your model
                on the test data. Else, set to False in which
                case the most recent version of the checkpoint
                is used.
        """
        print("[*] Loading model from {}".format(self.model_dir))

        filename = self.model_name + "_ckpt.pth.tar"
        if best:
            filename = self.model_name + "_model_best.pth.tar"
        model_path = os.path.join(self.model_dir, filename)
        model = torch.load(model_path)

        # load variables from checkpoint
        self.start_epoch = model["epoch"]
        self.best_valid_acc = model["best_valid_acc"]
        self.model.load_state_dict(model["model_state"])
        self.optimizer.load_state_dict(model["optim_state"])

        if best:
            print(
                "[*] Loaded {} checkpoint @ epoch {} "
                "with best valid acc of {:.3f}".format(
                    filename, model["epoch"], model["best_valid_acc"]
                )
            )
        else:
            print("[*] Loaded {} checkpoint @ epoch {}".format(filename, model["epoch"]))




































    def train_one_epoch(self, epoch):
        """
        Train the model for 1 epoch of the training set.

        An epoch corresponds to one full pass through the entire
        training set in successive mini-batches.

        This is used by train() and should not be called manually.
        """
        self.model.train()
        batch_time = AverageMeter()
        losses = AverageMeter()
        accs = AverageMeter()

        tic = time.time()
        with tqdm(total=self.num_train) as pbar:
            for i, (x, y) in enumerate(self.train_loader):
                self.optimizer.zero_grad()

                x, y = x.to(self.device), y.to(self.device, dtype=torch.int64)

                plot = False
                if (epoch % self.plot_freq == 0) and (i == 0):
                    plot = True

                # initialize location vector and hidden state
                self.batch_size = x.shape[0]
                #h_t, l_t, s_t = self.reset()
                h_t, l_t = self.reset()

                # save images
                imgs = []
                imgs.append(x[0:9])

                # extract the glimpses
                locs = []
                l_log_pi = []
                #s_log_pi = []
                baselines = []
                log_probas = []
                #stop_signals = []
                for t in range(self.num_glimpses):
                    # forward pass through model
                    #h_t, l_t, s_t, b_t, log_ps, l_p, s_p = self.model(x, l_t, h_t, s_t, t == self.num_glimpses - 1)
                    h_t, l_t, b_t, log_ps, l_p = self.model(x, l_t, h_t, t == self.num_glimpses - 1)

                    # store
                    locs.append(l_t[0:9])
                    baselines.append(b_t)
                    l_log_pi.append(l_p)
                    #s_log_pi.append(s_p)
                    log_probas.append(log_ps)
                    #stop_signals.append(s_t)


                # # last iteration
                # h_t, l_t, b_t, log_probas, p = self.model(x, l_t, h_t, last=True)
                # log_pi.append(p)
                # baselines.append(b_t)
                # locs.append(l_t[0:9])

                # convert list to tensors and reshape
                baselines = torch.stack(baselines).transpose(1, 0)
                l_log_pi = torch.stack(l_log_pi).transpose(1, 0)
                #s_log_pi = torch.stack(s_log_pi).transpose(1, 0)
                log_probas = torch.stack(log_probas).transpose(1, 0) 
                #stop_signals = torch.stack(stop_signals).transpose(1, 0).squeeze(2)

                #process stop signals
                #up_through_stop = stop_signals
                #count = torch.arange(self.batch_size)
                #num_steps = torch.sum(stop_signals, dim=1).long()
                #up_through_stop[count,num_steps] += 1

                #extract log_probas at first stop signal
                #log_probas = log_probas[count,num_steps,:]


                #clip histories after stop signal
                #baselines = baselines * up_through_stop
                #l_log_pi = l_log_pi * up_through_stop
                #s_log_pi = s_log_pi * up_through_stop



                # calculate reward
                predicted = torch.max(log_probas, 2)[1]
                repeat_y = y.unsqueeze(1).repeat(1, self.num_glimpses)
                R = (predicted.detach() == repeat_y).float()
                #R = R.unsqueeze(1).repeat(1, self.num_glimpses)
                #mask = (torch.arange(R.size(1), device=num_steps.device)==num_steps.unsqueeze(1))
                #R = mask*R #Reward of 1 at first stop signal
                #R = R - stop_signals * self.hesitation_penalty


                # compute losses for differentiable modules
                #loss_action = F.nll_loss(log_probas, y)
                loss_action = F.nll_loss(log_probas.reshape(self.batch_size * self.num_glimpses, -1), repeat_y.reshape(self.batch_size*self.num_glimpses))
                loss_baseline = F.mse_loss(baselines, R)

                # compute reinforce loss
                # summed over timesteps and averaged across batch
                adjusted_reward = R - baselines.detach()
                loss_reinforce = torch.sum(-l_log_pi * adjusted_reward, dim=1) #+ torch.sum(-s_log_pi * adjusted_reward, dim=1)
                loss_reinforce = torch.mean(loss_reinforce, dim=0)

                # sum up into a hybrid loss
                loss = loss_action + loss_baseline + loss_reinforce * 0.01

                # compute accuracy
                correct = (predicted[:,-1] == y).float()
                acc = 100 * (correct.sum() / len(y))

                # store
                losses.update(loss.item(), x.size()[0])
                accs.update(acc.item(), x.size()[0])

                # compute gradients and update SGD
                loss.backward()
                self.optimizer.step()

                # measure elapsed time
                toc = time.time()
                batch_time.update(toc - tic)

                pbar.set_description(
                    (
                        "{:.1f}s - loss: {:.3f} - acc: {:.3f}".format(
                            (toc - tic), loss.item(), acc.item()
                        )
                    )
                )
                pbar.update(self.batch_size)

                # dump the glimpses and locs
                if plot:
                    imgs = [g.cpu().data.numpy().squeeze() for g in imgs]
                    locs = [l.cpu().data.numpy() for l in locs]
                    pickle.dump(
                        imgs, open(self.plot_dir + "g_{}.p".format(epoch + 1), "wb")
                    )
                    pickle.dump(
                        locs, open(self.plot_dir + "l_{}.p".format(epoch + 1), "wb")
                    )

                # log to tensorboard
                if self.use_tensorboard:
                    iteration = epoch * len(self.train_loader) + i
                    log_value("train_loss", losses.avg, iteration)
                    log_value("train_acc", accs.avg, iteration)

            return losses.avg, accs.avg

    @torch.no_grad()
    def validate(self, epoch):
        """Evaluate the RAM model on the validation set.
        """
        losses = AverageMeter()
        accs = AverageMeter()

        for i, (x, y) in enumerate(self.valid_loader):
            x, y = x.to(self.device), y.to(self.device, dtype=torch.int64)

            # duplicate M times
            x = x.repeat(self.M, 1, 1, 1)

            # initialize location vector and hidden state
            self.batch_size = x.shape[0]
            #h_t, l_t, s_t = self.reset()
            h_t, l_t = self.reset()

            # extract the glimpses
            l_log_pi = []
            #s_log_pi = []
            baselines = []
            log_probas = []
            #stop_signals = []
            for t in range(self.num_glimpses):
                # forward pass through model
                #h_t, l_t, s_t, b_t, log_ps, l_p, s_p = self.model(x, l_t, h_t, s_t)
                h_t, l_t, b_t, log_ps, l_p = self.model(x, l_t, h_t)

                # store
                baselines.append(b_t)
                l_log_pi.append(l_p)
                #s_log_pi.append(s_p)
                log_probas.append(log_ps)
                #stop_signals.append(s_t)
                

            # convert list to tensors and reshape
            baselines = torch.stack(baselines).transpose(1, 0)
            l_log_pi = torch.stack(l_log_pi).transpose(1, 0)
            #s_log_pi = torch.stack(s_log_pi).transpose(1, 0)
            log_probas = torch.stack(log_probas).transpose(1, 0) 
            #stop_signals = torch.stack(stop_signals).transpose(1, 0).squeeze(2)

            #process stop signals
            #up_through_stop = stop_signals
            #count = torch.arange(self.batch_size)
            #num_steps = torch.sum(stop_signals, dim=1).long()
            #up_through_stop[count,num_steps] += 1

            #extract log_probas at first stop signal
            #log_probas = log_probas[count,num_steps,:]


            #clip histories after stop signal
            #baselines = baselines * up_through_stop
            #l_log_pi = l_log_pi * up_through_stop
            #s_log_pi = s_log_pi * up_through_stop
            

            # average
            log_probas = log_probas.contiguous().view(self.M, -1, log_probas.shape[-2], log_probas.shape[-1])
            log_probas = torch.mean(log_probas, dim=0)


            baselines = baselines.contiguous().view(self.M, -1, baselines.shape[-1])
            baselines = torch.mean(baselines, dim=0)

            l_log_pi = l_log_pi.contiguous().view(self.M, -1, l_log_pi.shape[-1])
            l_log_pi = torch.mean(l_log_pi, dim=0)
            #s_log_pi = s_log_pi.contiguous().view(self.M, -1, s_log_pi.shape[-1])
            #s_log_pi = torch.mean(s_log_pi, dim=0)

            # calculate reward
            repeat_y = y.unsqueeze(1).repeat(1, self.num_glimpses)
            predicted = torch.max(log_probas, 2)[1]
            R = (predicted.detach() == repeat_y).float()
            #R = R.unsqueeze(1).repeat(1, self.num_glimpses)
            #mask = (torch.arange(R.size(1), device=num_steps.device)==num_steps.unsqueeze(1))
            #R = mask*R
            #R = R - stop_signals * self.hesitation_penalty


            # compute losses for differentiable modules
            loss_action = F.nll_loss(log_probas.reshape(self.batch_size * self.num_glimpses, -1), repeat_y.reshape(self.batch_size * self.num_glimpses))
            loss_baseline = F.mse_loss(baselines, R)

            # compute reinforce loss
            adjusted_reward = R - baselines.detach()
            loss_reinforce = torch.sum(-l_log_pi * adjusted_reward, dim=1)# + torch.sum(-s_log_pi * adjusted_reward, dim=1)
            loss_reinforce = torch.mean(loss_reinforce, dim=0)

            # sum up into a hybrid loss
            loss = loss_action + loss_baseline + loss_reinforce * 0.01

            # compute accuracy
            correct = (predicted[:,-1] == y).float()
            acc = 100 * (correct.sum() / len(y))

            # store
            losses.update(loss.item(), x.size()[0])
            accs.update(acc.item(), x.size()[0])

            # log to tensorboard
            if self.use_tensorboard:
                iteration = epoch * len(self.valid_loader) + i
                log_value("valid_loss", losses.avg, iteration)
                log_value("valid_acc", accs.avg, iteration)

        return losses.avg, accs.avg

    @torch.no_grad()
    def test(self):
        """Test the RAM model.

        This function should only be called at the very
        end once the model has finished training.
        """
        correct = 0

        # load the best checkpoint
        self.load_checkpoint(best=self.best)

        # removed image, final_softmax, hidden layer, softmax, final_persistant_softmax
        cols = ['image_id', 'image', 'timestep', 'num_dots', 'area', 
                'next_location', 'prediction', 
                'absolute_error']
        # changed naming to SMALL
        filename = self.model_name + "_SMALL.csv"
        test_path = os.path.join(self.model_dir, filename)
        
        
        for i, (x, y, a) in enumerate(self.test_loader):
            df_dict = {column_name : [] for column_name in cols}

            batch_size = x.shape[0]
            df_dict['image_id'].extend(sum([[image_id]*self.num_glimpses for image_id in range(i*batch_size, (i+1)*batch_size)], []))
            df_dict['timestep'].extend(sum([list(range(self.num_glimpses)) for image_id in range(i*batch_size, (i+1)*batch_size)], []))
            df_dict['image'].extend(x.repeat_interleave(self.num_glimpses, dim=0).cpu().tolist())
            repeat_y = y.repeat_interleave(self.num_glimpses)
            df_dict['num_dots'].extend(repeat_y.cpu().tolist())
            df_dict['area'].extend(a.repeat_interleave(self.num_glimpses).cpu().tolist())



            x, y = x.to(self.device), y.to(self.device, dtype=torch.int64)


            # initialize location vector and hidden state
            self.batch_size = x.shape[0]
            #h_t, l_t, s_t = self.reset()
            h_t, l_t = self.reset()

            # extract the glimpses
            l_ts = []
            h_ts = []
            output_log_probas = []
            #stop_signals = []
            #stop_log_probs = []
            for t in range(self.num_glimpses):
                # forward pass through model
                #h_t, l_t, s_t, b_t, log_ps, l_p, s_p = self.model(x, l_t, h_t, s_t)
                h_t, l_t, b_t, log_ps, l_p = self.model(x, l_t, h_t)

                # store
                h_ts.append(h_t)
                l_ts.append(l_t)
                output_log_probas.append(log_ps)
                #stop_signals.append(s_t)
                #stop_log_probs.append(s_p)

            # convert list to tensors and reshape
            output_log_probas = torch.stack(output_log_probas).transpose(1, 0) 
            h_ts = torch.stack(h_ts).transpose(1, 0)
            l_ts = torch.stack(l_ts).transpose(1, 0)
            #stop_log_probs = torch.stack(stop_log_probs).transpose(1, 0)
            #stop_signals = torch.stack(stop_signals).transpose(1, 0)

            stretched_output_log_probas = output_log_probas.reshape(batch_size*self.num_glimpses, -1)
            stretched_h_ts = h_ts.reshape(batch_size * self.num_glimpses, -1)
            stretched_l_ts = l_ts.reshape(batch_size * self.num_glimpses, -1)
            #stretched_stop_log_probs = stop_log_probs.reshape(batch_size * self.num_glimpses, -1)
            #stretched_stop_signals = stop_signals.reshape(batch_size * self.num_glimpses, -1)

            softmaxes = torch.exp(output_log_probas)
            stretched_softmaxes = softmaxes.reshape(batch_size * self.num_glimpses, -1)
            #stretched_stop_probs = torch.exp(stretched_stop_log_probs)
            # df_dict['softmax'].extend(stretched_softmaxes.cpu().tolist())
            #df_dict['stop_probability'].extend(stretched_stop_probs.squeeze(1).cpu().tolist())
            # df_dict['hidden_layer'].extend(stretched_h_ts.cpu().tolist())
            df_dict['next_location'].extend(stretched_l_ts.cpu().tolist())
            #stop_signals = stop_signals.squeeze(2)
            #df_dict['stop_signal'].extend(stretched_stop_signals.squeeze(1).cpu().tolist())


            #process stop signals
            #count = torch.arange(batch_size)
            #num_steps = torch.sum(stop_signals, dim=1).long()
            #print(f"num steps: {num_steps}")
            #df_dict['num_steps'].extend(num_steps.repeat_interleave(self.num_glimpses).cpu().tolist())

            #up_through_stop = stop_signals
            #up_through_stop[count,num_steps] += 1
            #final_persistant_mask = (up_through_stop == 0)
            #print(f"mask shape: {final_persistant_mask.shape}")
            #print(f"mask: {final_persistant_mask}")

            #extract output_log_probas at first stop signal
            #final_softmax = softmaxes[count,num_steps,:]
            #print(f"final soft shape: {final_softmax.shape}")
            #print(f"final soft: {final_softmax}")
            # df_dict['final_softmax'].extend(final_softmax.repeat_interleave(self.num_glimpses, dim = 0).cpu().tolist())
            #unsqueezed_final_persistant_mask = final_persistant_mask.unsqueeze(2)
            #repeated_final_softmax = final_softmax.unsqueeze(1).repeat(1,self.num_glimpses,1)
            #final_persistant_softmaxes = torch.where(unsqueezed_final_persistant_mask, repeated_final_softmax, softmaxes)
            # df_dict['final_persistant_softmax'].extend(final_persistant_softmaxes.reshape(batch_size*self.num_glimpses, -1).cpu().tolist())

            final_pred = output_log_probas[:,-1,:].data.max(1, keepdim=True)[1]
            #print(f"final pred: {final_pred}")
            #df_dict['final_prediction'].extend(final_pred.repeat_interleave(self.num_glimpses).cpu().tolist())
            correct += final_pred.eq(y.data.view_as(final_pred)).cpu().sum()

            stretched_predictions = stretched_output_log_probas.data.max(1, keepdim=True)[1].squeeze(1)
            df_dict['prediction'].extend(stretched_predictions.cpu().tolist())
            predictions = stretched_predictions.reshape(batch_size, self.num_glimpses)
            #repeated_final_pred = final_pred.repeat(1, self.num_glimpses)
            #final_persistant_predictions = torch.where(final_persistant_mask, repeated_final_pred, predictions)
            #stretched_final_persistant_predictions = final_persistant_predictions.reshape(batch_size*self.num_glimpses, -1)
            #print(f"stretched_final_persistant_predictions shape: {stretched_final_persistant_predictions.shape}")
            #print(f"stretched_final_persistant_predictions: {stretched_final_persistant_predictions}")
            #df_dict['final_persistant_prediction'].extend(stretched_final_persistant_predictions.squeeze(1).cpu().tolist())


            #print(f"stretched_pred/y devices: {stretched_predictions.device}, {repeat_y.device}")
            stretched_error = torch.abs(stretched_predictions - repeat_y.cuda())
            df_dict['absolute_error'].extend(stretched_error.cpu().tolist())
            #print(f"error : {df_dict['absolute_error']}")
            #final_error = torch.abs(final_pred - y.unsqueeze(1))
            #error = stretched_error.reshape(batch_size, self.num_glimpses)
            #repeated_final_error = final_error.repeat(1, self.num_glimpses)
            # print(f"shapes: {final_persistant_mask.shape}, {repeated_final_error.shape}, {error.shape}")
            #final_persistant_error = torch.where(final_persistant_mask, repeated_final_error, error.long())
            #stretched_final_persistant_error = final_persistant_error.reshape(batch_size*self.num_glimpses, -1)
            #print(f"stretched_final_persistant_error shape: {stretched_final_persistant_error.shape}")
            #print(f"stretched_final_persistant_error: {stretched_final_persistant_error}")
            #df_dict['final_persistant_absolute_error'].extend(stretched_final_persistant_error.squeeze(1).cpu().tolist())
            
            df = pd.DataFrame(df_dict)
            df.to_csv(test_path, mode='a', header=not os.path.exists(test_path))


        perc = (100.0 * correct) / (self.num_test)
        error = 100 - perc
        print(
            "[*] Test Acc: {}/{} ({:.2f}% - {:.2f}%)".format(
                correct, self.num_test, perc, error
            )
        )



    def save_checkpoint(self, state, is_best):
        """Saves a checkpoint of the model.

        If this model has reached the best validation accuracy thus
        far, a seperate file with the suffix `best` is created.
        """
        filename = self.model_name + "_ckpt.pth.tar"
        model_path = os.path.join(self.model_dir, filename)
        torch.save(state, model_path)
        if is_best:
            filename = self.model_name + "_model_best.pth.tar"
            shutil.copyfile(model_path, os.path.join(self.model_dir, filename))

    def load_checkpoint(self, best=False):
        """Load the best copy of a model.

        This is useful for 2 cases:
        - Resuming training with the most recent model checkpoint.
        - Loading the best validation model to evaluate on the test data.

        Args:
            best: if set to True, loads the best model.
                Use this if you want to evaluate your model
                on the test data. Else, set to False in which
                case the most recent version of the checkpoint
                is used.
        """
        print("[*] Loading model from {}".format(self.model_dir))

        filename = self.model_name + "_ckpt.pth.tar"
        if best:
            filename = self.model_name + "_model_best.pth.tar"
        model_path = os.path.join(self.model_dir, filename)
        model = torch.load(model_path)

        # load variables from checkpoint
        self.start_epoch = model["epoch"]
        self.best_valid_acc = model["best_valid_acc"]
        self.model.load_state_dict(model["model_state"])
        self.optimizer.load_state_dict(model["optim_state"])

        if best:
            print(
                "[*] Loaded {} checkpoint @ epoch {} "
                "with best valid acc of {:.3f}".format(
                    filename, model["epoch"], model["best_valid_acc"]
                )
            )
        else:
            print("[*] Loaded {} checkpoint @ epoch {}".format(filename, model["epoch"]))


    def save_checkpoint(self, state, is_best):
        """Saves a checkpoint of the model.

        If this model has reached the best validation accuracy thus
        far, a seperate file with the suffix `best` is created.
        """
        filename = self.model_name + "_ckpt.pth.tar"
        model_path = os.path.join(self.model_dir, filename)
        torch.save(state, model_path)
        if is_best:
            filename = self.model_name + "_model_best.pth.tar"
            shutil.copyfile(model_path, os.path.join(self.model_dir, filename))

    def load_checkpoint(self, best=False):
        """Load the best copy of a model.

        This is useful for 2 cases:
        - Resuming training with the most recent model checkpoint.
        - Loading the best validation model to evaluate on the test data.

        Args:
            best: if set to True, loads the best model.
                Use this if you want to evaluate your model
                on the test data. Else, set to False in which
                case the most recent version of the checkpoint
                is used.
        """
        print("[*] Loading model from {}".format(self.model_dir))

        filename = self.model_name + "_ckpt.pth.tar"
        if best:
            filename = self.model_name + "_model_best.pth.tar"
        model_path = os.path.join(self.model_dir, filename)
        model = torch.load(model_path)

        # load variables from checkpoint
        self.start_epoch = model["epoch"]
        self.best_valid_acc = model["best_valid_acc"]
        self.model.load_state_dict(model["model_state"])
        self.optimizer.load_state_dict(model["optim_state"])

        if best:
            print(
                "[*] Loaded {} checkpoint @ epoch {} "
                "with best valid acc of {:.3f}".format(
                    filename, model["epoch"], model["best_valid_acc"]
                )
            )
        else:
            print("[*] Loaded {} checkpoint @ epoch {}".format(filename, model["epoch"]))
