import multiprocessing
import os
import csv

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from utils_pytorch import EarlyStopReduceLROnPlateau, save_model, save_to_csv
from factories import ModelFactory, LossesFactory


def load_loss_history(filepath):
    """Loads the numeric rows from an existing two-column loss CSV."""
    history = []
    if not os.path.exists(filepath):
        return history

    with open(filepath, newline='') as csv_file:
        for row in csv.reader(csv_file):
            if len(row) < 2:
                continue
            try:
                history.append([int(float(row[0])), float(row[1])])
            except ValueError:
                # Skip the header (and any malformed rows) safely.
                continue
    return history


def load_best_history(filepath):
    """Loads (loss, iteration) pairs from best_loss.csv."""
    history = []
    if not os.path.exists(filepath):
        return history

    with open(filepath, newline='') as csv_file:
        for row in csv.reader(csv_file):
            if len(row) < 2:
                continue
            try:
                history.append((float(row[0]), int(float(row[1]))))
            except ValueError:
                continue
    return history


def initialize_csv(filepath, header):
    """Creates a CSV header without duplicating it when training resumes."""
    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        save_to_csv([header], filepath)



def learning_curves(training, validation, outfile):
    """Builds learning curves: training and validation losses along
    iterations.
    """
    plt.rcParams["figure.figsize"] = [16, 9]
    fig, ax1 = plt.subplots(nrows=1, ncols=1, sharex=True)
    assert isinstance(ax1, Axes)
    x, y1 = zip(*training)
    ax1.plot(x, y1, 'b', label='training')

    x, y1 = zip(*validation)
    ax1.plot(x, y1, 'r', label='validation')

    ax1.set_yscale('log')

    ax1.legend()

    fig.savefig(outfile)
    plt.close(fig)


class R2Vessels:

    def __init__(
        self,
        base_channels=64,
        in_channels=5,
        out_channels=3,
        num_iterations=5,
        model=None,
        gpu_id=None,
        criterion=None,
        base_criterion=None,
        learning_rate=1e-4,
        optimizer_name='AdamW',
        weight_decay=1e-4,
        amp=False,
        max_epochs=None,
        tracker=None,
    ):
        current = multiprocessing.current_process()
        self.process_id = str(current.pid)

        self.use_cuda = torch.cuda.is_available()

        if gpu_id is None:
            self.device = torch.device('cuda', 0)
            torch.cuda.set_device(0)
        else:
            self.device = torch.device('cuda', gpu_id)
            torch.cuda.set_device(gpu_id)

        ### Loss
        self.criterion_name = criterion
        losses_factory = LossesFactory()
        if base_criterion is not None:
            base_criterion = losses_factory.create_class(base_criterion)
            self.criterion = losses_factory.create_class(criterion, base_criterion=base_criterion)
        else:
            self.criterion = losses_factory.create_class(criterion)

        ### Model
        self.model_name = model
        self.model = ModelFactory().create_class(
            model,
            input_ch=in_channels,
            output_ch=out_channels,
            base_ch=base_channels,
            num_iterations=num_iterations
        )
        if self.use_cuda:
            self.model.cuda()

        ### Optimizer
        optimizer_class = {
            'Adam': torch.optim.Adam,
            'AdamW': torch.optim.AdamW,
        }[optimizer_name]
        self.optimizer = optimizer_class(
            self.model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=weight_decay,
        )
        self.amp_enabled = bool(amp and self.use_cuda)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)
        self.max_epochs = max_epochs
        self.tracker = tracker

        # Number of images presented to the network
        self.iter = 0


    def train_epoch(self, r2v_loader):
        total_loss = 0.0

        self.model.train()

        for sample in r2v_loader:
            data = sample[1]

            retino = data[0].cuda(non_blocking=True).requires_grad_(False)
            vessels = data[1].cuda(non_blocking=True).requires_grad_(False)
            mask = data[2].cuda(non_blocking=True).requires_grad_(False)
            skeletons = (
                data[3].cuda(non_blocking=True).requires_grad_(False)
                if len(data) > 3 else None
            )

            self.optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                predictions = self.model(retino)
                loss = self.criterion(predictions, vessels, mask, skeletons)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()

            self.iter += 1

        pattern = '\n|{}| [PID: {}, {}, {}] >> training epoch mean loss: {}'
        avg_loss = total_loss / len(r2v_loader)
        print(pattern.format(
            self.iter,
            self.process_id,
            self.model_name,
            self.criterion_name,
            avg_loss
        ))

        return [avg_loss]


    def test(self, r2v_dataloader, prefix_to_save=None):
        with torch.no_grad():
            total_loss = 0.0

            self.model.eval()

            for sample in r2v_dataloader:
                try:
                    k = sample[0].numpy()[0]
                except AttributeError:
                    k = sample[0][0]
                data = sample[1]

                retino = data[0].cuda(non_blocking=True)
                vessels = data[1].cuda(non_blocking=True)
                mask = data[2].cuda(non_blocking=True)
                skeletons = (
                    data[3].cuda(non_blocking=True)
                    if len(data) > 3 else None
                )

                with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                    predictions = self.model(retino)
                    loss = self.criterion(predictions, vessels, mask, skeletons)

                if prefix_to_save is not None:
                    self.criterion.save_predicted(predictions, prefix_to_save + str(k) + '.png')

                total_loss += loss.item()

            pattern = '|{}| [PID: {}, {}, {}] >> validation mean loss: {}'
            avg_loss = total_loss / len(r2v_dataloader)
            print(pattern.format(
                self.iter,
                self.process_id,
                self.model_name,
                self.criterion_name,
                avg_loss
            ))

            return [avg_loss]

    def training(
        self,
        train_loader,
        test_loader,
        path_to_save,
        init_iter=0,
        save_period=100,
        scheduler_patience=25,
        stopping_patience=100
    ):
        best_loss_path = os.path.join(path_to_save, 'best_loss.csv')
        train_loss_path = os.path.join(path_to_save, 'train_loss.csv')
        test_loss_path = os.path.join(path_to_save, 'test_loss.csv')

        # Preserve prior logs and learning curves when training resumes.
        initialize_csv(best_loss_path, ['best_loss', 'iter'])
        initialize_csv(train_loss_path, ['loss', 'iter'])
        initialize_csv(test_loss_path, ['loss', 'iter'])

        train_loss = list()
        test_loss = list()
        all_train_loss = load_loss_history(train_loss_path)
        all_test_loss = load_loss_history(test_loss_path)
        best_history = load_best_history(best_loss_path)

        scheduler = EarlyStopReduceLROnPlateau(
            self.optimizer,
            self.model,
            path_to_save,
            factor=0.1,
            patience=scheduler_patience,
            patience_stopping=stopping_patience,
            verbose=True,
            cooldown=0,
            threshold=0,
            min_lr=1e-8,
            eps=0
        )

        self.iter = init_iter

        last_model_path = os.path.join(path_to_save, 'generator_best.pth')
        print("looking for last_model_path: ", last_model_path)
        if os.path.exists(last_model_path):
            print("get last_model_path: ", last_model_path)
            self.model.load_state_dict(torch.load(last_model_path, map_location=self.device))
            saved_directory_iters = [
                int(f) for f in os.listdir(path_to_save)
                if os.path.isdir(os.path.join(path_to_save, f)) and f.isdigit()
            ]
            logged_iters = [row[0] for row in all_test_loss]
            best_iters = [iteration for _, iteration in best_history]
            self.iter = max(
                [init_iter] + saved_directory_iters + logged_iters + best_iters
            )

            if best_history:
                best_loss, best_iter = min(best_history, key=lambda item: item[0])
                scheduler.best = best_loss
                scheduler.num_bad_epochs = sum(
                    1 for iteration, _ in all_test_loss if iteration > best_iter
                )
                print(
                    "restored validation best: {} at iter {} ({} bad epochs)".format(
                        scheduler.best, best_iter, scheduler.num_bad_epochs
                    )
                )
            
        else:
            self.iter = init_iter
        print("continue training from iter: ", self.iter)
        epochs_completed = self.iter // max(len(train_loader), 1)
        test_count = epochs_completed
        while scheduler.training() and (
            self.max_epochs is None or epochs_completed < self.max_epochs
        ):
            save = (test_count % save_period == 0)

            if save:
                save_path = os.path.join(path_to_save, str(self.iter))
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                prefix_to_save = save_path+'/'
            else:
                prefix_to_save = None

            train_loss.append([self.iter] + self.train_epoch(train_loader))
            test_loss.append([self.iter] + self.test(test_loader, prefix_to_save))

            is_best = scheduler.step(test_loss[-1][1], self.iter)

            if self.tracker is not None:
                self.tracker.log(
                    {
                        'epoch': epochs_completed + 1,
                        'iteration': self.iter,
                        'train/loss': train_loss[-1][1],
                        'validation/loss': test_loss[-1][1],
                        'optimizer/learning_rate': self.optimizer.param_groups[0]['lr'],
                    },
                    step=epochs_completed + 1,
                )
                if is_best:
                    self.tracker.summary['best/validation_loss'] = test_loss[-1][1]
                    self.tracker.summary['best/iteration'] = self.iter

            if is_best:
                save_to_csv([[str(test_loss[-1][1]),str(self.iter)]],
                             best_loss_path)
                save_model(self.model, path_to_save + '/generator_best.pth')

            if save:
                save_to_csv(train_loss, train_loss_path)
                save_to_csv(test_loss, test_loss_path)
                all_train_loss += train_loss
                all_test_loss += test_loss
                train_loss = []
                test_loss = []
                learning_curves(all_train_loss, all_test_loss, path_to_save + '/learning_curves.svg')

            test_count += 1
            epochs_completed += 1

        if len(train_loss) > 0:
            save_to_csv(train_loss, train_loss_path)
            all_train_loss += train_loss
        if len(test_loss) > 0:
            save_to_csv(test_loss, test_loss_path)
            all_test_loss += test_loss
        save_model(self.model, path_to_save + '/generator_last.pth')
        if all_train_loss and all_test_loss:
            learning_curves(all_train_loss, all_test_loss, path_to_save + '/learning_curves.svg')
