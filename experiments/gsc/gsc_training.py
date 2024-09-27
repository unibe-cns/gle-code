import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from sklearn.metrics import confusion_matrix


def train(params, MemoryClass, model, loss_fn, train_loader, optimizer, epoch, fn_out, use_le=True):
    model.train()
    n_correct = 0
    train_loss = 0
    running_loss = 0.0
    it = 0
    total_samples_cnt = 0


    if 'running_metrics' in params and params['running_metrics']:
        batch_size = train_loader.batch_size
        dataset = train_loader.dataset
        train_loader = tqdm(train_loader, unit="samples", unit_scale=train_loader.batch_size)
        train_loader.batch_size = batch_size
        train_loader.dataset = dataset
    for batch_idx, (data, target) in enumerate(train_loader):
        batch_loss = 0
        data = data.float()  # (batch_size, time)

        if params['use_cuda']:
            data, target = data.cuda(), target.cuda()

        if len(data.shape) > 3: # squeeze out CNN format
            data = data.squeeze()

        if len(data.shape) < 3: # insert a spatial dimension if it is missing
            data = data.unsqueeze(1)

        # implement sliding window and sum logits over windows
        memory = MemoryClass(data, **MemoryClass.kwargs)
        n_steps = len(memory)

        output_over_time = []
        for input in memory:
            input = input.reshape(input.shape[0], -1) # flatten all spatial dimensions to be one long input
            optimizer.zero_grad()
            if use_le:
                with torch.no_grad():
                    # calling the model automatically populates the gradients
                    output = model(input, target, beta=params['beta'])
                    loss = loss_fn(output, target)
            else:
                output = model(input)
                loss = loss_fn(output, target)
                loss.backward()

            output_over_time.append(output.detach())
            optimizer.step()

            # accumulate loss of batches over steps for running and final loss
            running_loss += loss.item()
            it += 1

            # average loss over steps for individual batch
            batch_loss += loss.item() / n_steps

        output_over_time = torch.stack(output_over_time, dim=1)  # (batch_size, n_steps, output_size)
        pred = output_over_time.sum(axis=1).argmax(dim=1, keepdim=True)
        n_correct += pred.eq(target.view_as(pred)).sum().item()

        if fn_out is not None and batch_idx % params['checkpoint_interval'] == 0:
            torch.save(model.state_dict(), fn_out.format(postfix=f'_{epoch}_{batch_idx}'))

        total_samples_cnt += len(target)
        if "running_metrics" in params and params['running_metrics']:

            # update the progress bar
            train_loader.set_postfix({
                'train running loss': "%.05f" % (running_loss / it),
                'train running acc': "%.02f%%" % (100*n_correct/total_samples_cnt)
            })


    # average loss over dataset
    train_loss = running_loss / it
    train_acc = 100. * n_correct / total_samples_cnt

    print('\nTrain: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        train_loss, n_correct, total_samples_cnt, train_acc))

    return train_loss, train_acc

def test(params, MemoryClass, model, loss_fn, test_loader, epoch=0, prefix='valid', lr_scheduler=None):
    model.eval()
    n_correct = 0
    test_loss = 0
    running_loss = 0.0
    it = 0
    total_samples_cnt = 0

    # collect for confusion matrix
    preds_list = []
    targets_list = []

    with torch.no_grad():
        if 'running_metrics' in params and params['running_metrics']:
            batch_size = test_loader.batch_size
            dataset = test_loader.dataset
            test_loader = tqdm(test_loader, unit="samples", unit_scale=test_loader.batch_size)
            test_loader.batch_size = batch_size
            test_loader.dataset = dataset
        for batch_idx, (data, target) in enumerate(test_loader):
            data = data.float()

            if params['use_cuda']:
                data, target = data.cuda(), target.cuda()

            if len(data.shape) > 3: # squeeze out CNN format
                data = data.squeeze()

            # MNIST1D is just (batch_size, time), so we need to add a spatial dimension
            if len(data.shape) < 3: # insert a spatial dimension if it is missing
                data = data.unsqueeze(1)

            memory = MemoryClass(data, **MemoryClass.kwargs)

            output_over_time = []
            for input in memory:
                input = input.reshape(input.shape[0], -1) # flatten all spatial dimensions to be one long input
                output = model(input)
                loss = loss_fn(output, target).item()
                output_over_time.append(output.detach())

                # accumulate loss of batches over steps for running and final loss
                running_loss += loss
                it += 1

            output_over_time = torch.stack(output_over_time, dim=1)  # (batch_size, n_steps, output_size)
            pred = output_over_time.sum(axis=1).argmax(dim=1, keepdim=True)
            n_correct += pred.eq(target.view_as(pred)).sum().item()

            preds_list.append(pred.detach().cpu().numpy())
            targets_list.append(target.view_as(pred).detach().cpu().numpy())

            total_samples_cnt += len(target)
            if 'running_metrics' in params and params['running_metrics']:

                # update the progress bar
                test_loader.set_postfix({
                    'test running loss': "%.05f" % (running_loss / it),
                    'test running acc': "%.02f%%" % (100*n_correct/total_samples_cnt)
                })


    # average loss over dataset
    test_loss = running_loss / it
    test_acc = 100. * n_correct / total_samples_cnt

    print('Evaluate on', prefix, 'set: Average loss per sample: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, n_correct, total_samples_cnt, test_acc))

    # plot confusion matrix to stdout (visible in neptune.ai under monitoring/stdout)
    preds = np.concatenate(preds_list)
    targets = np.concatenate(targets_list)
    print("Confusion matrix: ")
    print(confusion_matrix(preds, targets))

    return test_loss, test_acc


def gsc_run(params, memory, model, loss_fn, fn_out, train_data, val_data, test_data=None, optimizer=None, lr_scheduler=None, use_le=True):

    metrics = {metric: [] for metric in ['train_loss', 'train_acc', 'val_loss', 'val_acc', 'test_loss', 'test_acc']}

    # load data
    if not isinstance(train_data, DataLoader):
        train_loader = DataLoader(train_data, batch_size=params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_data, batch_size=params['batch_size_test'], shuffle=False)
    else:
        train_loader = train_data
        val_loader = val_data

    if test_data is not None:
        if not isinstance(test_data, DataLoader):
            test_loader = DataLoader(test_data, batch_size=params['batch_size_test'], shuffle=False)
        else:
            test_loader = test_data

    memory_type = memory

    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

    # evaluate model before training
    train_loss, train_acc = test(params, memory_type, model, loss_fn, train_loader, 0, prefix='train', lr_scheduler=lr_scheduler)
    val_loss, val_acc = test(params, memory_type, model, loss_fn, val_loader, 0, prefix='valid', lr_scheduler=lr_scheduler)
    if 'train_loss' in metrics:
        metrics['train_loss'].append(train_loss)
    if 'train_acc' in metrics:
        metrics['train_acc'].append(train_acc)
    if 'val_loss' in metrics:
        metrics['val_loss'].append(val_loss)
    if 'val_acc' in metrics:
        metrics['val_acc'].append(val_acc)

    print('Start training:')
    for epoch in range(1, params['epochs'] + 1):
        train_loss, train_acc = train(params, memory_type, model, loss_fn, train_loader, optimizer, epoch, fn_out, use_le=use_le)
        val_loss, val_acc = test(params, memory_type, model, loss_fn, val_loader, epoch=epoch, prefix='valid', lr_scheduler=lr_scheduler)
        if 'train_loss' in metrics:
            metrics['train_loss'].append(train_loss)
        if 'train_acc' in metrics:
            metrics['train_acc'].append(train_acc)
        if 'val_loss' in metrics:
            metrics['val_loss'].append(val_loss)
        if 'val_acc' in metrics:
            metrics['val_acc'].append(val_acc)

        if lr_scheduler is not None:
            lr_scheduler.step(val_loss)

    # test model on test set after training
    if test_loader is not None:
        test_loss, test_acc = test(params, memory_type, model, loss_fn, test_loader, epoch=epoch, prefix='test', lr_scheduler=lr_scheduler)
        if 'test_loss' in metrics:
            metrics['test_loss'].append(test_loss)
        if 'test_acc' in metrics:
            metrics['test_acc'].append(test_acc)

    if fn_out is not None:
        torch.save(model.state_dict(), fn_out.format(postfix=''))

    return metrics
