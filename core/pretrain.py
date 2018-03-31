"""Pre-train encoder and classifier for source dataset."""

import torch.nn as nn
import torch.optim as optim

import params
import os
import time
from utils import make_variable, save_model
from logger import Logger



def train_src(encoder, classifier, data_loader):
    """Train classifier for source domain."""
    ####################
    # 1. setup network #
    ####################

    # set train state for Dropout and BN layers
    encoder.train()
    classifier.train()

    # setup criterion and optimizer
    optimizer = optim.Adam(
        [{'params': encoder.parameters(), 'lr': 1e-4},
         {'params': classifier.parameters()}],
        lr=params.c_learning_rate)
    criterion = nn.CrossEntropyLoss()

    ####################
    # 2. train network #
    ####################
    logger = Logger(os.path.join(params.model_root,'enevts'))
    total_step = 0
    for epoch in range(params.num_epochs_pre):
        t0 = time.time()
        for step, (images, labels) in enumerate(data_loader):
            # make images and labels variable
            total_step = epoch * len(data_loader) + step + 1
            t1 = time.time()
            images = make_variable(images)
            labels = make_variable(labels.squeeze_())
            t2 = time.time()
            # zero gradients for optimizer
            optimizer.zero_grad()

            # compute loss for critic
            preds = classifier(encoder(images))
            loss = criterion(preds, labels)
            t3 = time.time()
            # optimize source classifier
            loss.backward()
            optimizer.step()
            t4 = time.time()

            load_time = t1 - t0
            c2g_time = t2 - t1
            forward_time = t3 - t2
            backward_time = t4 - t3
            total_time = t4 - t0
            # print step info
            if ((step + 1) % params.log_step_pre == 0):
                print("Epoch [{}/{}] Step [{}/{}]: loss={:.6f} batch time={:.5f}"
                      .format(epoch + 1, params.num_epochs_pre,
                              step + 1, len(data_loader),
                              loss.data[0], total_time))

                logger.scalar_summary(tag='train_loss', value=loss.data[0],
                                      step= total_step)
                # print("load data: {:.5f} cpu2gpu: {:.5f} forward: {:.5f}"
                #       "backward: {:.5f} total: {:.5f}"
                #       .format(load_time/params.batch_size, c2g_time/params.batch_size,
                #               forward_time/params.batch_size,
                #               backward_time/params.batch_size, total_time/params.batch_size))
            t0 = time.time()

        # eval model on train set
        if ((epoch + 1) % params.eval_epoch_pre == 0):
            loss, acc = eval_src(encoder, classifier, data_loader)
            encoder.train()
            classifier.train()

            info = {"train_loss_all": loss,
                    'train-acc': acc
                    }
            for tag, value in info.items():
                logger.scalar_summary(tag=tag, value=value,
                              step=total_step)

        # save model parameters
        if ((epoch + 1) % params.save_epoch_pre == 0):
            save_model(encoder, "ADDA-source-encoder-{}.pt".format(epoch + 1))
            save_model(
                classifier, "ADDA-source-classifier-{}.pt".format(epoch + 1))

    # save final model
    save_model(encoder, "ADDA-source-encoder-final.pt")
    save_model(classifier, "ADDA-source-classifier-final.pt")

    return encoder, classifier


def eval_src(encoder, classifier, data_loader):
    """Evaluate classifier for source domain."""
    # set eval state for Dropout and BN layers
    encoder.eval()
    classifier.eval()

    # init loss and accuracy
    loss = 0
    acc = 0

    # set loss function
    criterion = nn.CrossEntropyLoss()

    # evaluate network
    for (images, labels) in data_loader:
        images = make_variable(images, volatile=True)
        labels = make_variable(labels)

        preds = classifier(encoder(images))
        loss += criterion(preds, labels).data[0]

        pred_cls = preds.data.max(1)[1]
        acc += pred_cls.eq(labels.data).cpu().sum()

    loss /= len(data_loader)
    acc /= len(data_loader.dataset)

    print("Avg Loss = {}, Avg Accuracy = {:2%}".format(loss, acc))
    return loss, acc