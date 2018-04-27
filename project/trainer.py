from util import log
import torchvision
import argparse
import os
import torch
import time
from tqdm import tqdm
from tensorboardX import SummaryWriter
import numpy as np


class Trainer(object):
    def __init__(self, config):
        super(Trainer, self).__init__()
        self.config = config
        hyper_parameter_str = '_lr_' + str(config.learning_rate)
        train_dir = './train_dir/%s-%s-%s-%s' % (
            config.model,
            config.prefix,
            hyper_parameter_str,
            time.strftime("%Y%m%d-%H%M%S")
        )
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        log.info("Train Dir: %s" % (train_dir,))

        # create data loader
        self.batch_size = config.batch_size

        # create model
        model = Trainer.get_model_class(config.model)
        log.infov("Using Model class: %s" % (model, ))
        if self.config.cuda:
            self.model = model(config).cuda()
        else:
            self.model = model(config)

        # create optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate)
        self.writer = SummaryWriter()

    @staticmethod
    def get_model_class(model_name):
        if model_name == 'baseline':
            from model import BaseLine
            model = BaseLine
        elif model_name == 'rn':
            from model import RN
            model = RN
        else:
            raise ValueError(model_name)
        return model

    def train(self, train_loader, valid_loader):
        log.infov("Training Starts!")

        epochs = self.config.epochs
        loss = torch.nn.NLLLoss()

        total_count = 0
        for i in range(epochs):
            # train a epoch
            self.model.train()
            pbar = tqdm(enumerate(train_loader))
            count = 0
            epoch_loss = 0.0
            epoch_correct_predict = 0
            epoch_total_predict = 0
            for batch_idx, (img, q, a) in pbar:
                count += len(img)
                if self.config.cuda:
                    img, q, a = img.cuda(), q.cuda(), a.cuda()
                img = torch.autograd.Variable(img, requires_grad=True)
                q = torch.autograd.Variable(q, requires_grad=True)
                a = torch.autograd.Variable(a, requires_grad=False)
                predict_a = self.model.forward(img, q)
                # compute loss
                batch_loss = loss(predict_a, torch.max(a, 1)[1])
                # compute batch accuracy for train
                correct_prediction = torch.max(predict_a, 1)[1].eq(torch.max(a, 1)[1])
                epoch_correct_predict += (torch.sum(correct_prediction)).data.cpu().numpy()[0]
                epoch_total_predict += len(correct_prediction)
                epoch_loss += (batch_loss[0]).data.cpu().numpy()[0]
                batch_accuracy = torch.mean(correct_prediction.float())
                # update parameters
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()
                pbar.set_description(
                    "Epoch: {} -- Instances: {}/{} -- Loss: {:.3f} -- Accuracy: {:.3f}".format(
                        i+1, count, len(train_loader)*self.batch_size, batch_loss.data[0], batch_accuracy.data[0]
                    )
                )
            total_count += 1

            # compute validation accuracy
            self.model.eval()
            valid_correct_predict = 0
            valid_total_predict = 0
            for val_batch_idx, (img, q, a) in enumerate(valid_loader):
                if self.config.cuda:
                    img, q, a, = img.cuda(), q.cuda(), a.cuda()
                img = torch.autograd.Variable(img, requires_grad=False)
                q = torch.autograd.Variable(q, requires_grad=False)
                a = torch.autograd.Variable(a, requires_grad=False)
                predict_a = self.model.forward(img, q)

                # find correct prediction
                correct_prediction = torch.max(predict_a, 1)[1].eq(torch.max(a, 1)[1])
                valid_correct_predict += torch.sum(correct_prediction).data.cpu().numpy()[0]
                valid_total_predict += len(correct_prediction)

            print(
                "Epoch: {} -- Train Accuracy: {: .3f} -- Validation Accuracy: {: .3f}".format(
                    i+1, epoch_correct_predict/epoch_total_predict, valid_correct_predict/valid_total_predict
                )
            )
            # convert back
            self.model.train()
            # log info
            self.writer.add_scalar('loss/Epoch Loss', epoch_loss, total_count)
            self.writer.add_scalar('Accuracy/Epoch Train Accuracy', epoch_correct_predict/epoch_total_predict,
                                   total_count)
            self.writer.add_scalar('Accuracy/Epoch Valid Accuracy', valid_correct_predict/valid_total_predict,
                                   total_count)

            if not os.path.exists('model'):
                os.makedirs('model')
            torch.save({'state_dict': self.model.state_dict(), 'config': self.config}, 'model/checkpoint_epoch_{}.pth'.format(i))
            # test accuracy

        torch.save({'state_dict': self.model.state_dict(), 'config': self.config}, 'model/checkpoint_final.pth')

    def test(self, test_loader):
        self.model.eval()
        pbar = tqdm(enumerate(test_loader))
        correct_count = 0
        count = 0
        for batch_idx, (img, q, a) in pbar:
            if self.config.cuda:
                img, q, a = img.cuda(), q.cuda(), a.cuda()
            print(img.size(), q.size(), a.size())
            img = torch.autograd.Variable(img, requires_grad=False)
            q = torch.autograd.Variable(q, requires_grad=False)
            a = torch.autograd.Variable(a, requires_grad=False)

            predict_a = self.model.forward(img, q)

            # find correct predictions
            correct_predict = torch.max(predict_a, 1)[1].eq(torch.max(a, 1)[1])
            correct_count += torch.sum(correct_predict).data.cpu().numpy()[0]
            count += len(correct_predict)

        return correct_count/count

    def predict(self, img, q, a):
        q = q.reshape(1, -1)
        a = a.reshape(1, -1)
        img = np.stack([img])
        img, q, a = torch.from_numpy(img), torch.from_numpy(q), torch.from_numpy(a)
        print(img.size(), q.size(), a.size())
        if self.config.cuda:
            img, q, a = img.cuda(), q.cuda(), a.cuda()
        img = torch.autograd.Variable(img, requires_grad=False)
        q = torch.autograd.Variable(q, requires_grad=False)
        predict_a = self.model.forward(img, q)

        # find correct prediction
        predict_a = predict_a.data.cpu().numpy()
        return predict_a, a

    @staticmethod
    def load_model(path):
        state_dict = torch.load(path)
        trainer = Trainer(state_dict['config'])
        trainer.model.load_state_dict(state_dict['state_dict'])
        return trainer


def check_data_path(path):
    if os.path.isfile(os.path.join(path, 'data.hy')) and os.path.isfile(os.path.join(path, 'id.txt')):
        return True
    else:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64
    )
    parser.add_argument(
        '--model',
        type=str,
        default='rn',
        choices=['baseline', 'rn']
    )
    parser.add_argument(
        '--prefix',
        type=str,
        default='default'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None
    )
    parser.add_argument(
        '--dataset_path',
        type=str,
        default='Sort-of-CLEVR_default'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=5.0e-4
    )
    parser.add_argument(
        '--lr-weight-decay',
        action='store_true',
        default=False
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100
    )
    parser.add_argument(
        '--cuda',
        action='store_false',
        default=True
    )
    config = parser.parse_args()

    path = os.path.join('./DataGenerator/datasets', config.dataset_path)

    if check_data_path(path):
        import dataset as dataset
    else:
        raise ValueError(path)

    config.data_info = dataset.get_data_info()
    config.conv_info = dataset.get_conv_info()

    dataset_train, dataset_valid, dataset_test = dataset.create_default_splits(path)

    trainer = Trainer(config)

    log.warning("dataset: %s, learning_rate: %f" % (
        config.dataset_path,
        config.learning_rate
    ))
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4
    )
    valid_loader = torch.utils.data.DataLoader(
        dataset_valid,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4
    )
    test_loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4
    )
    trainer.train(train_loader, valid_loader)
    trainer = Trainer.load_model('model/checkpoint_final.pth')
    test_accuracy = trainer.test(test_loader)
    print("Test Accuracy: %.3f %%" % (test_accuracy*100, ))


if __name__ == '__main__':
    main()
