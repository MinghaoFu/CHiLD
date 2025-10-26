from data_provider.data_factory import data_provider
from experiments.exp_basic import Exp_Basic
from utils.metrics import metric
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from utils.tools import adjust_learning_rate
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')

from utils.context_fid import Context_FID
from utils.cross_correlation import CrossCorrelLoss
from sklearn.preprocessing import MinMaxScaler
from utils.model_utils import unnormalize_to_zero_to_one, normalize_to_neg_one_to_one


class Exp(Exp_Basic):
    def __init__(self, args):
        super(Exp, self).__init__(args)
        self.args = args
        self.scaler = MinMaxScaler()

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag, draw=0):
        data_set, data_loader = data_provider(self.args, flag, draw)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate,
                                 weight_decay=self.args.weight_decay)
        return model_optim

    def _select_criterion(self):
        if self.args.metric == 'mse':
            criterion = nn.MSELoss()
        elif self.args.metric == 'mae':
            criterion = F.l1_loss
        else:
            raise NotImplementedError(f'Metric {self.args.metric} is not implemented')
        return criterion

    def train_test(self, epoch, data, data_loader, criterion, setting, mode):
        total_loss = []
        self.model.eval()
        true_list = []
        fake_list = []
        with torch.no_grad():
            for i, (batch_x, batch_x_mark) in enumerate(data_loader):
                if self.args.data == 'Human' or self.args.data == 'Humaneva' or self.args.data == 'Human_All':
                    batch_size, d1, d2, d3 = batch_x.shape
                    batch_x = batch_x.float().to(self.device).view(batch_size, d1, d2 * d3)
                else:
                    batch_x = batch_x.float().to(self.device)

                outputs = self.model(batch_x, is_train=False)

                true_list.append(batch_x.detach().cpu())
                fake_list.append(outputs.detach().cpu())

                pred = outputs.detach().cpu()
                true = batch_x.detach().cpu()
                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        true_list = np.concatenate(true_list, axis=0)
        fake_list = np.concatenate(fake_list, axis=0)
        ori_data = true_list
        gen_data = fake_list

        output_path = os.path.join(self.args.output, self.args.model)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
        data_name = self.args.data_path.split('.')[0]
        if self.args.is_norm:
            ori_data = self.normalize(ori_data)
            gen_data = self.normalize(gen_data)
            gen_data = self.normalize(
                gen_data)  # Secondary normalization to prevent errors caused by data being too small
        print(f'Shape of original data: {ori_data.shape}, generated data: {gen_data.shape}')
        print(
            f'Min and max of original data: {np.min(ori_data)}, {np.max(ori_data)}, generated data: {np.min(gen_data)}, {np.max(gen_data)}')

        # ------------------------------------------  start  ------------------------------------------
        print(f"\n\n-----metric--name:{data_name}--epoch:{epoch}--setting:{setting}")
        self.write_result(output_path, data_name,
                          f"-----metric--name:{data_name}--epoch:{epoch}--setting:{setting}--mode:{mode}\n\ttotal_loss:{total_loss:.4f}",
                          mode=mode)
        metric_results = dict()
        iter_num = 5

        print(
            "-------------------------------------------  start calculating Context fid score  -------------------------------------------")
        Context_FID_score = list()
        for i in range(iter_num):
            temp_fid = Context_FID(ori_data, gen_data)
            Context_FID_score.append(temp_fid)
            print(f'Context_FID score:    {temp_fid}')
        metric_results['Context_FID_mean'] = np.mean(Context_FID_score)
        metric_results['Context_FID_std'] = np.std(Context_FID_score)

        contf_score = f'Context fid score: {metric_results["Context_FID_mean"]:.3f} ± {metric_results["Context_FID_std"]:.3f}'
        print(f'Context fid score: {metric_results["Context_FID_mean"]} ± {metric_results["Context_FID_std"]}')
        self.write_result(output_path, data_name, contf_score, mode=mode)

        print(
            "-------------------------------------------  start calculating Correlational score  -------------------------------------------")
        Cross_correlation_score = list()
        x_real = torch.from_numpy(ori_data)
        x_fake = torch.from_numpy(gen_data)
        for i in range(iter_num):
            idx = np.random.randint(0, x_real.shape[0], x_real.shape[0])
            temp_cross = CrossCorrelLoss(x_real[idx, :, :]).to(self.device).compute(x_fake[idx, :, :])
            Cross_correlation_score.append(temp_cross)
            print(f'Cross_correlation_score score: {temp_cross}')
        metric_results['Cross_correlation_score'] = np.mean(Cross_correlation_score)
        metric_results['Cross_correlation_score_std'] = np.std(Cross_correlation_score)

        corr_score = f'Correlational score:  {metric_results["Cross_correlation_score"]:.3f} ± {metric_results["Cross_correlation_score_std"]:.3f}'
        print(
            f'Correlational score:  {metric_results["Cross_correlation_score"]} ± {metric_results["Cross_correlation_score_std"]}')
        self.write_result(output_path, data_name, corr_score, mode=mode)
        if mode == 'test':
            self.write_result(output_path, data_name, result='\n\n', mode='vali')
            self.write_result(output_path, data_name, result='\n\n', mode='test')
        del x_real, x_fake
        print(metric_results)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        output_path = os.path.join(self.args.output, self.args.model)
        data_name = self.args.data_path.split('.')[0]
        self.write_result(output_path, data_name, f'args:{self.args}', mode='vali')
        self.write_result(output_path, data_name, f'args:{self.args}', mode='test')
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        else:
            scaler = None
        best_loss = np.inf
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            true_list = []
            fake_list = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_x_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                if self.args.data == 'Human' or self.args.data == 'Humaneva' or self.args.data == 'Human_All':
                    batch_size, d1, d2, d3 = batch_x.shape
                    batch_x = batch_x.float().to(self.device).view(batch_size, d1, d2 * d3)
                else:
                    batch_x = batch_x.float().to(self.device)

                outputs, other_loss = self.model(batch_x, is_train=True)

                true_list.append(batch_x.detach().cpu())
                fake_list.append(outputs.detach().cpu())
                rec_loss = criterion(outputs, batch_x)
                loss = rec_loss + other_loss
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | rec_loss: {2:.7f}".format(i + 1, epoch + 1, rec_loss.item()))
                    print("\titers: {0}, epoch: {1} | total_loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    print(f'\tkld_loss: {other_loss:.10f}, raw kld_loss: {other_loss / self.args.kld_weight:.5f}')
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
            vali_loss = 0
            if (epoch + 1) % self.args.test_epoch == 0:
                vali_loss = self.train_test(epoch + 1, vali_data, vali_loader, criterion, setting, 'vali')
                if vali_loss < best_loss:
                    best_loss = vali_loss
                    torch.save(self.model.state_dict(), f'{path}/checkpoint.pth')

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = np.average(vali_loss)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            # adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(str(best_model_path), map_location='cuda'))
        test_loss = self.train_test(self.args.train_epochs, test_data, test_loader, criterion, setting, 'test')
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test', draw=1)
        if test:
            print('loading model')
            self.model.load_state_dict(
                torch.load(os.path.join(self.args.checkpoints + setting, 'checkpoint.pth'), map_location='cuda:0'))

        preds = []
        trues = []

        y_list = []
        y_hat_list = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_x_mark) in enumerate(test_loader):

                if self.args.data == 'Human' or self.args.data == 'Humaneva' or self.args.data == 'Human_All':
                    batch_size, d1, d2, d3 = batch_x.shape
                    batch_x = batch_x.float().to(self.device).view(batch_size, d1, d2 * d3)
                else:
                    batch_x = batch_x.float().to(self.device)

                outputs = self.model(batch_x, is_train=False)

                outputs = outputs.detach().cpu().numpy()
                batch_x = batch_x.detach().cpu().numpy()

                y_list.append(batch_x[:, 0, :])
                y_hat_list.append(outputs[:, 0, :])

                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_x = test_data.inverse_transform(batch_x.squeeze(0)).reshape(shape)
                pred = outputs
                true = batch_x

                preds.append(pred)
                trues.append(true)
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

    def write_result(self, path, dataset, result, mode):
        path = os.path.join(path, self.args.data)
        if not os.path.exists(path):
            os.makedirs(path)
        if self.args.filename is not None:
            file_name = f'{dataset}_{str(self.args.layer)}_{self.args.filename}_{mode}.txt'
        else:
            file_name = f'{dataset}_{str(self.args.layer)}_{mode}.txt'
        file_path = os.path.join(str(path), file_name)
        with open(file_path, 'a') as f:
            f.write(result + "\n")

    def normalize(self, sq):
        # sq [B,L,D]
        d = sq.reshape(-1, sq.shape[-1])  # [BL,D]
        self.scaler.fit(d)
        d = self.scaler.transform(d)
        d = normalize_to_neg_one_to_one(d)
        d = unnormalize_to_zero_to_one(d)
        return d.reshape(sq.shape)
