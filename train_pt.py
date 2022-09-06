import os
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn as nn
import yaml
from einops import rearrange
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from data import load_data
from model import build_model
from render import save_sign_video
from search import greedy
from tokenizer import (SimpleTokenizer, build_vocab_from_phoenix,
                       white_space_tokenizer)
from utils import postprocess


def load_yaml(fpath):
    with open(fpath) as f:
        config = yaml.safe_load(f)
    return config


class ProgressiveTransformer(nn.Module):
    def __init__(
        self,
        config,
        src_vocab = None,
        trg_vocab = None,
        **kwargs
    ):
        super().__init__()

        model = build_model(config, src_vocab = src_vocab, trg_vocab = trg_vocab)
        
        self.model = model

    def forward(
        self,
        text_input_ids,
        text_pad_mask,
        joint_inputs,
        joint_pad_mask,
        loss_mask,
        future_trg = None
    ):
        src_lengths = text_pad_mask.sum(-1)
        
        outs, _ = self.model(
            src = text_input_ids, 
            trg_input = joint_inputs,
            src_mask = text_pad_mask,
            src_lengths = src_lengths,
            trg_mask = joint_pad_mask
        )
        
        if future_trg != None:
            outs = outs[:, -(future_trg.size(1)):, :]
            loss = F.mse_loss(outs, future_trg, reduction = 'none')
            loss = loss.sum(-1)
            # loss.masked_fill_(~(loss_mask[:, -(future_trg.size(1)):]), 0.)
            loss = loss.mean()
        else:
            loss = F.mse_loss(outs, joint_inputs, reduction = 'none')
            loss = loss.sum(-1)
            # loss.masked_fill_(~(loss_mask), 0.)
            loss = loss.mean()
            
        return loss, outs

    def generate(
        self,
        text_input_ids,
        text_pad_mask,
        joint_inputs
    ):
        src_lengths = text_pad_mask.sum(-1)

        enc_outs, enc_hidden = self.model.encode(
            src = text_input_ids,
            src_length = src_lengths,
            src_mask = text_pad_mask
        )
        
        stacked_outs, _ = greedy(
            encoder_output = enc_outs,
            src_mask = text_pad_mask,
            embed = self.model.trg_embed,
            decoder = self.model.decoder,
            trg_input = joint_inputs,
            model = self.model
        )
        
        return stacked_outs

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('arslp')        
        parser.add_argument('--config_path', type = str, default = './Configs/Base.yaml')
        parser.add_argument('--min_seq_len', type = int, default = 32)
        return parent_parser


class ProgressiveTransformerModule(pl.LightningModule):
    def __init__(
        self,
        config_path,
        dataset_type,
        min_seq_len,
        num_save,
        train_path,
        valid_path,
        test_path,
        batch_size,
        num_workers,
        lr,
        save_vids,
        output_dir,
        **kwargs
    ):
        super().__init__()

        self.save_hyperparameters()

        self.dataset_type = dataset_type
        self.train_path = train_path
        self.valid_path = valid_path
        self.test_path = test_path

        self.min_seq_len = min_seq_len

        self.num_save = num_save
        self.batch_size = batch_size
        self.num_worker = num_workers
        self.lr = lr
        self.output_dir = output_dir

        text_vocab = build_vocab_from_phoenix(train_path, white_space_tokenizer, mode = 'text')
        text_tokenizer = SimpleTokenizer(white_space_tokenizer, text_vocab)

        config = load_yaml(config_path)

        model = ProgressiveTransformer(config, text_vocab)
        
        self.tokenizer = text_tokenizer
        self.model = model
        self.save_vids = save_vids
        
        self.future_prediction = config['model']['future_prediction']
        self.gaussian_noise = config['model']['gaussian_noise']
        
        self.std = torch.zeros(240)

    def setup(self, stage):
        # load dataset
        self.trainset, self.validset, self.testset = \
            load_data(
                self.dataset_type, 
                self.train_path, 
                self.valid_path, 
                self.test_path, 
                seq_len = -1, 
                min_seq_len = self.min_seq_len
            )
        
        print(f'[INFO] {self.dataset_type} dataset loaded with sequence length {self.min_seq_len}.')

        if self.dataset_type == 'how2sign':
            self.S = 3
        else:
            self.S = 1.5

    def _common_step(self, batch, stage, batch_idx):
        id, text, joint, future_trg = batch['id'], batch['text'], batch['joints'], batch['future_trg']

        text_input_ids, text_pad_mask = self.tokenizer.encode(
            text, 
            padding = True,
            add_special_tokens = True,
            device = self.device
        )

        text_pad_mask = rearrange(text_pad_mask, 'b t -> b () t').bool()

        loss_mask = [torch.ones(j.size(0), device = self.device) for j in joint]
        loss_mask = pad_sequence(loss_mask, batch_first = True)
        loss_mask = rearrange(loss_mask, 'b t -> b t')
        loss_mask = loss_mask.bool()
        
        joint_inputs = pad_sequence(joint, batch_first = True, padding_value = 0.)
        
        # follow the authors padding method for joint inputs
        joint_pad_mask = (joint_inputs != 0.0).unsqueeze(1)
        pad_amount = joint_inputs.shape[1] - joint_inputs.shape[2]
        joint_pad_mask = (F.pad(input=joint_pad_mask.double(), pad=(pad_amount, 0, 0, 0), mode='replicate') == 1.0)
        
        # future trg
        if self.future_prediction != 0:
            future_trg = pad_sequence(future_trg, batch_first = True, padding_value = 0.)

        # gaussian noise: follows the author method
        if self.gaussian_noise:
            if torch.sum(self.std) != 0.:
                self.model.out_stds = self.std
        
        loss, outs = self.model(
            text_input_ids = text_input_ids,
            text_pad_mask = text_pad_mask,
            joint_inputs = joint_inputs,
            joint_pad_mask = joint_pad_mask,
            loss_mask = loss_mask,
            future_trg = future_trg if self.future_prediction != 0 else None
        )
        
        self.log(f'{stage}/loss', loss, batch_size = self.batch_size)
        
        if stage == 'tr':
            if self.gaussian_noise:
                with torch.no_grad():
                    noise = outs.detach() - future_trg.detach()
                if self.future_prediction != 0:
                    noise = noise[:, :, :noise.shape[2] // (self.future_prediction)]
                noise = noise.cpu().detach()
            else:
                noise = None

            if self.gaussian_noise:
                if self.future_prediction != 0:
                    noise = rearrange(noise, 'b t d -> (b t) d')
                    noise_std = noise.std(dim = 0)
                    self.std += noise_std
                    if batch_idx != 0:
                        self.std /= 2

            return loss

        idx = torch.randperm(text_input_ids.size(0))[:self.num_save]
        
        generated = self.model.generate(
            text_input_ids = text_input_ids[idx],
            text_pad_mask = text_pad_mask[idx],
            joint_inputs = joint_inputs[idx]
        )

        generated = list(rearrange(gen[:, :-1], 't (v c) -> t v c', c = 2).cpu() for gen in generated)
        origin = list(rearrange(joint[i][:, :-1], 't (v c) -> t v c', c = 2).cpu() for i in idx)
        text = list(text[i] for i in idx)
        name = list(id[i] for i in idx)
        
        return {
            'loss': loss,
            'id': id,
            'name': name,
            'text': text,
            'origin': origin,
            'generated': generated
        }

    def _common_epoch_end(self, outputs, stage):
        H, W = 256, 256
        S = self.S

        output = outputs[0] # select only one example
    
        origin = output['origin']
        generated = output['generated']
        text = output['text']
        name = output['name']

        processed_origin, processed_generated = [], []
        for ori, gen in zip(origin, generated):
            processed_ori, processed_gen = map(lambda t: postprocess(t, H, W, S), [ori, gen])
            
            processed_origin.append(processed_ori)
            processed_generated.append(processed_gen)
    
        if self.current_epoch != 0:
            vid_save_path = os.path.join(self.logger.save_dir, self.logger.name, f'version_{str(self.logger.version)}', 'vid_outputs', str(self.global_step))
            
            if not os.path.exists(vid_save_path):
                os.makedirs(vid_save_path)
            
            for n, t, g, o in zip(name, text, processed_generated, processed_origin):
                save_sign_video(fpath = os.path.join(vid_save_path, f'{n}.mp4'), hyp = g, ref = o, sent = t, H = H, W = W)

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, 'tr', batch_idx)

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, 'val', batch_idx)

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, 'tst', batch_idx)

    def validation_epoch_end(self, outputs):
        return self._common_epoch_end(outputs, 'val')

    def test_epoch_end(self, outputs):
        H, W = 256, 256
        S = self.S

        id_list, text_list, generated_list, origin_list = [], [], [], []
        for output in outputs:
            id = output['id']
            text = output['text']
            generated = output['generated']
            origin = output['origin']

            id_list += id
            text_list += text
            generated_list += generated
            origin_list += origin

        if self.logger.save_dir != None:
            save_path = os.path.join(self.logger.save_dir, self.logger.name, f'version_{str(self.logger.version)}/test_outputs')
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            # save generated joint outputs
            outputs = {
                'outputs': generated_list,
                'reference': origin_list,
                'texts': text_list,
                'ids': id_list
            }

            torch.save(outputs, os.path.join(save_path, 'outputs.pt'))
            
            if not self.save_vids:
                return
            
            _iter = zip(origin_list[:self.num_save], generated_list[:self.num_save], id_list[:self.num_save], text_list[:self.num_save])
            for j, d, id, text in _iter:
                if len(j.size()) == 2:
                    j, d = map(lambda x: rearrange(x, 't (v c) -> t v c', c = 2), [j, d])
                
                origin = postprocess(j, H, W, S)
                generated = postprocess(d, H, W, S)
                
                save_sign_video(os.path.join(save_path, f'{id}.mp4'), generated, origin, text, H, W)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr = self.lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer = optim,
            mode = 'min',
            factor = 0.5,
            patience = 10,
            cooldown = 10,
            min_lr = 1e-6,
            verbose = True
        )
        
        return {
            'optimizer': optim,
            'lr_scheduler': {
                'scheduler': sched,
                'monitor': 'val/loss',
                'frequency': self.trainer.check_val_every_n_epoch
            },
        }

    def train_dataloader(self):
        return DataLoader(
            self.trainset, 
            batch_size = self.batch_size, 
            shuffle = True, 
            num_workers = self.num_worker, 
            collate_fn = self._collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.validset, 
            batch_size = self.batch_size, 
            shuffle = False, 
            num_workers = self.num_worker, 
            collate_fn = self._collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            self.testset, 
            batch_size = self.batch_size, 
            shuffle = False, 
            num_workers = self.num_worker, 
            collate_fn = self._collate_fn
        )

    def _collate_fn(self, batch):
        id_list, text_list, joint_list, future_trg_list = [], [], [], []
        
        sorted_batch = sorted(batch, key = lambda x: x['frame_len'], reverse = True)
        for data in sorted_batch:
            id_list.append(data['id'])
            text_list.append(data['text'])

            joints = rearrange(data['joint_feats'], 't v c -> t (v c)')

            # add invidual counter
            counter = torch.arange(0, len(joints)).float()
            counter -= torch.min(counter)
            counter /= torch.max(counter)
            counter = rearrange(counter, 't -> t ()')
            
            joints = torch.cat((joints, counter), dim = -1)

            # follows the authors method
            if self.future_prediction != 0:
                future_trg = torch.Tensor()
                # suspicious
                for i in range(0, self.future_prediction):
                    future_trg = torch.cat((future_trg, joints[i:-(self.future_prediction - i), :-1].clone()), dim = 1)
                future_trg = torch.cat((future_trg, joints[:-self.future_prediction,-1:]), dim = 1)
                future_trg_list.append(future_trg)
            
            joint_list.append(joints)
        
        return {
            'id': id_list,
            'text': text_list,
            'joints': joint_list,
            'future_trg': future_trg_list
        }

    def get_callback_fn(self, monitor = 'val/loss', patience = 50):
        early_stopping_callback = EarlyStopping(
            monitor = monitor, 
            patience = patience, 
            mode = 'min', 
            verbose = True
        )
        ckpt_callback = ModelCheckpoint(
            dirpath=self.output_dir,
            filename = 'epoch={epoch}-val_loss={val/loss:.2f}', 
            monitor = monitor, 
            save_last = True, 
            save_top_k = 1, 
            mode = 'min', 
            verbose = True,
            auto_insert_metric_name = False
        )
        lr_monitor = LearningRateMonitor(logging_interval='step')

        return early_stopping_callback, ckpt_callback, lr_monitor

    def get_logger(self, logger_type = 'tensorboard', name = 'slp'):
        if logger_type == 'tensorboard':
            logger = TensorBoardLogger("slp_logs", name = name)
        elif logger_type == 'wandb':
            logger = WandbLogger(project=name, save_dir=self.output_dir)
        else:
            raise NotImplementedError
        return logger


def main(hparams):
    pl.seed_everything(hparams.seed)
    
    module = ProgressiveTransformerModule(**vars(hparams))
    
    early_stopping, ckpt, lr_monitor = module.get_callback_fn('val/loss', 50)
    
    callbacks_list = [ckpt, lr_monitor]

    if hparams.use_early_stopping:
        callbacks_list.append(early_stopping)
    
    logger = module.get_logger('wandb', name = hparams.log_name)
    hparams.logger = logger
    
    trainer = pl.Trainer.from_argparse_args(hparams, callbacks = callbacks_list)

    if not hparams.test:
        trainer.fit(module)
    else:
        assert hparams.ckpt != None, 'Trained checkpoint must be provided.'
        trainer.test(module, ckpt_path = hparams.ckpt)


if __name__=='__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--seed', type = int, default = 42)
    parser.add_argument("--fast_dev_run", action = "store_true")
    parser.add_argument('--dataset_type', default = 'phoenix')
    parser.add_argument('--train_path', default = '/home/ejhwang/projects/phoenix14t/data/phoenix14t.pose.train')
    parser.add_argument('--valid_path', default = '/home/ejhwang/projects/phoenix14t/data/phoenix14t.pose.dev')
    parser.add_argument('--test_path', default = '/home/ejhwang/projects/phoenix14t/data/phoenix14t.pose.test')
    parser.add_argument("--batch_size", type = int, default = 32)
    parser.add_argument("--num_workers", type = int, default = 12)
    parser.add_argument("--max_epochs", type = int, default = 500)
    parser.add_argument('--check_val_every_n_epoch', type = int, default = 3)
    parser.add_argument('--accelerator', default = 'gpu')
    parser.add_argument('--devices', nargs = '+', type = int, default = [2])
    parser.add_argument('--strategy', default = None)
    parser.add_argument('--num_save', type = int, default = 3)
    parser.add_argument('--lr', type = float, default = 1e-3)
    parser.add_argument('--use_early_stopping', action = 'store_true', default=True)
    parser.add_argument('--gradient_clip_val', type = float, default = 0.0)
    parser.add_argument('--test', action = 'store_true')
    parser.add_argument('--save_vids', action = 'store_true')
    parser.add_argument('--log_name', type = str, default = 'pt-phoenix-base')
    parser.add_argument('--output_dir', default='output/')
    parser.add_argument('--ckpt', default = None)

    parser = ProgressiveTransformer.add_model_specific_args(parser)

    hparams = parser.parse_args()

    main(hparams)


# --test --ckpt output/last.ckpt --save_vids --num_save 5
