from modules.utils import load_yaml, save_yaml, get_logger

from modules.earlystoppers import EarlyStopper
from modules.recorders import Recorder
from modules.datasets import QADataset
from modules.trainer import Trainer

from modules.optimizers import get_optimizer
from modules.metrics import get_metric
from modules.losses import get_loss
from models.utils import get_model

from transformers import ElectraTokenizerFast

from torch.utils.data import DataLoader
import torch

from datetime import datetime, timezone, timedelta
import numpy as np
import random
import os
import copy
import sys, getopt

import wandb

# Root directory
PROJECT_DIR = os.path.dirname(__file__)

# Load config
config_path = os.path.join(PROJECT_DIR, 'config', 'train_config.yml')
config = load_yaml(config_path)

# Train Serial
kst = timezone(timedelta(hours=9))
train_serial = datetime.now(tz=kst).strftime("%Y%m%d_%H%M%S")

# Recorder directory
DEBUG = config['TRAINER']['debug']
print(f'debug {DEBUG}')

RECORDER_DIR = os.path.join(PROJECT_DIR, 'results', 'train', train_serial)
os.makedirs(RECORDER_DIR, exist_ok=True)

# Data directory
DATA_DIR = config['DIRECTORY']['dataset']

# Seed
torch.manual_seed(config['TRAINER']['seed'])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(config['TRAINER']['seed'])
random.seed(config['TRAINER']['seed'])

# GPU
os.environ['CUDA_VISIBLE_DEVICES'] = str(config['TRAINER']['gpu'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Train:

    def __init__(self, argv):
        self.FILENAME = argv[0]
        self.pretrained_prefix = "monologg/"
        self.pretrained = config['TRAINER']['pretrained']
        self.learning_rate = config['TRAINER']['learning_rate']
        self.optimizer = config['TRAINER']['optimizer']
        self.epoch = config['TRAINER']['n_epochs']
        self.patience = config['TRAINER']['early_stopping_patience']
        self.wandb = config['LOGGER']['wandb']
        self.wandb_username = ""
        try:
            opts, etc_args = getopt.getopt(
                argv[1:], 
                "hp:l:o:e:p:w:u", 
                ["help", "pretrained=", "lr=", "opt=", "epoch=", "patience=", "wandb=", "username="]
                )
        except getopt.GetoptError:
            print(self.FILENAME, '-p <pretrained> \n-l <learning rate> \n-o <optimizer> \n-e <epoch> \n-p <patience> \n-w <wandb> \n-u <wandb username>')
            sys.exit(2)

        for opt, arg in opts:
            if opt in ("-h", "--help"):
                print(self.FILENAME, '-p <pretrained> \n-l <learning rate> \n-o <optimizer> \n-e <epoch> \n-p <patience> \n-w <wandb> \n-u <wandb username>')
                sys.exit(2)

            elif opt in ("-p", "--pretrained"):
                if arg in ('koelectra-small-v3-discriminator', 'koelectra-base-v3-discriminator', 'koelectra-base-v3-finetuned-korquad'):
                    self.pretrained = self.pretrained_prefix + arg
                else:
                    print("Choose one of pretrained model names below!")
                    print("'koelectra-small-v3-discriminator', 'koelectra-base-v3-discriminator', 'koelectra-base-v3-finetuned-korquad'")
                    sys.exit(2)

            elif opt in ("-l", "--lr"):
                self.learning_rate = float(arg)

            elif opt in ("-o", "--opt"):
                if arg in ("adam", "adamw", "sgd"):
                    self.optimizer = arg
                else:
                    print("Choose one of optimizer names below!")
                    print('"adam", "adamw", "sgd"')
                    sys.exit(2)

            elif opt in ('-e', '--epoch'):
                self.epoch = int(arg)

            elif opt in ('-p', '--patience'):
                self.patience = int(arg)

            elif opt in ('-w', '--wandb'):
                if arg == 'True':
                    self.wandb = True
                elif arg == 'False':
                    self.wandb = False

            elif opt in ('-u', '--username'):
                self.wandb_username = arg
        
        print("PRETRAINED MODEL:", self.pretrained)
        print("LEARNING RATE:", self.learning_rate)
        print("OPTIMIZER:", self.optimizer)
        print("EPOCH:", self.epoch)
        print("PATIENCE:", self.patience)
        if self.wandb:
            print(f"RECORD on WANDB USERNAME: {self.wandb_username}")

        print("SET LOGGER")
        self.logger = self.set_logger()
        print("LOAD DATA")
        self.tokenizer, self.train_dataloader, self.val_dataloader = self.load_data()
        print("SET MODEL")
        self.model = self.set_model()
        print("SET TRAINER")
        self.trainer, self.early_stopper, self.recorder = self.set_trainer(model=self.model)
        print("START TRAIN!")
        self.train()

    def set_logger(self):
        """
        00. Set Logger
        """
        logger = get_logger(name='train', dir_=RECORDER_DIR, stream=False)
        logger.info(f"Set Logger {RECORDER_DIR}")
        print('/'.join(logger.handlers[0].baseFilename.split('/')[:-1]))
        return logger

    def load_data(self):
        """
        01. Load data
        """
        # Load tokenizer

        tokenizer_dict = {'ElectraTokenizerFast': ElectraTokenizerFast}
        tokenizer = tokenizer_dict[config['TRAINER']['tokenizer']].from_pretrained(self.pretrained)

        # Load Dataset
        if not DEBUG and os.path.isfile(os.path.join(DATA_DIR,'train_dataset.pt')):
            train_dataset = torch.load(os.path.join(DATA_DIR,'train_dataset.pt'))
            val_dataset = torch.load(os.path.join(DATA_DIR,'val_dataset.pt'))
            self.logger.info("loaded existing .pt")
        else:
            train_dataset = QADataset(data_dir=os.path.join(DATA_DIR, 'train.json'), tokenizer = tokenizer, max_seq_len = 512, mode = 'train')
            val_dataset = QADataset(data_dir=os.path.join(DATA_DIR, 'train.json'), tokenizer = tokenizer, max_seq_len = 512, mode = 'val')
            if not DEBUG:
                torch.save(train_dataset, os.path.join(DATA_DIR,'train_dataset.pt'))
                torch.save(val_dataset, os.path.join(DATA_DIR,'val_dataset.pt'))
            self.logger.info("loaded data, created .pt")

        if DEBUG:
            print(len(train_dataset), len(val_dataset))
            for i in range(10):
                txt = train_dataset[i]['input_ids']
                start_idx = train_dataset[i]['start_positions']
                end_idx = train_dataset[i]['end_positions']
                print(tokenizer.decode(txt[start_idx:end_idx]))
        
        # DataLoader
        train_dataloader = DataLoader(dataset=train_dataset,
                                    batch_size=config['DATALOADER']['batch_size'],
                                    num_workers=config['DATALOADER']['num_workers'], 
                                    shuffle=config['DATALOADER']['shuffle'],
                                    pin_memory=config['DATALOADER']['pin_memory'],
                                    drop_last=config['DATALOADER']['drop_last'])
        val_dataloader = DataLoader(dataset=val_dataset,
                                    batch_size=config['DATALOADER']['batch_size'],
                                    num_workers=config['DATALOADER']['num_workers'], 
                                    shuffle=False,
                                    pin_memory=config['DATALOADER']['pin_memory'],
                                    drop_last=config['DATALOADER']['drop_last'])

        self.logger.info(f"Load data, train:{len(train_dataset)} val:{len(val_dataset)}")

        return tokenizer, train_dataloader, val_dataloader
        
    def set_model(self):
        """
        02. Set model
        """
        # Load model
        model_name = config['TRAINER']['model']
        model = get_model(model_name=model_name, pretrained=self.pretrained).to(device)
        return model

    def set_trainer(self, model):
        """
        03. Set trainer
        """
        # Optimizer
        optimizer = get_optimizer(optimizer_name=self.optimizer)
        optimizer = optimizer(params=model.parameters(),lr=self.learning_rate)

        # Loss
        loss = get_loss(loss_name=config['TRAINER']['loss'])
        
        # Metric
        metrics = {metric_name: get_metric(metric_name) for metric_name in config['TRAINER']['metric']}
        
        # Early stoppper
        early_stopper = EarlyStopper(patience=self.patience,
                                    mode=config['TRAINER']['early_stopping_mode'],
                                    logger=self.logger)
        # AMP
        if config['TRAINER']['amp'] == True:
            from apex import amp
            model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

        # Trainer
        trainer = Trainer(model=model,
                        optimizer=optimizer,
                        loss=loss,
                        metrics=metrics,
                        device=device,
                        logger=self.logger,
                        tokenizer=self.tokenizer,
                        amp=amp if config['TRAINER']['amp'] else None,
                        interval=config['LOGGER']['logging_interval'])
        
        """
        Logger
        """
        # Recorder
        recorder = Recorder(record_dir=RECORDER_DIR,
                            model=model,
                            optimizer=optimizer,
                            scheduler=None,
                            amp=amp if config['TRAINER']['amp'] else None,
                            logger=self.logger)

        # !Wandb
        if self.wandb == True:
            wandb_project_serial = 'gravylab ai contest'
            wandb.init(project=wandb_project_serial, dir=RECORDER_DIR, entity=self.wandb_username)
            wandb.run.name = train_serial
            wandb.config.update(config)
            wandb.watch(model)

        # Save train config
        save_yaml(os.path.join(RECORDER_DIR, 'train_config.yml'), config)

        return trainer, early_stopper, recorder

    def train(self):
        """
        04. TRAIN
        """
        # Train
        n_epochs = self.epoch
        for epoch_index in range(n_epochs):

            # Set Recorder row
            row_dict = dict()
            row_dict['epoch_index'] = epoch_index
            row_dict['train_serial'] = train_serial
            
            """
            Train
            """
            print(f"Train {epoch_index}/{n_epochs}")
            self.logger.info(f"--Train {epoch_index}/{n_epochs}")
            self.trainer.train(dataloader=self.train_dataloader, epoch_index=epoch_index, tokenizer=self.tokenizer, mode='train')
            
            row_dict['train_loss'] = self.trainer.loss_mean
            row_dict['train_elapsed_time'] = self.trainer.elapsed_time 
            
            for metric_str, score in self.trainer.score_dict.items():
                row_dict[f"train_{metric_str}"] = score
            self.trainer.clear_history()
            
            """
            Validation
            """
            print(f"Val {epoch_index}/{n_epochs}")
            self.logger.info(f"--Val {epoch_index}/{n_epochs}")
            self.trainer.train(dataloader=self.val_dataloader, epoch_index=epoch_index, tokenizer=self.tokenizer, mode='val')
            
            row_dict['val_loss'] = self.trainer.loss_mean
            row_dict['val_elapsed_time'] = self.trainer.elapsed_time 
            
            for metric_str, score in self.trainer.score_dict.items():
                row_dict[f"val_{metric_str}"] = score
            self.trainer.clear_history()
            
            """
            Record
            """
            self.recorder.add_row(row_dict)
            self.recorder.save_plot(config['LOGGER']['plot'])

            #!WANDB
            if self.wandb == True:
                wandb.log(row_dict)

            """
            Early stopper
            """
            early_stopping_target = config['TRAINER']['early_stopping_target']
            self.early_stopper.check_early_stopping(loss=row_dict[early_stopping_target])

            if self.early_stopper.patience_counter == 0:
                self.recorder.save_weight(epoch=epoch_index)
                best_row_dict = copy.deepcopy(row_dict)
            
            if self.early_stopper.stop == True:
                self.logger.info(f"Early stopped, counter {self.early_stopper.patience_counter}/{self.patience}")
                
                if self.wandb == True:
                    wandb.log(best_row_dict)
                break




if __name__ == '__main__': 
    """
    To Handle => 'RuntimeError: CUDA out of memory.'
    """
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    Train(sys.argv)
