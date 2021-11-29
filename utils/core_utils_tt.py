import numpy as np
import torch
from utils.utils_tt import *
import os
from datasets.dataset_generic_tt import save_splits
# from models.model_mil import MIL_fc, MIL_fc_mc
from models.model_clam_tt import TOAD_mtl
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc

class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super(Accuracy_Logger, self).__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)
    
    def log_batch(self, count, correct,c):
        self.data[c]["count"] += count
        self.data[c]["correct"] += correct
    
    def get_summary(self, c):
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]
        
        if count == 0: 
            acc = None
        else:
            acc = float(correct) / count
        
        return acc, correct, count

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.train_loss_min = np.Inf

    def __call__(self, epoch, train_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -train_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(train_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(train_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, train_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.train_loss_min:.6f} --> {train_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.train_loss_min = train_loss

def train(datasets, cur, args):
    """   
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter  # 可视化
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None

    print('\nInit train/test splits...', end=' ')
    train_split, test_split = datasets
    # save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Testing on {} samples".format(len(test_split)))

    #损失函数
    loss_fn = nn.CrossEntropyLoss()
    #loss_fn = nn.BCELoss()

    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}

    model = TOAD_mtl(**model_dict)
    
    model.relocate()
    print('Done!')
    print_network(model)

    print('\nInit optimizer ...', end=' ')
    #梯度下降法
    optimizer = get_optim(model, args)
    print('Done!')
    
    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing = args.testing, weighted = args.weighted_sample)
    datasize = len(train_split)
    test_loader = get_split_loader(test_split, testing = args.testing)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(patience = 20, stop_epoch=100, verbose = True)

    else:
        early_stopping = None
    print('Done!')

    for epoch in range(args.max_epochs):
        print("epoch:  ",epoch)
        train_loss = train_loop(epoch, model, train_loader, optimizer, args.n_classes, datasize, writer, loss_fn)
        stop = validate(cur, epoch,model,train_loss, early_stopping,  loss_fn, args.results_dir)
        
        if stop: 
            break

    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))

    _, train_error, train_auc, _= summary(model, train_loader, args.n_classes)
    print('Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))

    results_dict, test_error, test_auc, acc_logger = summary(model, test_loader, args.n_classes)
    print('Test error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))

    for i in range(args.n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

        if writer:
            writer.add_scalar('final/test_class_{}_acc'.format(i), acc, 0)

    if writer:
        writer.add_scalar('final/test_error', test_error, 0)
        writer.add_scalar('final/test_auc', test_auc, 0)
    
    writer.close()
    return results_dict, test_auc, 1-test_error

def train_loop(epoch, model, loader, optimizer, n_classes, datasize, writer = None, loss_fn = None,):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_error = 0.

    print('\n')
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    # for batch_idx, (data,label,gender...) in enumerate(loader):
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        # label = label.float().to(device)
        # gender = gender.to(device)

        optimizer.zero_grad()
        results_dict = model(data)
        # results_dict = model(data.gender...)
        
        logits, Y_prob, Y_hat  = results_dict['logits'], results_dict['Y_prob'], results_dict['Y_hat']
        #_, preds = torch.max(results_dict, 1)
        prob[batch_idx] = Y_prob.cpu().detach().numpy() 
        labels[batch_idx] = label.item()
        print("label and Y_prob ------------------->",label, Y_prob, "\n")
        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()
        
        #loss = loss_fn(results_dict, label)
        
        #loss.backward()
        #optimizer.step()
        
        #running_loss += loss.item() * data.size(0)
        #running_corrects += torch.sum((preds == data.data).int())
        
    #epoch_loss = running_loss / datasize
    #epoch_acc = running_corrects / datasize
    
    #print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))   
        
        train_loss += loss_value
        if (batch_idx + 1) % 5 == 0:
            print('batch {}, loss: {:.4f}, label: {}, bag_size: {}'.format(batch_idx, loss_value, label.item(), data.size(0)))
           
        error = calculate_error(Y_hat, label)
        train_error += error
        
        # backward pass
        loss.backward()
        #for name, parms in model.named_parameters():
        #    print("grad_value --->",parms.grad)
        # step
        optimizer.step()
        

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)
    auc = roc_auc_score(labels, prob[:, 1])

    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f} ,auc: {:.4f}'.format(epoch, train_loss, train_error, auc))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)

    return train_loss



   
def validate(cur, epoch, model,train_loss, early_stopping = None, loss_fn = None, results_dir=None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    if early_stopping:
        assert results_dir
        early_stopping(epoch, train_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False


def summary(model, loader, n_classes):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    #for batch_idx, (data, label,gender...) in enumerate(loader):
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        # gender = gender.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            #results_dict = model(data,gender...)
            results_dict = model(data)
            logits, Y_prob, Y_hat = results_dict['logits'], results_dict['Y_prob'], results_dict['Y_hat']
            del results_dict

        acc_logger.log(Y_hat, label)
        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        error = calculate_error(Y_hat, label)
        test_error += error

    test_error /= len(loader)

    if n_classes == 2:
        print("all_labels", all_labels)
        print("all_probs[:, 1]", all_probs[:, 1])
        auc = roc_auc_score(all_labels, all_probs[:, 1])
    else:
        aucs = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

    return patient_results, test_error, auc, acc_logger
