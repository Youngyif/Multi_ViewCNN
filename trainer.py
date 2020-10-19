import torch
import torch.nn as nn
from torch.autograd import Variable
# from models.model_mscale import *##0.84 auc model
from models.modelDifine import  *
import torch.autograd
import time
import math
from saveModel.graphgen import *
import numpy as np
from sklearn import metrics
import pandas as pd
import collections
import matplotlib.pyplot as plt
from models.contrastiveLoss import *
from models.nt_xent  import *
from models.Focal_loss_sigmoid  import *

single_train_time = 0
single_test_time = 0
single_train_iters = 0
single_test_iters = 0


def svb(m):
    eps = 0.5
    for layer in m.modules ():
        if isinstance (layer, nn.Conv2d):
            w_size = layer.weight.data.size ()
            layer_weight = (layer.weight.data.view (w_size[0], w_size[1] * w_size[2] * w_size[3])).cpu ()
            U, S, V = torch.svd (layer_weight)
            S = S.clamp (1.0 / (1 + eps), 1 + eps)
            layer_weight = torch.mm (torch.mm (U, torch.diag (S)), V.t ())
            layer.weight.data.copy_ (layer_weight.view (w_size[0], w_size[1], w_size[2] * w_size[3]))


def bbn(m):
    eps = 1.0
    for layer in m.modules ():
        if isinstance (layer, nn.BatchNorm2d):
            std = torch.sqrt (layer.running_var + layer.eps)
            alpha = torch.mean (layer.weight.data / std)
            low_bound = (std * alpha / (1 + eps)).cpu ()
            up_bound = (std * alpha * (1 + eps)).cpu ()
            layer_weight_cpu = layer.weight.data.cpu ()
            layer_weight = layer_weight_cpu.numpy ()
            layer_weight.clip (low_bound.numpy (), up_bound.numpy ())
            layer.weight.data.copy_ (torch.Tensor (layer_weight))


def getlearningrate(epoch, opt):
    # update lr
    lr = opt.LR
    if opt.lrPolicy == "multistep":
        if epoch + 1.0 >= opt.nEpochs * opt.ratio[1]:  # 0.6 or 0.8
            lr = opt.LR * 0.01
        elif epoch + 1.0 >= opt.nEpochs * opt.ratio[0]:  # 0.4 or 0.6
            lr = opt.LR * 0.1
    elif opt.lrPolicy == "linear":
        k = (0.001 - opt.LR) / math.ceil (opt.nEpochs / 2.0)
        lr = k * math.ceil ((epoch + 1) / opt.step) + opt.LR
    elif opt.lrPolicy == "exp":
        power = math.floor ((epoch + 1) / opt.step)
        lr = lr * math.pow (opt.gamma, power)
    else:
        assert False, "invalid lr policy"

    return lr


def computetencrop(outputs, labels):
    output_size = outputs.size ()
    outputs = outputs.view (output_size[0] / 10, 10, output_size[1])
    outputs = outputs.sum (1).squeeze (1)
    # compute top1
    _, pred = outputs.topk (1, 1, True, True)
    pred = pred.t ()
    top1_count = pred.eq (labels.data.view (1, -1).expand_as (pred)).view (-1).float ().sum (0)
    top1_error = 100.0 - 100.0 * top1_count / labels.size (0)
    top1_error = float (top1_error.cpu ().numpy ())

    # compute top5
    _, pred = outputs.topk (5, 1, True, True)
    pred = pred.t ()
    top5_count = pred.eq (labels.data.view (1, -1).expand_as (pred)).view (-1).float ().sum (0)
    top5_error = 100.0 - 100.0 * top5_count / labels.size (0)
    top5_error = float (top5_error.cpu ().numpy ())
    return top1_error, 0, top5_error


def computeresult(outputs, labels, loss, top5_flag=False):
    if isinstance (outputs, list):
        top1_loss = []
        top1_error = []
        top5_error = []
        for i in range (len (outputs)):
            # get index of the max log-probability
            predicted = outputs[i].data.max (1)[1]
            top1_count = predicted.ne (labels.data).cpu ().sum ()
            top1_error.append (100.0 * top1_count / labels.size (0))
            top1_loss.append (loss[i].data[0])
            if top5_flag:
                _, pred = outputs[i].data.topk (5, 1, True, True)
                pred = pred.t ()
                top5_count = pred.eq (labels.data.view (1, -1).expand_as (pred)).view (-1).float ().sum (0)
                single_top5 = 100.0 - 100.0 * top5_count / labels.size (0)
                single_top5 = float (single_top5.cpu ().numpy ())
                top5_error.append (single_top5)

    else:
        # get index of the max log-probability
        predicted = outputs.data.max (1)[1]
        top1_count = predicted.ne (labels.data).cpu ().sum ()
        top1_error = 100.0 * top1_count / labels.size (0)
        top1_loss = loss.data[0]
        top5_error = 100.0
        if top5_flag:
            _, pred = outputs.data.topk (5, 1, True, True)
            pred = pred.t ()
            top5_count = pred.eq (labels.data.view (1, -1).expand_as (pred)).view (-1).float ().sum (0)
            top5_error = 100.0 - 100.0 * top5_count / labels.size (0)
            top5_error = float (top5_error.cpu ().numpy ())

    if top5_flag:
        return top1_error, top1_loss, top5_error
    else:
        return top1_error, top1_loss, top5_error


def computeAUC(outputs, labels, epoch):
    if isinstance (outputs, list):
        pred = np.concatenate (outputs, axis=0)
        y = np.concatenate (labels, axis=0)
    else:
        pred = outputs
        y = labels
    fpr, tpr, thresholds = metrics.roc_curve (y, pred, pos_label=1)
    roc_auc = metrics.auc (fpr, tpr)
    if np.isnan (roc_auc):
        roc_auc = 0
    # roc_auc = metrics.auc(fpr, tpr)

    if opt.draw_ROC == True:
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label='AUC = %0.4f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.savefig(str(epoch)+"epoch_RocOf"+opt.experimentID+".png")
    return roc_auc, fpr, tpr

def readwrongpath(lists):
    res=[]
    for i in lists:
        for j in i:
            res.append(j)
    return res

def one_to_quarter(dicts):
    detailist=[]
    prolist = []
    for key in dicts:
        probability = dicts[key]
        detailist.append(key)
        prolist.append(probability)
    dicts_csv = {"details":detailist,"probability":prolist}
    df_csv = pd.DataFrame(dicts_csv)
    df_csv.to_csv("RU_probability_statistic_CONTRA_focal1.csv")








def computeEval(outputs, labels, pathlist):
    dicts = {}
    wronglist=[]
    path = readwrongpath(pathlist)
    if isinstance (outputs, list):
        pred = np.concatenate (outputs, axis=0)
        y = np.concatenate (labels, axis=0)
    else:
        pred = outputs
        y = labels
    prolist = pred.copy()
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    for i in range(len(pred)):
        dicts[path[i]] = prolist[i]
        if pred[i]!=y[i]:
            wronglist.append(path[i])
    one_to_quarter(dicts)
    # acc
    acc = metrics.accuracy_score (y, pred)
    # tn, fp, fn, tp
    tn, fp, fn, tp = metrics.confusion_matrix (y, pred).ravel ()
    # precision
    precision = np.nan if (tp + fp) == 0 else float (tp) / (tp + fp)
    # recall
    recall = np.nan if (tp + fn) == 0 else float (tp) / (tp + fn)
    # F1
    f1 = metrics.f1_score (y, pred, pos_label=1, average='binary')
    # g-mean
    specificity = np.nan if (tn + fp) == 0 else float (tn) / (tn + fp)
    gmean = math.sqrt (recall * specificity)
    print ("tn, fp, fn, tp")
    print (tn, fp, fn, tp)
    return acc, precision, recall, f1, gmean, tn, fp, fn, tp, wronglist


def printresult(epoch, nEpochs, count, iters, lr, data_time, iter_time, loss, mode="Train"):
    global single_train_time, single_test_time
    global single_train_iters, single_test_iters

    log_str = ">>> %s [%.3d|%.3d], Iter[%.3d|%.3d], DataTime: %.4f, IterTime: %.4f, lr: %.4f" \
              % (mode, epoch + 1, nEpochs, count, iters, data_time, iter_time, lr)

    # compute cost time
    if mode == "Train":
        single_train_time = single_train_time * 0.95 + 0.05 * (data_time + iter_time)
        # single_train_time = data_time + iter_time
        single_train_iters = iters
    else:
        single_test_time = single_test_time * 0.95 + 0.05 * (data_time + iter_time)
        # single_test_time = data_time+iter_time
        single_test_iters = iters
    total_time = (single_train_time * single_train_iters + single_test_time * single_test_iters) * nEpochs
    time_str = ",Cost Time: %d Days %d Hours %d Mins %.4f Secs" % (total_time // (3600 * 24),
                                                                   total_time // 3600.0 % 24,
                                                                   total_time % 3600.0 // 60,
                                                                   total_time % 60)
    print (log_str + time_str)


def writeData(x, y):
    path_name = 'test.csv'
    # df_new = pd.DataFrame({'fpr':x, 'tpr':y})
    # df = pd.read_csv(path_name)
    # result = pd.concat([df, df_new], axis=1)
    # result.to_csv(path_name, index=False)


def writeDiseaseType(x, y):
    path_name = 'disease_type.csv'
    df = pd.DataFrame ({'realLabel': x, 'predictLabel': y})
    df['isTrue'] = df.apply (lambda x: ('True' if (x['realLabel'] == x['predictLabel']) else 'False'), axis=1)
    df.to_csv (path_name, index=False)


def generateTarget(images, labels):
    target_disease = torch.LongTensor (labels.size (0)).zero_ () + int (1)
    reduce_labels = labels == target_disease
    # reduce_labels = reduce_labels.type_as (images)
    reduce_labels = reduce_labels.float()
    return reduce_labels



class L1norm(torch.nn.Module): ##loss_testure
    def __init__(self):
        super(L1norm,self).__init__()

    def forward(self,x):
        out = 0
        count = 0
        for i in range(x.size(0)):
            xt = x[i, ...]
            out += torch.sum(torch.abs(xt)).sum()
            count += self._tensor_size(xt)
        return out/count

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]*t.size()[0]


class Trainer (object):
    realLabelsarr = []
    predictLabelsarr = []

    def __init__(self, model, opt, optimizer=None):
        self.opt = opt
        self.model = model
        # print (model)
        if self.opt.trainingType == 'onevsall':
            self.criterion = nn.BCELoss ().cuda ()
        else:
            self.criterion = nn.CrossEntropyLoss ().cuda ()
        self.lr = self.opt.LR
        # self.optimzer = optimizer or torch.optim.RMSprop(self.model.parameters(),
        #                                              lr=self.lr,
        #                                              eps=1,
        #                                              momentum=self.opt.momentum,
        #                                              weight_decay=self.opt.weightDecay)
        self.optimzer = optimizer or torch.optim.SGD (self.model.parameters (),
                                                      lr=self.lr,
                                                      momentum=self.opt.momentum,
                                                      weight_decay=self.opt.weightDecay,
                                                      nesterov=True)

    def updateopts(self):
        self.optimzer = torch.optim.SGD (self.model.parameters (),
                                         lr=self.lr,
                                         momentum=self.opt.momentum,
                                         weight_decay=self.opt.weightDecay,
                                         nesterov=True)

    def updatelearningrate(self, epoch):
        self.lr = getlearningrate (epoch=epoch, opt=self.opt)
        # update learning rate of model optimizer
        for param_group in self.optimzer.param_groups:
            param_group['lr'] = self.lr

    def forward(self, dark_input_var, light_input_var, labels_var=None):
        # forward and backward and optimize
        # print("darkinputsize", dark_input_var.size())
        Pair = (dark_input_var, light_input_var)
        # print("pair0size", Pair[0].size())
        # labelsize = labels_var[1].size()
        # pairsize = Pair[0].size()
        output = self.model (Pair)
        # outputsize = output[0].size()
        if labels_var is not None :  ##(x, x_structure)  labelopennarrow, labelsyne
                loss = self.criterion (output, labels_var)
        return output,loss



    # def forward(self, dark_input_var, light_input_var, labels_var=None): ##forward for two_branch
    #     # forward and backward and optimize   x_opennarrow, x_sysnec labels_opennarrow_var,labels_synechia_var
    #     Pair = (dark_input_var, light_input_var)
    #     output = self.model (Pair)
    #     if labels_var is not None :
    #         loss1 = self.criterion (output[0], labels_var[0])
    #         loss2 = self.criterion(output[1], labels_var[1])
    #         print("loss", loss1, loss2)
    #     else:
    #         loss = None
    #
    #     return output[0], (0.5*loss1)+loss2

    def backward(self, loss):
        self.optimzer.zero_grad ()
        loss.backward ()

        self.optimzer.step ()

    def train(self, epoch, train_loader):
        loss_sum = 0
        iters = len (train_loader)
        output_list = []
        label_list = []
        self.updatelearningrate (epoch)

        self.model.train ()

        start_time = time.time ()
        end_time = start_time

        for i, (dark_input, light_input, labels, _ ) in enumerate (train_loader):
            self.model.train ()
            start_time = time.time ()
            data_time = start_time - end_time
            """
            label for two branch
            
            """
            labels_synechia = generateTarget(dark_input[0], labels)
            reduce_labels_synechia = labels_synechia
            labels_synechia = labels_synechia.cuda()
            labels_var = Variable(labels_synechia)

            # labels_opennarrow = generateTarget(dark_input, labels[0])
            # reduce_labels_opennarrow = labels_opennarrow
            # labels_opennarrow = labels_opennarrow.cuda()
            # labels_opennarrow_var = Variable(labels_opennarrow)


            ####  process image
            # labels = generateTarget (dark_input, labels[1])
            # reduce_labels = labels
            # labels = labels.cuda ()
            # labels_var = Variable(labels)



            dark_var = Variable(dark_input.cuda ())
            light_var = Variable (light_input.cuda ())

            output, loss = self.forward (dark_var, light_var, labels_var)
            # output, loss = self.forward(dark_var, light_var, (labels_opennarrow_var,labels_synechia_var))

            prediction = output.data.cpu ()

            output_list.append (prediction.numpy ())
            # label_list.append (reduce_labels.cpu ().numpy ())
            label_list.append(reduce_labels_synechia.cpu().numpy())

            self.backward (loss)
            loss_sum += float (loss.data)
            # Here, total_loss is accumulating history across your training loop, since loss is a differentiable variable with autograd history.
            # You can fix this by writing total_loss += float(loss) instead.
            end_time = time.time ()

            iter_time = end_time - start_time

            printresult (epoch, self.opt.nEpochs, i + 1, iters, self.lr, data_time, iter_time,
                         loss.data, mode="Train")
        loss_sum /= iters
        auc, fpr, tpr = computeAUC (output_list, label_list, epoch)
        print ("|===>Training AUC: %.4f Loss: %.4f " % (auc, loss_sum))
        return auc, loss_sum

    def test(self, epoch, test_loader):
        loss_sum = 0
        iters = len (test_loader)
        output_list = []
        label_list = []
        pathlist= []
        self.model.eval ()

        start_time = time.time ()
        end_time = start_time
        for i, (dark_input, light_input, labels, image_name) in enumerate (test_loader):
            with torch.no_grad():
                pathlist.append(image_name)
                start_time = time.time ()
                data_time = start_time - end_time

                labels_synechia = generateTarget(dark_input[0], labels)
                reduce_labels_synechia = labels_synechia

                labels_synechia = labels_synechia.cuda()
                labels_var = Variable(labels_synechia)

                # labels_opennarrow = generateTarget(dark_input, labels[0])
                # reduce_labels_opennarrow = labels_opennarrow
                # labels_opennarrow = labels_opennarrow.cuda()
                # labels_opennarrow_var = Variable(labels_opennarrow)



                    # labels_var = Variable (labels)
                labels_var = Variable(labels_var)
                    # labels_opennarrow_var = Variable(labels_opennarrow_var)


                dark_input= dark_input.cuda ()
                light_input=light_input.cuda()

                dark_var = Variable (dark_input)
                light_var = Variable (light_input)


                # output, loss = self.forward (dark_var, light_var, labels_var)
                output, loss = self.forward(dark_var, light_var, labels_var)

                loss_sum += float(loss.data)/iters
                prediction = output.data.cpu ()
                output_list.append (prediction.numpy ())
                # label_list.append (reduce_labels.cpu ().numpy ())
                label_list.append (reduce_labels_synechia.cpu ().numpy ())
                end_time = time.time ()
                iter_time = end_time - start_time

                printresult (epoch, self.opt.nEpochs, i + 1, iters, self.lr, data_time, iter_time,
                             # loss.data[0], mode="Test")
                             loss.data, mode="Test")

        # loss_sum /= iters
        auc, fpr, tpr = computeAUC (output_list, label_list, epoch)
        acc, precision, recall, f1, gmean, tn, fp, fn, tp, wronglist = computeEval (output_list, label_list, pathlist)
        print ("|===>Testing AUC: %.4f Loss: %.4f acc: %.4f precision: %.4f recall: %.4f f1: %.4f gmean: %.4f" % (
        auc, loss_sum, acc, precision, recall, f1, gmean))
        return auc, loss_sum, acc, precision, recall, f1, gmean, tn, fp, fn, tp, wronglist


def generate_factor(T, T_max=200):
    a = (1-(math.pow(float((T/T_max)),2)))
    # a = (math.pow (float ((T / T_max)), 2))
    # a =  1*(1 - (math.pow (float ((T / T_max)), 2)))
    # a=1
    return a

class Trainer_contra(object):
    realLabelsarr = []
    predictLabelsarr = []

    def __init__(self, model, opt, optimizer=None):
        self.opt = opt
        self.model = model
        # print (model)
        if self.opt.trainingType == 'onevsall':
            self.criterion = nn.BCELoss().cuda()
        else:
            self.criterion = nn.CrossEntropyLoss().cuda()

        self.criterion_contra = ContrastiveLoss()
        if opt.contra_focal == True:
            self.criterion_focal = FocalLoss (alpha=0.75, gamma=2)
        self.lr = self.opt.LR
        # self.optimzer = optimizer or torch.optim.RMSprop(self.model.parameters(),
        #                                              lr=self.lr,
        #                                              eps=1,
        #                                              momentum=self.opt.momentum,
        #                                              weight_decay=self.opt.weightDecay)
        self.optimzer = optimizer or torch.optim.SGD(self.model.parameters(),
                                                     lr=self.lr,
                                                     momentum=self.opt.momentum,
                                                     weight_decay=self.opt.weightDecay,
                                                     nesterov=True)

    def updateopts(self):
        self.optimzer = torch.optim.SGD(self.model.parameters(),
                                        lr=self.lr,
                                        momentum=self.opt.momentum,
                                        weight_decay=self.opt.weightDecay,
                                        nesterov=True)

    def updatelearningrate(self, epoch):
        self.lr = getlearningrate(epoch=epoch, opt=self.opt)
        # update learning rate of model optimizer
        for param_group in self.optimzer.param_groups:
            param_group['lr'] = self.lr

    def custom_replace(self, tensor, on_zero, on_non_zero):
        # we create a copy of the original tensor,
        # because of the way we are replacing them.
        res = tensor.clone()
        res[tensor == 0] = on_zero
        res[tensor != 0] = on_non_zero
        return res

    def forward(self, dark_input_var, light_input_var, fulldark_var, fulllight_var, labels_var=None):
        # forward and backward and optimize
        Pair = (dark_input_var, light_input_var, fulldark_var, fulllight_var)
        predict = self.model (Pair)  # h_d, h_l, x
        h_full_d, h_full_l, h_d, h_l, x = predict["h_full_d"], predict["h_full_l"], predict["h_d"], predict["h_l"], \
                                          predict["x"]
        labels_contra = self.custom_replace (labels_var, 1., 0.)
        if labels_var is not None:  ##(x, x_structure)  labelopennarrow, labelsyne
            # print("focal loss")
            loss0 = self.criterion_focal (x, labels_var)
            loss1 = self.criterion_contra (h_d, h_l, labels_contra) + 1*self.criterion_contra (h_full_d, h_full_l,
                                                                                             labels_contra)

            # loss1 = self.criterion_contra(h_d, h_l, labels_var)
            # loss = loss0+0.1*loss1
        else:
            loss = None

        # print("0.1")
        return x, loss0, loss1


        # h_d, h_l, x = self.model(Pair)  #h_d, h_l, x
        #
        # labels_contra = self.custom_replace(labels_var, 1., 0.)
        # # labels_contra = labels_var
        # if labels_var is not None:  ##(x, x_structure)  labelopennarrow, labelsyne
        #     if opt.contra_focal == True:
        #         loss0 = self.criterion_focal(x, labels_var)
        #         # loss0 = self.criterion (x, labels_var)##bce loss
        #     else:
        #         loss0 = self.criterion(x, labels_var)
        #     loss1 = self.criterion_contra(h_d,h_l,labels_contra)
        #     # loss1 = self.criterion_contra(h_d, h_l, labels_var)
        #     loss = loss0+1*loss1
        # else:
        #     loss = None
        # return x, loss

    def backward(self, loss):
        self.optimzer.zero_grad()
        loss.backward()

        self.optimzer.step()

    def train(self, epoch, train_loader):
        loss_sum = 0
        iters = len(train_loader)
        output_list = []
        label_list = []
        self.updatelearningrate(epoch)

        self.model.train()

        start_time = time.time()
        end_time = start_time
        ###for cumulative learning
        # Tmax = opt.nEpochs
        self.factor = generate_factor (T=epoch)
        # self.criterion_contra.change_margin(epoch, Tmax)
        ###for cumulative learning
        for i, (dark_input, light_input, labels, _) in enumerate(train_loader):
            self.model.train()
            start_time = time.time()
            data_time = start_time - end_time
            """
            label for two branch

            """
            labels_synechia = generateTarget(dark_input[0], labels)
            reduce_labels_synechia = labels_synechia
            labels_synechia = labels_synechia.cuda()
            labels_var = Variable(labels_synechia)


            dark_var = Variable(dark_input[0].cuda())
            light_var = Variable(light_input[0].cuda())
            dark_full_var = Variable(dark_input[1].cuda())
            light_full_var = Variable(light_input[1].cuda())
            output, loss0, loss1 = self.forward(dark_var, light_var, dark_full_var, light_full_var,labels_var)
            loss = loss0 + self.factor * loss1
            prediction = output.data.cpu()
            output_list.append(prediction.numpy())
            label_list.append(reduce_labels_synechia.cpu().numpy())

            self.backward(loss)
            loss_sum += float(loss.data)
            # Here, total_loss is accumulating history across your training loop, since loss is a differentiable variable with autograd history.
            # You can fix this by writing total_loss += float(loss) instead.
            end_time = time.time()

            iter_time = end_time - start_time

            printresult(epoch, self.opt.nEpochs, i + 1, iters, self.lr, data_time, iter_time,
                        loss.data, mode="Train")
        loss_sum /= iters
        auc, fpr, tpr = computeAUC(output_list, label_list, epoch)
        print("|===>Training AUC: %.4f Loss: %.4f " % (auc, loss_sum))
        return auc, loss_sum

    def test(self, epoch, test_loader):
        loss_sum = 0
        iters = len(test_loader)
        output_list = []
        label_list = []
        pathlist = []
        self.model.eval()

        start_time = time.time()
        end_time = start_time
        for i, (dark_input, light_input, labels, image_name) in enumerate(test_loader):
            with torch.no_grad():
                pathlist.append(image_name)
                start_time = time.time()
                data_time = start_time - end_time

                labels_synechia = generateTarget(dark_input[0], labels)
                reduce_labels_synechia = labels_synechia

                labels_synechia = labels_synechia.cuda()
                labels_var = Variable(labels_synechia)
                labels_var = Variable(labels_var)


                dark_var = Variable(dark_input[0].cuda())
                light_var = Variable(light_input[0].cuda())
                dark_full_var = Variable(dark_input[1].cuda())
                light_full_var = Variable(light_input[1].cuda())
                output, loss0, loss1 = self.forward(dark_var, light_var, dark_full_var, light_full_var, labels_var)
                loss = loss0+self.factor*loss1
                # loss = self.factor*loss0+ (1-self.factor)*loss1
                # loss = loss0 + 1 * loss1
                loss_sum += float(loss.data) / iters
                prediction = output.data.cpu()
                output_list.append(prediction.numpy())
                # label_list.append (reduce_labels.cpu ().numpy ())
                label_list.append(reduce_labels_synechia.cpu().numpy())
                end_time = time.time()
                iter_time = end_time - start_time

                printresult(epoch, self.opt.nEpochs, i + 1, iters, self.lr, data_time, iter_time,
                            # loss.data[0], mode="Test")
                            loss.data, mode="Test")

        # loss_sum /= iters
        auc, fpr, tpr = computeAUC(output_list, label_list, epoch)
        acc, precision, recall, f1, gmean, tn, fp, fn, tp, wronglist = computeEval(output_list, label_list, pathlist)
        print("|===>Testing AUC: %.4f Loss: %.4f acc: %.4f precision: %.4f recall: %.4f f1: %.4f gmean: %.4f" % (
            auc, loss_sum, acc, precision, recall, f1, gmean))
        return auc, loss_sum, acc, precision, recall, f1, gmean, tn, fp, fn, tp, wronglist



class Trainer_multiscale (object):
    realLabelsarr = []
    predictLabelsarr = []

    def __init__(self, model, opt, optimizer=None):
        self.opt = opt
        self.model = model
        # print (model)
        if self.opt.trainingType == 'onevsall':
            self.criterion = nn.BCELoss ().cuda ()
            self.criterion_focal = FocalLoss(alpha=0.75,gamma=2).cuda()
        else:
            self.criterion = nn.CrossEntropyLoss ().cuda ()
        self.lr = self.opt.LR
        # self.optimzer = optimizer or torch.optim.RMSprop(self.model.parameters(),
        #                                              lr=self.lr,
        #                                              eps=1,
        #                                              momentum=self.opt.momentum,
        #                                              weight_decay=self.opt.weightDecay)
        self.optimzer = optimizer or torch.optim.SGD (self.model.parameters (),
                                                      lr=self.lr,
                                                      momentum=self.opt.momentum,
                                                      weight_decay=self.opt.weightDecay,
                                                      nesterov=True)

    def updateopts(self):
        self.optimzer = torch.optim.SGD (self.model.parameters (),
                                         lr=self.lr,
                                         momentum=self.opt.momentum,
                                         weight_decay=self.opt.weightDecay,
                                         nesterov=True)

    def updatelearningrate(self, epoch):
        self.lr = getlearningrate (epoch=epoch, opt=self.opt)
        # update learning rate of model optimizer
        for param_group in self.optimzer.param_groups:
            param_group['lr'] = self.lr

    def forward(self, dark_input_var, light_input_var, fulldark_var, fulllight_var, labels_var=None):##
        # forward and backward and optimize
        # print("darkinputsize", dark_input_var.size())
        Pair = (dark_input_var, light_input_var, fulldark_var, fulllight_var)
        # print("pair0size", Pair[0].size())
        if opt.typedata =="light":
            output = self.model (Pair[1]) ##0 dark 1 light 2 full dark 3 full light
        elif opt.typedata == "dark":
            output = self.model (Pair[0]) ##0 dark 1 light 2 full dark 3 full light
        if labels_var is not None :
            # print("focal")
            loss1 = self.criterion_focal(output, labels_var)
            # loss = self.criterion (output, labels_var)
        else:
            loss = None

        return output, loss1

    def backward(self, loss):
        self.optimzer.zero_grad ()
        loss.backward ()

        self.optimzer.step ()

    def train(self, epoch, train_loader):
        loss_sum = 0
        iters = len (train_loader)
        output_list = []
        label_list = []
        self.updatelearningrate (epoch)

        self.model.train ()

        start_time = time.time ()
        end_time = start_time

        for i, (dark_input, light_input, labels, _) in enumerate (train_loader):##(dark_input, dark_full_input), (light_input, light_full_input), label, details
            self.model.train ()
            start_time = time.time ()
            data_time = start_time - end_time
            ####  process image
            labels = generateTarget (dark_input[0], labels)
            reduce_labels = labels
            labels = labels.cuda ()
            labels_var = Variable(labels)



            dark_var = Variable(dark_input[0].cuda ())
            fulldark_var = Variable(dark_input[1].cuda())

            light_var = Variable (light_input[0].cuda ())
            fulllight_var = Variable(light_input[1].cuda())

            output, loss = self.forward (dark_var, light_var, fulldark_var, fulllight_var, labels_var)

            prediction = output.data.cpu ()

            output_list.append (prediction.numpy ())
            label_list.append (reduce_labels.cpu ().numpy ())

            self.backward (loss)
            loss_sum += float (loss.data)
            # Here, total_loss is accumulating history across your training loop, since loss is a differentiable variable with autograd history.
            # You can fix this by writing total_loss += float(loss) instead.
            end_time = time.time ()

            iter_time = end_time - start_time

            printresult (epoch, self.opt.nEpochs, i + 1, iters, self.lr, data_time, iter_time,
                         loss.data, mode="Train")
        loss_sum /= iters
        auc, fpr, tpr = computeAUC (output_list, label_list, epoch)
        # print ("|===>Training AUC: %.4f Loss: %.4f " % (auc, loss_sum))
        return auc, loss_sum

    def test(self, epoch, test_loader):
        loss_sum = 0
        iters = len (test_loader)
        output_list = []
        label_list = []
        pathlist= []
        self.model.eval ()

        start_time = time.time ()
        end_time = start_time
        for i, (dark_input, light_input, labels, orgpath) in enumerate (test_loader):
            with torch.no_grad():
                pathlist.append(orgpath)
                start_time = time.time ()
                data_time = start_time - end_time

                labels = generateTarget (dark_input[0], labels)
                reduce_labels = labels
                labels = labels.cuda ()



                labels_var = Variable (labels)

                # dark_input= dark_input[0].cuda ()
                # fulldark_var =  dark_input[1].cuda ()
                # light_input=light_input[0].cuda()
                # fulllight_var = light_input[1].cuda()


                fulldark_var = Variable (dark_input[1].cuda ())
                dark_var = Variable (dark_input[0].cuda ())
                fulllight_var = Variable (light_input[1].cuda())
                light_var = Variable (light_input[0].cuda())

                # print("fulllightvar", fulllight_var.size())
                # print("fulldarkvar", fulldark_var.size())
                # print("lightvar", light_var.size())
                # print("darkvar", dark_var.size())
                output, loss = self.forward (dark_var, light_var, fulldark_var, fulllight_var, labels_var)
                loss_sum += float(loss.data)/iters
                prediction = output.data.cpu ()
                output_list.append (prediction.numpy ())
                label_list.append (reduce_labels.cpu ().numpy ())
                end_time = time.time ()
                iter_time = end_time - start_time

                printresult (epoch, self.opt.nEpochs, i + 1, iters, self.lr, data_time, iter_time,
                             # loss.data[0], mode="Test")
                             loss.data, mode="Test")

        # loss_sum /= iters
        auc, fpr, tpr = computeAUC (output_list, label_list, epoch)
        acc, precision, recall, f1, gmean, tn, fp, fn, tp, wronglist = computeEval (output_list, label_list, pathlist)
        print ("|===>Testing AUC: %.4f Loss: %.4f acc: %.4f precision: %.4f recall: %.4f f1: %.4f gmean: %.4f" % (
        auc, loss_sum, acc, precision, recall, f1, gmean))
        return auc, loss_sum, acc, precision, recall, f1, gmean, tn, fp, fn, tp, wronglist