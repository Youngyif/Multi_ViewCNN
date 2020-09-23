from opt.opt232 import *
from dataloader.dataloader232 import *
from visualization import *
from termcolor import colored
import torch.backends.cudnn as cudnn
from saveModel.checkpoint import *
from trainer import *
import wandb

def dataparallel(model, ngpus, gpu0=0):
    if ngpus == 0:
        assert False, "only support gpu mode"
    gpu_list = list (range (gpu0, gpu0 + ngpus))
    # assert torch.cuda.device_count() >= gpu0+ngpus, "Invalid Number of GPUs"
    if isinstance (model, list):
        for i in range (len (model)):
            if ngpus >= 2:
                if not isinstance (model[i], nn.DataParallel):
                    model[i] = torch.nn.DataParallel (model[i], gpu_list).cuda ()
            else:
                model[i] = model[i].cuda ()
    else:
        if ngpus >= 2:
            if not isinstance (model, nn.DataParallel):
                model = torch.nn.DataParallel (model, gpu_list).cuda ()
        else:
            model = model.cuda ()
    return model


def getweights(layer, epoch_id, block_id, layer_id, log_writer):
    if isinstance (layer, nn.Conv2d):
        weights = layer.weight.data.cpu ().numpy ()
        weights_view = weights.reshape (weights.size)
        log_writer (input_data=weights_view, block_id=block_id,
                    layer_id=layer_id, epoch_id=epoch_id)


def main(net_opt=None):
    """requirements:
    apt-get install graphviz
    pip install pydot termcolor"""

    start_time = time.time ()
    opt = net_opt or NetOption ()

    # set torch seed
    # init random seed
    torch.manual_seed (opt.manualSeed)
    torch.cuda.manual_seed (opt.manualSeed)
    cudnn.benchmark = True
    print(opt.nGPU)
    if opt.nGPU == 1 and torch.cuda.device_count () >= 1:
        assert opt.GPU <= torch.cuda.device_count () - 1, "Invalid GPU ID"
        torch.cuda.set_device (opt.GPU)
    else:
        torch.cuda.set_device (opt.GPU)

    # create data loader
    data_loader = DataLoader (dataset=opt.data_set, data_path=opt.data_path, label_path=opt.label_path,
                              batch_size=opt.batchSize, rootpath =opt.rootpath,
                              n_threads=opt.nThreads, ten_crop=opt.tenCrop, dataset_ratio=opt.datasetRatio)
    train_loader, test_loader = data_loader.getloader ()

    # define check point
    check_point = CheckPoint (opt=opt)
    # create residual network mode
    if opt.retrain:
        # check_point_params = check_point.retrainmodel ()
        print("do not retrain")
        check_point_params = check_point.check_point_params
    elif opt.resume:
        print("resume")
        check_point_params = check_point.resumemodel ()
    else:
        check_point_params = check_point.check_point_params

    greedynet = None
    try:
        optimizer = check_point_params['opts']
        start_stage = check_point_params['stage']
        start_epoch = check_point_params['resume_epoch']
        if check_point_params['resume_epoch'] is not None:
            start_epoch += 1
        if start_epoch >= opt.nEpochs:
            start_epoch = 0
            start_stage += 1
    except:
        print("loading bestmodel")
        optimizer = None
        start_stage = None
        start_epoch = None


    # model
    # if opt.netType =="dual_resnet3d":
    #     model = dual_resnet3d(num_classes=opt.numclass, use_nl=True)
        # mydict = model.state_dict()
        # state_dict = torch.load(opt.pretrain)["model"]
        # # print(state_dict)
        # pretrained_dict = {k: v for k, v in enumerate(state_dict) if k not in ["fc.bias", 'fc.weight']}
        # mydict.update(pretrained_dict)
        # # a = mydict
        # model.load_state_dict(mydict)
        # for p in model.parameters():
        #     p.requires_grad = False
        # model.fc = nn.Linear(2048, 1)
    if opt.netType == 'multi_viewCNN':
        model = my_mvcnn(opt.numOfView)
    if opt.netType =="resnet3d":
        model = resnet3d(num_classes=opt.numclass, use_nl=True)
        print("no nonlocal")
        if opt.resume and opt.pretrain:
            print("!!can not load two model at one time!!")
            return
        if opt.resume:
            state_dict = check_point_params["model"]
            model.load_state_dict(state_dict)
            # print(state_dict)
            # if opt.contra_learning_2 == True:
            #     for param in model.parameters():  # nn.Module有成员函数parameters()
            #         param.requires_grad = False  ##固定所有层
            # model.fc = nn.Linear(512*4*2, 1)


        elif opt.pretrain:
            print("loading model cpu")
            mydict = model.state_dict()
            state_dict = torch.load(opt.pretrain, map_location=torch.device('cpu'))
            # print(state_dict)
            pretrained_dict = {k: v for k, v in state_dict.items() if k not in ["fc.bias", 'fc.weight']}
            mydict.update(pretrained_dict)
            model.load_state_dict(mydict)
        # for param in model.parameters():  # nn.Module有成员函数parameters()
        #     param.requires_grad = False  ##固定所有层
        # model.fc = nn.Linear(512*4, 1)
    if opt.netType =="lstm_mvcnn":
        model = my_mvcnn_lstm(opt.numOfView)

    if opt.netType == "C3D":
        model = C3D ()
        print ("loading pretrain model")
        # model.load_state_dict(torch.load("/home/yangyifan/code/multiViewCNN/nvcnn_baseline/weights/c3d.pickle"))
        mydict = model.state_dict ()
        state_dict = torch.load ("/home/yangyifan/code/multiViewCNN/pretrained/weights/c3d.pickle",
                                 map_location=torch.device ('cpu'))
        # print(state_dict)
        pretrained_dict = {k: v for k, v in state_dict.items () if k not in ["fc8.bias", 'fc8.weight']}
        mydict.update (pretrained_dict)
        model.load_state_dict (mydict)
    if opt.netType == "I3D":
        model = InceptionI3d ()

        if opt.resume_I3D:
            model.replace_logits (1)
            print ("resume")
            mydict = model.state_dict ()
            state_dict = torch.load (
                "/home/yangyifan/model/synechiae/log_asoct_I3D_18_onevsall_bs8_baseline_contra1_I3D_oversample_alpha0.75_0723/model/checkpoint77.pkl",
                map_location=torch.device ('cpu'))["model"]
            # print (state_dict)
            pretrained_dict = {k: v for k, v in state_dict.items ()}
            mydict.update (pretrained_dict)
            model.load_state_dict (mydict)
        else:
            print ("loading pretrain model")
            # model.load_state_dict(torch.load("/home/yangyifan/code/multiViewCNN/nvcnn_baseline/weights/c3d.pickle"))
            mydict = model.state_dict ()
            state_dict = torch.load ("/home/yangyifan/code/multiViewCNN/pretrained/weights/rgb_imagenet.pt",
                                     map_location=torch.device ('cpu'))
            # print(state_dict)
            pretrained_dict = {k: v for k, v in state_dict.items () if k not in ["fc8.bias", 'fc8.weight']}
            mydict.update (pretrained_dict)
            model.load_state_dict (mydict)
            model.replace_logits (1)
    if opt.netType == "S3D":
        model = S3D (opt.numclass)
        file_weight = "/home/yangyifan/code/multiViewCNN/pretrained/weights/S3D_kinetics400.pt"
        weight_dict = torch.load (file_weight)
        model_dict = model.state_dict ()
        for name, param in weight_dict.items ():
            if 'module' in name:
                name = '.'.join (name.split ('.')[1:])
            if name in model_dict:
                if param.size () == model_dict[name].size ():
                    model_dict[name].copy_ (param)
                else:
                    print (' size? ' + name, param.size (), model_dict[name].size ())
            else:
                print (' name? ' + name)
    # if opt.netType == "slowfast":
    #     model = resnet50_sl (class_num=opt.numclass)
    #     pretrained = "/home/yangyifan/code/multiViewCNN/nvcnn_baseline/weights/SLOWFAST_4x16_R50.pkl"
    #     if pretrained is not None:
    #         pretrained_dict = torch.load (pretrained, map_location='cpu')
    #         try:
    #             model_dict = model.module.state_dict ()
    #         except AttributeError:
    #             model_dict = model.state_dict ()
    #         pretrained_dict = {k: v for k, v in pretrained_dict.items () if k in model_dict}
    #         print ("load pretrain model")
    #         model_dict.update (pretrained_dict)
    #         model.load_state_dict (model_dict)

    if opt.netType == "TSN":
        num_class  =opt.numclass
        model = TSN (num_class, 21, "RGB",
                   base_model='resnet50',
                   consensus_type="avg",
                   img_feature_dim=244,
                   pretrain='imagenet',
                   is_shift=False, shift_div=21, shift_place='blockres',
                   non_local=False,
                   )

    model = dataparallel (model, opt.nGPU, opt.GPU)
    # trainer = Trainer(model=model, opt=opt, optimizer=optimizer)
    if opt.contra_learning:
        print(">>>>trainer : contra_learning")
        trainer = Trainer_contra_learning(model=model, opt=opt, optimizer=optimizer)
    elif opt.circle_loss == True:
        trainer = Trainer_circle(model=model, opt=opt, optimizer=optimizer)
    elif opt.contra_learning_2:
        print(">>>>trainer : contra_learning")
        trainer = Trainer_contra_learning_2(model=model, opt=opt, optimizer=optimizer)
    elif opt.contra ==True or opt.contra_focal == True or opt.contra_focal_bilinear == True or opt.multiscale==True:
        print(">>>>trainer : contra")
        trainer = Trainer_contra(model=model, opt=opt, optimizer=optimizer)
        # trainer = Trainer_multiscale(model=model, opt=opt, optimizer=optimizer)
    elif opt.contra_single == True:
        print(">>>>trainer : contra_SINGLE")
        trainer = Trainer_multiscale(model=model, opt=opt, optimizer=optimizer)
    elif opt.mscale == True:
        print(">>>>trainer : multiscale")
        trainer = Trainer_multiscale(model=model, opt=opt, optimizer=optimizer)
    else:
        trainer = Trainer(model=model, opt=opt, optimizer=optimizer)


    print ("|===>Create trainer")

    if opt.testOnly:
        visualize = Visualization (opt=opt)
        visualize.writeopt (opt=opt)
        teststart_time = time.time ()
        test_auc, test_loss, test_acc, test_precision, test_recall, test_f1, test_gmean, tn, fp, fn, tp, wronglist=trainer.test (epoch=0, test_loader=test_loader)
        testend_time = time.time ()
        testtime = testend_time- teststart_time
        log_str = "%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%d\t%d\t%d\t%d\t%.4f\t" % (
             test_auc,
            test_loss, test_acc, test_precision, test_recall, test_f1, test_gmean, tn, fp, fn, tp,
             testtime)
        print(">>>>running experiment:", opt.experimentID, testtime)
        visualize.writelog (log_str)
        return

    # define visualizer
    visualize = Visualization (opt=opt)
    visualize.writeopt (opt=opt)
    wandb.init (
        project=opt.experimentID,
        config=opt.__dict__
    )
    wandb.watch (model)

    best_auc = 0
    best_loss = 0
    start_epoch = opt.resumeEpoch
    for epoch in range (start_epoch, opt.nEpochs):
        epochstart_time = time.time()

        train_auc, train_loss = trainer.train (
            epoch=epoch, train_loader=train_loader)
        wandb.log({"train_loss": train_loss, "train_auc":train_auc})
        epochtrain_end_time = time.time()
        epochTrainDurationtime  = epochtrain_end_time-epochstart_time
        test_auc, test_loss, test_acc, test_precision, test_recall, test_f1, test_gmean, tn, fp, fn, tp, wronglist= trainer.test (
            epoch=epoch, test_loader=test_loader)
        wandb.log ({"test_auc":test_auc, "test_loss":test_loss, "test_acc":test_acc})
        epochend_time = time.time ()
        epochtime = epochend_time-epochstart_time
        # write and print result
        log_str = "%d\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%d\t%d\t%d\t%d\t%.4f\t%.4f\t" % (
        epoch, train_auc, train_loss, test_auc,
        test_loss, test_acc, test_precision, test_recall, test_f1, test_gmean, tn, fp, fn, tp, epochTrainDurationtime, epochtime)
        visualize.writelog (log_str)
        best_flag = False
        if opt.contra_learning == True:
            if best_loss <= train_loss:
                bestepoch = epoch
                best_loss = train_loss
                best_flag = True
                print (colored ("# %d ==>Best Result is: LOSS: %f\n" % (
                    epoch, best_loss), "red"))
                visualize.writepath(wronglist)  ###write path of images which are wrongly classified
            else:
                print (colored ("# %d ==>Best Result is: AUC: %f in epoch:%d\n" % (
                    epoch, best_auc, bestepoch), "blue"))
        else:
            if best_auc <= test_auc:
                bestepoch = epoch
                best_auc = test_auc
                best_flag = True
                print (colored ("# %d ==>Best Result is: AUC: %f in epoch%d\n" % (
                    epoch, best_auc, bestepoch), "red"))
                visualize.writepath(wronglist)  ###write path of images which are wrongly classified
            else:
                print (colored ("# %d ==>Best Result is: AUC: %f\n" % (
                    epoch, best_auc), "blue"))

        # save check_point
        check_point.savemodel (epoch=epoch, model=trainer.model,
                               opts=trainer.optimzer, best_flag=best_flag)


    end_time = time.time ()
    time_interval = end_time - start_time
    t_hours = time_interval // 3600
    t_mins = time_interval % 3600 // 60
    t_sec = time_interval % 60
    t_string = "Running Time is: " + \
               str (t_hours) + " hours, " + str (t_mins) + \
               " minutes," + str (t_sec) + " seconds\n"
    print (t_string)


if __name__ == '__main__':
    # main()
    main_opt = NetOption ()  ##class NetOption()
    print(">>>>running experiment:", main_opt.experimentID)
    main_opt.paramscheck ()
    main (main_opt)
