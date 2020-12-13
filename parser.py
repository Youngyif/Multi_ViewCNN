import argparse
def parse_args():
    parser = argparse.ArgumentParser(description='synechiae parameters')
    parser.add_argument('--backbone', dest='backbone',
                        default='resnet3d', type=str)
    parser.add_argument('--tb', dest='train_batch',
                        default=8, type=int)
    parser.add_argument ('--focal scale', dest='scale',
                         default=16, type=int)
    parser.add_argument ('--focal gamma', dest='gamma',
                         default=2, type=int)
    parser.add_argument ('--focal alpha', dest='alpha',
                         default=0.25, type=float)
    parser.add_argument ('--focal mining margin', dest='mining margin',
                         default=0.8, type=int)
    parser.add_argument ('--gpulist', dest='gpulist',
                         default=[6,7], type=list)
    parser.add_argument ('--imgsize', dest='imgsize',
                         default=244, type=int)
    args = parser.parse_args()
    return args