import os
from saveModel.resultcurve import *
from saveModel.graphgen import *
import shutil


class Visualization(object):
    def __init__(self, opt):
        if not os.path.isdir(opt.save_head_path+opt.save_path):
            os.mkdir(opt.save_head_path+opt.save_path)
        self.save_path = opt.save_head_path+opt.save_path
        self.wrong_path  =self.save_path+"wrongpath.txt"
        self.log_file = self.save_path + "log.txt"
        self.readme = self.save_path + "README.md"
        self.opt_file = self.save_path + "opt.log"
        self.code_path = os.path.join(self.save_path, "code/")
        # self.weight_folder = self.save_path + "weight/"
        # self.weight_fig_folder = self.save_path + "weight_fig/"
        if os.path.isfile(self.log_file):
            os.remove(self.log_file)
        if os.path.isfile(self.readme):
            os.remove(self.readme)
        # if not os.path.isdir(self.code_path):
        #     os.mkdir(self.code_path)
        self.copy_code(dst=self.code_path)
        """if os.path.isdir(self.weight_folder):
            shutil.rmtree(self.weight_folder, ignore_errors=True)
        os.mkdir(self.weight_folder)
        if os.path.isdir(self.weight_fig_folder):
            shutil.rmtree(self.weight_fig_folder, ignore_errors=True)
        os.mkdir(self.weight_fig_folder)"""

        self.graph = Graph()
        # print "|===>Result will be saved at", self.save_path

    # def copy_code1(self, src="./", dst="./code/"): ##只能拷贝本路径的py文件
    #     for file in os.listdir(src):
    #         file_split = file.split('.')
    #         if len(file_split) >= 2 and file_split[1] == "py":
    #             src_file = src + file
    #             dst_file = dst + file
    #             print("src",src_file)
    #             print("dst", dst_file)
    #             try:
    #                 shutil.copyfile(src=src_file, dst=dst_file)
    #             except:
    #                 print ("copy file error")

    def copy_code(self, src="./", dst="./code/"):##可以拷贝子文件夹中的代码
        # L = []
        # for dirpath, dirnames, filenames in os.walk(src):
        #     for file in filenames:
        #         if os.path.splitext(file)[1] == '.py':
        #             L.append(dirpath + "/" + file)
        # 
        # for path in L:
        #     try:
        #         shutil.copyfile(src=path, dst=dst + path.split("/")[-1])
        #     except:
        #         print("copy file error")
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns('*.pyc', '.csv', '.txt','.json', '.jsonl','log'))
    def writeopt(self, opt):
        with open(self.opt_file, "w") as f:
            for k, v in opt.__dict__.items():
                f.write(str(k)+": "+str(v)+"\n")
    def writepath(self, wrongpath):
        print(">>>>>write wrong path")
        if os.path.isfile(self.wrong_path):
            os.remove(self.wrong_path)
        wrong_file = open(self.wrong_path, "a+")
        for path in wrongpath:
            wrong_file.writelines(str(path)+"\n")
        wrong_file.close()

    def writelog(self, input_data):
        print(">>>>> write to log")
        txt_file = open(self.log_file, 'a+')
        txt_file.write(str(input_data) + "\n")
        txt_file.close()

    def writereadme(self, input_data):
        txt_file = open(self.readme, 'a+')
        txt_file.write(str(input_data) + "\n")
        txt_file.close()

    def drawcurves(self):
        drawer = DrawCurves(file_path=self.log_file, fig_path=self.save_path)
        drawer.draw(target="test_error")
        drawer.draw(target="train_error")

    def gennetwork(self, var):
        self.graph.draw(var=var)

    def savenetwork(self):
        self.graph.save(file_name=self.save_path+"network.svg")

    """def writeweights(self, input_data, block_id, layer_id, epoch_id):
        txt_path = self.weight_folder + "conv_weight_" + str(epoch_id) + ".log"
        txt_file = open(txt_path, 'a+')
        write_str = "%d\t%d\t%d\t" % (epoch_id, block_id, layer_id)
        for x in input_data:
            write_str += str(x) + "\t"
        txt_file.write(write_str+"\n")

    def drawhist(self):
        drawer = DrawHistogram(txt_folder=self.weight_folder, fig_folder=self.weight_fig_folder)
        drawer.draw()"""

