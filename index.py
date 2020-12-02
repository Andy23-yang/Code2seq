import pickle
"""
split java_files into train test and val
"""
import pandas as pd
import os
from multiprocessing import cpu_count, Pool


class Data():
    def __init__(self, data_root):
        self.data_root = data_root

    def cp_train_file(self, idx):
        command = "cp /datadrive/Data/codenet/deepcom/java_files/" + str(idx) + ".java "+self.data_root+"train/"
        os.system(command)

    def cp_test_file(self, idx):
        command = "cp /datadrive/Data/codenet/deepcom/java_files/" + str(idx) + ".java " + self.data_root + "test/"
        os.system(command)

    def cp_val_file(self, idx):
        command = "cp /datadrive/Data/codenet/deepcom/java_files/" + str(idx) + ".java " + self.data_root + "val/"
        os.system(command)

    def get_train_together(self, fid):
         command = "java -cp JavaExtractor/JPredict/target/JavaExtractor-0.0.1-SNAPSHOT.jar JavaExtractor.App " \
                   "--dir /datadrive/Data/codenet/deepcom/java_files/"+str(fid)+".java --max_path_length=8 --max_path_width=2 >"\
                   +self.data_root+"train/"+str(fid)+".txt"
         os.system(command)

    def get_test_together(self, fid):
         command = "java -cp JavaExtractor/JPredict/target/JavaExtractor-0.0.1-SNAPSHOT.jar JavaExtractor.App " \
                   "--dir  /datadrive/Data/codenet/deepcom/java_files/"+str(fid)+".java --max_path_length=8 --max_path_width=2 >"\
                   +self.data_root+"test/"+str(fid)+".txt"
         os.system(command)

    def get_val_together(self, fid):
         command = "java -cp JavaExtractor/JPredict/target/JavaExtractor-0.0.1-SNAPSHOT.jar JavaExtractor.App " \
                   "--dir  /datadrive/Data/codenet/deepcom/java_files/"+str(fid)+".java --max_path_length=8 --max_path_width=2 >"\
                   +self.data_root+"val/"+str(fid)+".txt"
         os.system(command)

    def merge(self, outpath, fid_list, source):
        with open(outpath, 'w') as outfile:
            for fid in fid_list:
                with open(source+str(fid) + ".txt", "rb") as ff:
                    line = ff.readline()
                    content = str(fid) + "+" + str(line)
                    outfile.write(content+'\n')
                ff.close()

    def replace_target(self, source, summary_dict, output):
        with open(output, 'w') as outfile:
            with open(source, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    fid = int(line[:line.index('+')])
                    summary_list = summary_dict[fid]
                    target = ''
                    # use summary full = False, test and val length is 11
                    if 'train' in file:
                        for i in summary_list:
                            target += i + '|'
                    else:
                        for i in summary_list[:11]:
                            target += i + '|'
                    try:
                        path = line[line.index(' '):-4]
                    except:
                        continue
                    content = target[:-1]+' '+path
                    outfile.write(content+'\n')

    def delete_target(self, source, summary_dict, output):
        with open(output, 'w') as outfile:
            with open(source, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    try:
                        path = line[line.index(' '):-4]
                        content = line[line.index('+')+3:-4]
                    except:
                        continue
                    outfile.write(content+'\n')

if __name__ == "__main__":

    root = 'codenet_data/'
    id_path = '/datadrive/Data/codenet/deepcom/correct_fid.pkl'
    summary_path = '/datadrive/Data/codenet/deepcom/summary/cfp1_csi1_cfd0_clc1.pkl'

    data = Data(root)
    id_list = pd.read_pickle(id_path)
    train = id_list['train']
    val = id_list['val']
    test = id_list['test']

    '''继续处理train数据集'''
    # processed_id = []
    # for root, dirs, files in os.walk('codenet_data/train'):
    #     file_list = files
    # for item in file_list:
    #     id = int(item.split('.')[0])
    #     processed_id.append(id)
    # print(len(train))
    # print(len(processed_id))
    # unprocessed_id = list(set(train).difference(set(processed_id)))
    # print(len(unprocessed_id))

    """
        1 split dataset
    """
    # p = Pool(cpu_count())
    # p.map(data.cp_train_file, train)
    # print('finish train')
    # p = Pool(cpu_count())
    # p.map(data.cp_test_file, test)
    # print('finish test')
    # p = Pool(cpu_count())
    # p.map(data.cp_val_file, val)
    # print('finish val')

    """
        2 process id.txt
    """
    # p = Pool(10)
    # p.map(data.get_val_together, val)
    # print("finish val")
    # p.map(data.get_train_together, unprocessed_id)
    # print('finish train')
    # p.map(data.get_test_together, test)
    # print('finish test')

    """
    3 merge fix.txt to one file: .raw.txt : fid + abstract path
    """
    data.merge(root+"all.train.raw.txt", train, root+'train/')
    print('finish train')
    # data.merge(root+"all.test.raw.txt", test, root+'test/')
    # print('finish test')
    # data.merge(root+"all.val.raw.txt", val, root+'val/')
    # print('finish val')

    """
    4 替换target name 为csn的summary
    """
    summary = pd.read_pickle(summary_path)
    summary_train = summary['train']
    # summary_test = summary['test']
    # summary_val = summary['val']
    #
    # data.replace_target(root+"all.val.raw.txt", summary_val, root+"all.val.process.txt")
    # print("finish val")
    # data.replace_target(root+'all.test.raw.txt', summary_test, root+"all.test.process.txt")
    # print("finish test")
    data.replace_target(root+'all.train.raw.txt', summary_train, root+"all.train.process.txt")
    print("finish train")

