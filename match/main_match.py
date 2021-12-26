import numpy as np
import pandas as pd
from astropy.table import Table
import os
import glob
from match_cloud import match_simu_detect

num = 10  # 核表密度
match_result = r'Match_result/match_LDC_testdata_1/n_clump_%03d' % num  # 结果保存路径
glob_path = r'/home/data/zhangr/detect_LDC_testdata_1/n_clump_%03d/outcat_path/*.txt' % num  # 检测核表文件夹路径（用于计数）
path_file_number = glob.glob(glob_path)  ## 检测核表数量
file_number = len(path_file_number)
print('file_number = ', file_number)
for i in range(0, file_number):
    # i = 214
    id = str(i).rjust(3, '0')
    # 仿真数据路径
    simulated_outcat_path = r'/home/share/project_clump/data_zhougr/test_data/n_clump_%03d/outcat/gaussian_outcat_%s.txt' % (num, id)
    # 检测数据路径
    detected_outcat_path = r'/home/data/zhangr/detect_LDC_testdata_1/n_clump_%03d/outcat_path/gaussian_out_%s_outcat.txt' % (num, id)

    # LDC算法检测核表参数输入
    para_s = {"scl": 'Cen1', "scb": 'Cen2', "scv": 'Cen3', "ssl": 'Size1', "ssb": 'Size2', "ssv": 'Size3'}  # 仿真核表表头
    para_d = {"dcl": 'Cen1', "dcb": 'Cen2', "dcv": 'Cen3'}  # 检测核表表头
    para_n = {"son": 1, "don": 1}  # 仿真和检测核表坐标起始值

    # Conbased算法生成数据匹配
    # para_s = {"scl": 'Cen1', "scb": 'Cen2', "scv": 'Cen3', "ssl": 'Size1', "ssb": 'Size2', "ssv": 'Size3'}  # 仿真核表表头
    # para_d = {"dcl": 'Cen3', "dcb": 'Cen2', "dcv": 'Cen1'}  # 检测核表表头
    # para_n = {"son": 1, "don": 0}  # 仿真和检测核表坐标起始值

    match_simu_detect(simulated_outcat_path, detected_outcat_path, match_result, para_s, para_d, para_n)
    # break