import numpy as np
import pandas as pd
from astropy.table import Table
import os
import glob

def create_folders(path_list):
    for item_path in path_list:
        if not os.path.exists(item_path):
            os.mkdir(item_path)

def match_simu_detect(simulated_outcat_path, detected_outcat_path, match_save_path, para_s, para_d, para_n):
    """
    match_simu_detect algorithm
    :param simulated_outcat_path: 仿真核表路径(str)，txt文件
    :param detected_outcat_path: 检测核表路径(str)，txt文件
    :param match_save_path: 匹配结果保存路径(str)
        程序会自动在该路径下生成 Match_table,Miss_table,False_table三个文件夹
    :param para_s: 仿真核表列名
        默认参数：para_s={"scl":'Cen1',"scb":'Cen2',"scv":'Cen3',"ssl":'Size1',"ssb":'Size2',"ssv":'Size3'}
        para.scl: 云核银经方向中心坐标，无输入时默认‘Cen1’
        para.scb: 云核银纬方向中心坐标，无输入时默认‘Cen2’
        para.scv: 云核速度方向中心坐标，无输入时默认‘Cen3’
        para.ssl: 云核峰值在银经方向的映射，无输入时默认‘Size1’
        para.ssb: 云核峰值在银纬方向的映射，无输入时默认‘Size2’
        para.ssv: 云核峰值在速度方向的映射，无输入时默认‘Size3’
    :param para_d: 检测核表列名
        LDC算法参数：即默认参数
        Conbased算法参数：para_d={"dcl":'Cen3',"dcb":'Cen2',"dcv":'Cen1'}
        para.dcl: 云核银经方向中心坐标，无输入时默认‘Cen1’
        para.dcb: 云核银纬方向中心坐标，无输入时默认‘Cen2’
        para.dcv: 云核速度方向中心坐标，无输入时默认‘Cen3’
    :param para_n: 核表坐标起始值
        LDC算法参数：即默认参数
        Conbased算法参数：para_n={"son":1,"don":0}
        para.son: 仿真核表坐标起始值，无输入时默认 1
        para.don: 检测核表坐标起始值，无输入时默认 1
    :return:
    """
    if not os.path.exists(match_save_path):
        os.mkdir(match_save_path)

    scl = para_s['scl']
    scb = para_s['scb']
    scv = para_s['scv']
    ssl = para_s['ssl']
    ssb = para_s['ssb']
    ssv = para_s['ssv']

    dcl = para_d['dcl']
    dcb = para_d['dcb']
    dcv = para_d['dcv']

    son = para_n['son']
    don = para_n['don']

    if para_s is None:
        para_s = {"scl": 'Cen1', "scb": 'Cen2', "scv": 'Cen3', "ssl": 'Size1', "ssb": 'Size2', "ssv": 'Size3'}

    if para_d is None:
        para_d = {"dcl":'Cen1',"dcb":'Cen2',"dcv":'Cen3'}

    if para_n is None:
        para_n = {"son":1,"don":1}

    clump_item = simulated_outcat_path.split('_')[-1].split('.')[0]

    Match_table = os.path.join(match_save_path, 'Match_table')
    Miss_table = os.path.join(match_save_path, 'Miss_table')
    False_table = os.path.join(match_save_path, 'False_table')
    create_folders([Match_table, Miss_table,False_table])

    Match_table_name = os.path.join(Match_table, 'Match_%s.txt' %clump_item)
    Miss_table_name = os.path.join(Miss_table, 'Miss_%s.txt' % clump_item)
    False_table_name = os.path.join(False_table, 'False_%s.txt' % clump_item)


    table_s = pd.read_csv(simulated_outcat_path, sep='\t')
    table_g = pd.read_csv(detected_outcat_path, sep='\t')

    # table_simulate1=pd.read_csv(path1,sep=' ')
    # table_g=pd.read_csv(path2,sep='\t')
    # table_g.columns = new_cols

    Error_xyz = np.array([2, 2, 2])  # 匹配容许的最大误差(单位：像素)

    Cen_simulate = np.vstack([table_s[scl], table_s[scb], table_s[scv]]).T + 1 - son
    Size_simulate = np.vstack([table_s[ssl], table_s[ssb], table_s[ssv]]).T + 1 - son
    Cen_detected = np.vstack([table_g[dcl], table_g[dcb], table_g[dcv]]).T + 1 - don

    Cen_detected = Cen_detected[~np.isnan(Cen_detected).any(axis=1), :]  #去掉空字符
    # calculate distance
    simu_len = Cen_simulate.shape[0]
    detect_len = Cen_detected.shape[0]
    distance = np.zeros([simu_len, detect_len])

    for i, item_simu in enumerate(Cen_simulate):
        for j, item_detect in enumerate(Cen_detected):
            cen_simu = item_simu
            cen_detect = item_detect
            temp = np.sqrt(((cen_detect - cen_simu)**2).sum())
            distance[i,j] = temp
    max_d = 1.2 * distance.max()

    match_record_simu_detect = [] #匹配核表
    match_num = 0

    while 1:
        # 找到距离最小的行和列
        d_ij_value = distance.min()
        if d_ij_value == max_d:  # 表示距离矩阵里面所有元素都匹配上了
            break
        [simu_i, detect_j] = np.where(distance==d_ij_value)
        simu_i, detect_j = simu_i[0], detect_j[0]
        cen_simu_i = Cen_simulate[simu_i]
        size_simu_i = Size_simulate[simu_i]
        cen_detect_j = Cen_detected[detect_j]

        # 确定误差项
        temp = np.array([Error_xyz, size_simu_i / 2.3548])
        Error_xyz1 = temp.min(axis=0)

        d_ij = np.abs(cen_simu_i - cen_detect_j)
        match_num_ = match_num
        if (d_ij<= Error_xyz1).all():
            distance[simu_i,:] = np.ones([detect_len]) * max_d
            distance[:, detect_j] = np.ones([simu_len]) * max_d
            match_num = match_num + 1
            match_record_simu_detect.append(np.array([d_ij_value, simu_i + 1, detect_j + 1]))  # 误差 仿真表索引 检测表索引

        if match_num == match_num_:
            break
    match_record_simu_detect = np.array(match_record_simu_detect)

    F1, precision, recall = 0, 0, 0
    if match_num > 0:
        precision = match_num / detect_len
        recall = match_num / simu_len
        F1 = 2 * precision * recall / (precision + recall)
        # print("simulated num = %d\t detected num %d\t match num %d" % (simu_len, detect_len, match_num))
    print("F1_precision_recall = %.3f, %.3f, %.3f" % (F1, precision, recall))

    # new_cols = ['PIDENT', 'Peak1', 'Peak2', 'Peak3', 'Cen1', 'Cen2', 'Cen3', 'Size1', 'Size2', 'Size3', 'theta', 'Peak',
    #             'Sum', 'Volume']
    if match_record_simu_detect.shape[0] > 0:

        new_cols_sium = table_s.keys()
        new_cols_detect = table_g.keys()

        names = ['s_' + item for item in new_cols_sium] #列名
        names1 = ['f_' + item for item in new_cols_detect]  # 列名
        table_title = names + names1

        # match_simu_inx = match_record_simu_detect[:, 1].astype(np.int)
        match_simu_inx = match_record_simu_detect[:, 1].astype(int)
        table_s_np = table_s.values[match_simu_inx - 1,:]

        # match_detect = match_record_simu_detect[:, 2].astype(np.int)
        match_detect = match_record_simu_detect[:, 2].astype(int)
        table_g_np = table_g.values[match_detect - 1, :]

        match_outcat = np.hstack([table_s_np, table_g_np])

        dataframe = pd.DataFrame(match_outcat, columns=table_title)
        # dataframe = dataframe.round({'ID': 0, 'Peak1': 0, 'Peak2': 0, 'Peak3': 0, 'Cen1': 3, 'Cen2': 3, 'Cen3': 3,
        #                              'Size1': 3, 'Size2': 3, 'Size3': 3, 'Peak': 3, 'Sum': 3, 'Volume': 3})
        dataframe.to_csv(Match_table_name, sep='\t', index=False)

        simu_inx = table_s['ID']

        # x = set([0.0])
        # miss_idx = np.setdiff1d(simu_inx, match_simu_inx).astype(np.int)  # 未检测到的云核编号
        miss_idx = np.setdiff1d(simu_inx, match_simu_inx).astype(int)  # 未检测到的云核编号

        miss_names = ['s_' + item for item in new_cols_sium] #列名
        if len(miss_idx) == 0:
            miss_outcat = []
        else:
            miss_outcat = table_s.values[miss_idx - 1, :]
        dataframe = pd.DataFrame(miss_outcat, columns=miss_names)
        # dataframe = dataframe.round({'ID': 0, 'Peak1': 0, 'Peak2': 0, 'Peak3': 0, 'Cen1': 3, 'Cen2': 3, 'Cen3': 3,
        #                              'Size1': 3, 'Size2': 3, 'Size3': 3, 'Peak': 3, 'Sum': 3, 'Volume': 3})
        dataframe.to_csv(Miss_table_name, sep='\t', index=False)
        # miss = Table(names=miss_names)
        # for item in miss_idx:  # 未检出表
        #     miss.add_row(list(table_s[int(item) - 1, :]))
        # miss.write(Miss_table_name, overwrite=True, format='ascii')
        try:
            detect_inx = table_g['ID']
        except KeyError:
            detect_inx = table_g['PIDENT']
        # false_idx = np.setdiff1d(detect_inx, match_detect).astype(np.int)
        false_idx = np.setdiff1d(detect_inx, match_detect).astype(int)

        if len(false_idx) == 0:
            false_outcat = []
        else:
            # print(false_idx)
            false_outcat = table_g.values[false_idx - 1, :]

        false_names = ['f_' + item for item in new_cols_detect]  # 列名
        dataframe = pd.DataFrame(false_outcat, columns=false_names)
        # dataframe = dataframe.round({'ID': 0, 'Peak1': 0, 'Peak2': 0, 'Peak3': 0, 'Cen1': 3, 'Cen2': 3, 'Cen3': 3,
        #                              'Size1': 3, 'Size2': 3, 'Size3': 3, 'Peak': 3, 'Sum': 3, 'Volume': 3})
        dataframe.to_csv(False_table_name, sep='\t', index=False)

    else:
        new_cols_sium = table_s.keys()
        new_cols_detect = table_g.keys()

        names = ['s_' + item for item in new_cols_sium]  # 列名
        names1 = ['f_' + item for item in new_cols_detect]  # 列名

        table_title = names + names1
        match_outcat = []
        dataframe = pd.DataFrame(match_outcat, columns=table_title)
        dataframe.to_csv(Match_table_name, sep='\t', index=False)

        miss_names = ['s_' + item for item in new_cols_sium]  # 列名
        miss_outcat = table_s.values
        dataframe = pd.DataFrame(miss_outcat, columns=miss_names)
        dataframe.to_csv(Miss_table_name, sep='\t', index=False)

        false_outcat = table_g.values
        false_names = ['f_' + item for item in new_cols_detect]  # 列名
        dataframe = pd.DataFrame(false_outcat, columns=false_names)
        dataframe.to_csv(False_table_name, sep='\t', index=False)


# if __name__ == '__main__':
#     # num = 100  # 核表密度
#     # match_result = r'match_1007'
#     match_result = r'Match_result/match_detect_LDC_115'
#     glob_path = r'E:/Simulated/synthetic_clump_015/outcat/*.txt'  # 数据
#     path_file_number = glob.glob(glob_path)
#     file_number = len(path_file_number)
#     print('file_number = ', file_number)
#     for i in range(0, file_number):
#         # i = 214
#         id = str(i).rjust(3, '0')
#         simulated_outcat_path = r'E:\Simulated\synthetic_clump_015\outcat\synthetic_outcat_%s.txt' % id
#         detected_outcat_path = r'E:\Detected\detect_LDC_115\outcat_path\synthetic_model_%s_outcat.txt' % id
#
#         match_simu_detect(simulated_outcat_path, detected_outcat_path, match_result)
#         # break

