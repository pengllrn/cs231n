# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


class PreSampling(object):
    def __init__(self, p_data):
        """
        初始化参数
        :param p_data:原始数据，DataFrame,shape of (N,D)
        """
        self.p_data = p_data
        self.a_data = np.array(self.p_data)

        self.date_list = []
        self.data_parms = {}

        dates = p_data['date']
        self.date_list = list(np.sort(list(set(dates))))

        num_aisle = pd.groupby(p_data, 'date').size().max()

        self.data_parms['days'] = len(self.date_list)
        self.data_parms['aisles'] = num_aisle
        self.data_parms['dimension'] = len(p_data.columns)

    def check_data(self):
        """
        check row data,and padding it if lack of records.
        :return: boolean
        the result show if had fulled up the row data.
        """
        new_p_data = pd.DataFrame(columns=self.p_data.columns)
        days, aisles, dimension = self.data_parms['days'], self.data_parms['aisles'], self.data_parms['dimension']

        if len(self.p_data) == days * aisles:
            return

        for date, pd_i in self.p_data.groupby('date'):
            pd_i = pd_i.copy()
            if len(pd_i) < aisles:  # 有缺的
                for i in range(aisles):  # 检查哪里有缺
                    if len(pd_i[pd_i['index'] == (i + 1)]) == 0:
                        xw = pd_i['xw'].mean()
                        fd = pd_i['fd'].mean()
                        listdata = list(pd_i.iloc[0, 0:7])
                        listdata.extend([fd, xw, i + 1, 0, (360 - xw) if xw > 180 else xw])
                        insertRow = pd.DataFrame([listdata], columns=pd_i.columns)
                        pd_i = pd.concat([pd_i.iloc[0:i, :], insertRow, pd_i.iloc[i:, :]])
                    if len(pd_i) == aisles:
                        break
            new_p_data = pd.concat([new_p_data, pd_i], ignore_index=True)

        self.p_data = new_p_data
        return len(self.p_data) == days * aisles

    def sampling(self, steps=5, length=1):
        """
        predealing.
        :param length:refer to how long  days want to predict
        :return: the sampling result of a input length
        - out : a numpy array shape of (N`,5,aisles,dimension)
        N'= len(self.p_data)-5*length
        5:layers,represent 5 days assemble.
        one day is : aisles * dimension
        """
        assert self.check_data(), "Unkown Error,but the row_p_data is not complete! "
        aisles = self.data_parms['aisles']
        p_data = self.p_data.iloc[length * 5 * aisles:, :]  # ((days-steps*length)*aisles,12),这里为了保持长度一致，取最大步长5

        # sampling 5days
        day_steps_list = pd.DataFrame()
        for date, one_day in p_data.groupby("date"):  # days
            date_index = self.date_list.index(date)
            day_list = pd.DataFrame()
            for i in range(steps):
                start = (date_index - (i + 1) * length) * aisles
                end = start + aisles
                day_list = pd.concat([day_list, self.p_data.iloc[start:end, :]])  # (aisles,12)*5=====>5*aisles,12
            day_steps_list = pd.concat([day_steps_list, day_list], ignore_index=True)  # days*5*aisles,12)

        out = np.array(day_steps_list).reshape(len(p_data) / aisles, steps, aisles, -1)

        return out

    def get_p_data(self):
        return self.p_data


class ComputeDepriveVarious(object):
    def __init__(self, presampling, length=1):
        """
        Initial this class with a Presampling object.
        Inputs:
        :param presampling: a prepare object,witch include a row_p_data
        :param length: how long want to predict.
        """
        self.presampling = presampling
        self.length = length
        self.p_data = pd.DataFrame()
        self.D = 0

        out = presampling.sampling(steps=1, length=length)
        D = out.shape[3]
        out.reshape(-1, D)
        self.p_data['date'] = out[:, 0]
        self.p_data['index'] = out[:, 9]
        self.p_data['target'] = out[:, 10]
        self.D = D

    def compute_one_day(self):
        data = self.presampling.sampling(steps=1, length=self.length)
        data = data[:, :, :, 7:9]
        N, A, D = data.shape[0], data.shape[2], data.shape[3]
        data = data.reshape(N, A, D)  # N,A,2

        # 幅度相位变化趋势
        tendency_1, tendency_1_ratio = self.compute_tendency(data)
        self.p_data['fd_1_tend'] = tendency_1[:, 0]
        self.p_data['xw_1_tend'] = tendency_1[:, 1]
        self.p_data['fd_1_tend_ratio'] = tendency_1_ratio[:, 0]
        self.p_data['xw_1_tend_ratio'] = tendency_1_ratio[:, 1]
        # 幅度、相位差
        fdxwc = np.zeros((N, A, D))
        data_shift = data[1:N, :, :]
        for i in range(N - 1):
            fdxwc[i + 1] = data_shift[i] - data[i]
        # 幅度、相位差  变化趋势（分母可能有很多为0 的情况，所以不要求计算变化率）
        tendency_1 = self.compute_tendency(fdxwc, need_ratio=False)
        self.p_data['fdc_1_tend'] = tendency_1[:, 0]
        self.p_data['xwc_1_tend'] = tendency_1[:, 1]

    def compute_three_days(self):
        data = self.presampling.sampling(steps=3, length=self.length)  # N,3,A,D
        data = data[:, :, :, 7:9]  # 只需要幅度和相位两列
        N, A, D = data.shape[0], data.shape[2], data.shape[3]

        # 计算3日均值，包括幅度和相位的均值  shape:N,1,A,D--->N*A,D
        avg_3 = np.mean(data, axis=1).reshape(-1, D)
        self.p_data['fd_3_avg'] = avg_3[:, 0]
        self.p_data['xw_3_avg'] = avg_3[:, 1]
        # 计算3日最大值，最小值
        max_3 = np.max(data, axis=1).reshape(-1, D)
        min_3 = np.min(data, axis=1).reshape(-1, D)
        self.p_data['fd_3_max'] = max_3[:, 0]
        self.p_data['xw_3_max'] = max_3[:, 1]
        self.p_data['fd_3_min'] = min_3[:, 0]
        self.p_data['xw_3_min'] = min_3[:, 1]
        # 计算3日均值变化趋势，变化率
        mean_3 = np.mean(data, axis=1).reshape(N, A, D)
        tendency_3, tendency_3_ratio = self.compute_tendency(mean_3)
        self.p_data['fd_3_tend'] = tendency_3[:, 0]
        self.p_data['xw_3_tend'] = tendency_3[:, 1]
        self.p_data['fd_3_tend_ratio'] = tendency_3_ratio[:, 0]
        self.p_data['xw_3_tend_ratio'] = tendency_3_ratio[:, 1]

        # 3日幅度、相位差
        fdxwc_3 = np.zeros((N, 3, A, D))
        data_shift = data[1:N, :, :, :]
        for i in range(N - 1):
            fdxwc_3[i + 1] = data_shift[i] - data[i]
        # 3日幅度、相位差 均值
        fdxwc_avg_3 = np.mean(fdxwc_3, axis=1).reshape(N, A, D)
        fdxwc_3_avg = fdxwc_avg_3.reshape(-1, D)
        self.p_data['fdc_3_avg'] = fdxwc_3_avg[:, 0]
        self.p_data['xwc_3_avg'] = fdxwc_3_avg[:, 1]
        # 幅度、相位差 均值 变化趋势（分母可能有很多为0 的情况，所以不要求计算变化率）
        tendency_3 = self.compute_tendency(fdxwc_avg_3, need_ratio=False)
        self.p_data['fdc_3_avg_tend'] = tendency_3[:, 0]
        self.p_data['xwc_3_avg_tend'] = tendency_3[:, 1]
        # 相位差fdxwc_3极大极小值
        max_3_fdxwc = np.max(fdxwc_3, axis=1).reshape(-1, D)
        min_3_fdxwc = np.min(fdxwc_3, axis=1).reshape(-1, D)
        self.p_data['fdc_3_max'] = max_3_fdxwc[:, 0]
        self.p_data['xwc_3_max'] = max_3_fdxwc[:, 1]
        self.p_data['fdc_3_min'] = min_3_fdxwc[:, 0]
        self.p_data['xwc_3_min'] = min_3_fdxwc[:, 1]

        # 极值差
        peak_diff = (np.max(data, axis=1) - np.min(data, axis=1)).reshape(N, A, D)
        fdxw_peak_diff_3 = self.compute_tendency(peak_diff, need_ratio=False)
        self.p_data['fd_peak_diff_tend_3'] = fdxw_peak_diff_3[:, 0]
        self.p_data['xw_peak_diff_tend_3'] = fdxw_peak_diff_3[:, 1]
        # 极值差 差异
        shift_peak_diff = peak_diff[1:N, :, :]
        peak_diff_c = np.zeros_like(peak_diff)
        for i in range(N - 1):
            peak_diff_c[i + 1] = shift_peak_diff[i] - peak_diff[i]
        # 极值差 差异 趋势
        tendency_3_peak_diff_c = self.compute_tendency(peak_diff_c)
        self.p_data['fd_peak_diff_c_tend_3'] = tendency_3_peak_diff_c[:, 0]
        self.p_data['xw_peak_diff_c_tend_3'] = tendency_3_peak_diff_c[:, 1]

    def compute_five_days(self):
        data = self.presampling.sampling(steps=5, length=self.length)  # N,5,A,D
        data = data[:, :, :, 7:9]  # 只需要幅度和相位两列
        N, A, D = data.shape[0], data.shape[2], data.shape[3]

        # 计算5日均值，包括幅度和相位的均值  shape:N,1,A,D ---> N * A,D
        avg_5 = np.mean(data, axis=1).reshape(-1, D)
        self.p_data['fd_5_avg'] = avg_5[:, 0]
        self.p_data['xw_5_avg'] = avg_5[:, 1]
        # 计算5日最大值，最小值
        max_5 = np.max(data, axis=1).reshape(-1, D)
        min_5 = np.min(data, axis=1).reshape(-1, D)
        self.p_data['fd_5_max'] = max_5[:, 0]
        self.p_data['xw_5_max'] = max_5[:, 1]
        self.p_data['fd_5_min'] = min_5[:, 0]
        self.p_data['xw_5_min'] = min_5[:, 1]


        # 计算5日均值变化率
        mean_5 = np.mean(data, axis=1).reshape(N, A, D)
        shift_mean_5 = mean_5[1:N, :, :]
        tendency_ratio = np.zeros(N, A, D)
        for i in range(N - 1):
            tendency_ratio[i + 1] = (shift_mean_5[i] - mean_5[i]) / mean_5[i]
        tendency_ratio = tendency_ratio.reshape(-1, D)
        self.p_data['fd_5_tend_ratio'] = tendency_ratio[:, 0]
        self.p_data['xw_5_tend_ratio'] = tendency_ratio[:, 1]

    def compute_tendency(self, data, need_ratio=True):
        """
        计算趋势变化
        :param data:shape of (N,A,D)
        :param need_ratio: whither need commute tendency ratio at the same time
        :return:
        - tendency:the relationship between data[N-1] and data[N],the shape of
            tendency is (N*A,D)
        - tendency_ratio:matrics, values of (data[N]-data[N-1])/data[N-1]
          the shape of tendency_ratio is (N*A,D)
        """
        N, A, D = data.shape
        shift_data = data[1:N, :, :]
        tendency = np.random.binomial(1, 0.5, size=(N, A, D))
        tendency_ratio = np.zeros((N, A, D))
        for i in range(N - 1):
            tendency[i + 1] = ((shift_data[i] - data[i]) > 0) * (np.ones((A, D), dtype='i2'))
            if need_ratio:
                tendency_ratio[i + 1] = (shift_data[i] - data[i]) / (data[i] + 1e-6)
        tendency = tendency.reshape(-1, D)
        tendency_ratio = tendency_ratio.reshape(-1, D)
        if need_ratio:
            return tendency, tendency_ratio
        else:
            return tendency


if __name__ == '__main__':
    dd = {'Name': ['Tom', 'Jack', "Alice", 'LIli', "DD"], 'Age': [28, 34, 29, 28, 34], "Sex": [1, 1, 0, 0, 1],
          'date': ['2018-1-1', '2018-1-2', '2018-1-3', '2018-1-4', '2018-1-2']}
    pdd = pd.DataFrame(dd)
    # s=list(set(pdd['Age']))
    # print s
    # print pdd
    # print np.array(pdd)
    # print np.sort(list(s))
    print pdd
    a = pdd.groupby('date')
    # print a["date"].describe()
    # print a["date"].count()
    # print a['date'].count().min()
    print a.size()
    # # print len(pdd.columns)
    print len(a)
    for name, group in a:
        print name
        print group

    print '------------'
    # pieces=dict(list(a))
    # print type(pieces['2018-1-1'])
    # print type(pieces['2018-1-1']['Age'])
    #
    # print type(pieces['2018-1-1']['Age'][0])
    print pd.DataFrame(columns=pdd.columns)
