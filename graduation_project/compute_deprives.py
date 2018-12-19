# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


class PreSampling:
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
        for date, pd_i in self.p_data.groupby('date'):
            if len(pd_i) < aisles:
                pd_i = pd_i.copy()
                pd_i.append(pd_i.iloc[0:(aisles - len(pd_i)), :])
            new_p_data.append(pd_i, ignore_index=True)

        self.p_data = new_p_data
        return len(self.p_data) == days

    def sampling(self, length=1):
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
        p_data = self.p_data.iloc[length * 5:, :]

        # sampling 5days
        day_5_list = []
        for date, one_day in p_data.groupby("date"):
            date_index = self.date_list.index(date)
            day_list = []
            for i in range(5):
                start = (date_index - (i + 1) * length) * aisles
                end = start + aisles
                day_list.extend(self.p_data.iloc[start:end, :])
            day_5_list.extend(day_list)  # check the oder is right

        out = np.array(day_5_list).reshape(len(p_data), 5, aisles, -1)

        return out


class ComputeDepriveVarious:
    def __init__(self,data):
        self.data=data
        self.p_data=pd.DataFrame(data[0],columns=['date'])

    def compute_two_day(self):
        N=self.data.shape[0]
        new_p_data=pd.DataFrame()
        for i in range(N):
            one_day_mean=np.mean(self.data[i][0:2],axis=0)
            tmp_p=pd.DataFrame(one_day_mean[:,7:8],columns=['fd_mean_2day','xw_mean_2day'])
            new_p_data.append(tmp_p)
        self.p_data=pd.concat([self.p_data,new_p_data],axis=1)
        return new_p_data



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
