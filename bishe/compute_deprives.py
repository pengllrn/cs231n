import numpy as np
import pandas as pd

class ComputeDepriveVarious:

    def __init__(self,p_data):
        self.p_data=p_data
        self.a_data=np.array(self.p_data)

        self.date_list=[]
        self.data_parms={}

        dates=p_data['date']
        self.date_list=np.sort(list(set(dates)))

        num_aisle=pd.groupby(p_data,'date').size().max()

        self.data_parms['days']=len(self.date_list)
        self.data_parms['aisles']=num_aisle
        self.data_parms['dimension']=len(p_data.columns)

    def check_data(self):
        new_p_data=pd.DataFrame(columns=self.p_data.columns)
        days,aisles,dimension=self.data_parms['days'],self.data_parms['aisles'],self.data_parms['dimension']
        for date,pd_i in self.p_data.groupby('date'):
            if len(pd_i)<aisles:
                pd_i=pd_i.copy()
                pd_i.append(pd_i.iloc[0:(aisles-len(pd_i)),:])
            new_p_data.append(pd_i,ignore_index=True)

        self.p_data=new_p_data
        return len(self.p_data) == days

    def compute(self):
        assert self.check_data(),"Unkown Error,but the row_p_data is not complete! "
        p_data=self.p_data





if __name__ == '__main__':
    dd={'Name':['Tom','Jack',"Alice",'LIli',"DD"],'Age':[28,34,29,28,34],"Sex":[1,1,0,0,1],'date':['2018-1-1','2018-1-2','2018-1-3','2018-1-4','2018-1-2']}
    pdd=pd.DataFrame(dd)
    # s=list(set(pdd['Age']))
    # print s
    # print pdd
    # print np.array(pdd)
    # print np.sort(list(s))
    print pdd
    a=pdd.groupby('date')
    # print a["date"].describe()
    # print a["date"].count()
    # print a['date'].count().min()
    print a.size()
    # # print len(pdd.columns)
    for name,group in a:
        print name
        print group

    print '------------'
    # pieces=dict(list(a))
    # print type(pieces['2018-1-1'])
    # print type(pieces['2018-1-1']['Age'])
    #
    # print type(pieces['2018-1-1']['Age'][0])
    print pd.DataFrame(columns=pdd.columns)
