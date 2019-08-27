

class piecewise_data():

    __slots__ = ['data', 'partition', '_current_index', '_nextT']

    def __init__(self, data, partition):
        # np 2dim array data
        self.data = data
        # list partition
        self.partition = partition
        self._current_index = 0
        self._nextT = 0

    # should add picewise data combining partitions and data
    def add_rows(self, pdata):
        pass

    # should append columns to data and partition to partition
    def add_columns(self, data, partition):
        pass

    @property
    def nextT(self):
        return self._nextT

    def next_data(self):
        self._current_index +=1
        #check if we at the end of list if yes set self._nextT=np.inf
        self._nextT = self.partition[self._current_index+1]
        return self.data[:,self._current_index]


class piecewise_LP_data():

    def __init__(self, rhs, objective):
        pass

    # should return minimum between objective and rhs
    def get_nextT(self):
        pass

    # should return next data from objective and/or rhs and indicator what data are returned
    def next_data(self):
        pass