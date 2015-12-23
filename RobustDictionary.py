class RobustDictionary(dict):
    def __getitem__(self,key):
        try:
            result = dict.__getitem__(self,key)
            if (type(result) is dict):
                return RobustDictionary(result)
            elif (type(result) is list):
                return RobustList(result)
            else:
                return result
        except Exception:
            return None

class RobustList(list):
    def __getitem__(self,index):
        result = list.__getitem__(self,index)
        if (type(result) is not dict):
            return result
        else:
            return RobustDictionary(result)
