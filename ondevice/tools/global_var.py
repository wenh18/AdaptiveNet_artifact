def _init():  
    global global_dict
    global_dict = {}


def set_value(key, value):
    global_dict[key] = value


def get_value(key):
    try:
        return global_dict[key]
    except:
        print('Read '+key+'failure\r\n')