import os
import pickle as pk


def check_create_save_dir(save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        print('SAVE DIRECTORY ALREADY EXISTS, POTENTIALLY TO OVERWRITE EXISTING DATA.')


def unpack_file(filename, dataPath):

    data_fn = os.path.abspath(os.path.join(dataPath, filename))

    names = []
    data = []

    f = open(data_fn, 'rb')

    read = True
    while read == True:
        dat_temp = pk.load(f)
        if dat_temp == 'end':
            read = False
        else:
            # print(isinstance(dat_temp, str))
            if isinstance(dat_temp, str):
                names.append(dat_temp)
                data.append(pk.load(f))
                # print(data)
    f.close()

    return names, data


def save_non_tf_data(names, data, filename, savePath):

    check_create_save_dir(savePath)

    data_fn = os.path.abspath(os.path.join(savePath, filename))

    f = open(data_fn, 'wb')

    for i in range(0,len(names)):
        pk.dump(names[i],f)
        pk.dump(data[i], f)
    pk.dump('end',f)

    f.close()
    print('File__'+str(filename)+'__saved to__'+data_fn)