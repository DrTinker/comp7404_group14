def WriteList(data, path, file_name):
    file=open(path + '/' + file_name,'w')
    for item in data:
        file.write(item+'\n')
    file.close()


def ReadList(file_path):
    data = []
    file = open(file_path, 'r')
    