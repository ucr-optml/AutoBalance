

def reopen_file(filename,checkpoint):
    temp=open(filename).readlines()
    file=open(filename,'w')
    file.writelines(temp[:checkpoint])
    file.flush()
    return file