import re
def preproccessing(datapath):
    data = open(datapath, 'r')
    newdata = open('./Titanic_dataset.txt','w')
    agesum = 0
    num = 0
    for line in data.readlines():

        line = line.split(',')
        line1 = line[:3]
        line2 = line[5:]

        newline = ''
        for l in line1:
            newline += l+','
        for l in line2:
            newline += l+','
        print(newline[:-1])
        newdata.write(newline[:-1])
    #     lines = line.split('"')
    #
    #     age = lines[6][1:][:-1]
    #     # print(age)
    #
    #     if age!='NA':
    #         if float(age) - int(float(age))==0:
    #             agesum+=int(float(age))
    #             num+=1
    #
    # print(agesum/num)
    #     # print(lines[7])





if __name__ == '__main__':
    preproccessing('./dataset.txt')