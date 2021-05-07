




import os, os.path
import shutil

def get_count_in_folder(pth):
    dirList = os.listdir(pth)[:-1] #removing last element which is .DS
    relativePathToDir = list(map(lambda x: os.path.join(pth, x), dirList))
    filesInDir = list(map(lambda x:os.listdir(x), relativePathToDir))
    count = list(map(lambda x:len(x), filesInDir))
    

    #debug
    if __name__ == '__main__':
        print(dirList)
        print(relativePathToDir[0])
        print(filesInDir)
        print(count)
        print("\n\n\n\n")
        
    return count


#not done
def balance_data(list_count, pth):
    dirList = os.listdir(pth)[:-1] #removing last element which is .DS
    relativePathToDir = list(map(lambda x: os.path.join(pth, x), dirList)) 
    largestNum = max(list_count)
    diffBetweenMax = list(map(lambda x: largestNum-x, list_count))
    

    #debug
    if __name__ == '__main__':
        print(diffBetweenMax)


if __name__ == '__main__':
    cnt = get_count_in_folder('../food11re/food11re/skewed_training/')
    balance_data(cnt, '../food11re/food11re/skewed_training/')
