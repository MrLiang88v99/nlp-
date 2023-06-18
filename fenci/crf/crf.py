import codecs
import sys
#将训练语料标注成B M E S存储
def character_tagging(input_file,output_file):
    input_data = open(input_file,'r',encoding='UTF-8')
    output_data = open(output_file,'w',encoding='UTF-8')
    for line in input_data.readlines():
        word_list = line.strip().split()
        for word in word_list:
            if len(word)==1:
                output_data.write(word+'\tS\n')
            else:
                output_data.write(word[0]+'\tB\n')
                for w in word[1:len(word)-1]:
                    output_data.write(w+'\tM\n')
                output_data.write(word[len(word)-1]+'\tE\n')
        output_data.write('\n')
    input_data.close()
    output_data.close()
# if __name__ == '__main__':
#
#     input_file = "C:\\Users\\26233\\Desktop\\nlp\\dataset\\icwb2-data\\training\\pku_training.utf8"
#     output_file = "F:\\nlp\dataset\\fenci\\fenci.txt"
#     character_tagging(input_file, output_file)
# # python make_crf_train.py ../icwb2-data/training/msr_training.utf8 msr_training.tagging4crf.utf8

def character_split(input_file, output_file):
    input_data = open(input_file, 'r',encoding='UTF-8-sig')
    output_data =open(output_file, 'w',encoding='UTF-8-sig')
    for line in input_data.readlines():
        for word in line.strip():
            word = word.strip()
            if word:
                output_data.write(word + "\tB\n")
        output_data.write("\n")
    input_data.close()
    output_data.close()
# if __name__ == '__main__':
#
#     input_file = "C:\\Users\\26233\\Desktop\\nlp\\dataset\\icwb2-data\\testing\\pku_test.utf8"
#     output_file = "F:\\nlp\dataset\\fenci\\fenci-test.txt"
#     character_split(input_file, output_file)
# # python make_crf_train.py ../icwb2-data/training/msr_training.utf8 msr_training.tagging4crf.utf8


#分完词后将文本处理会原段落形式
def character_2_word(input_file,output_file):
    input_data = open(input_file, 'r',encoding= 'UTF-8-sig')
    output_data = open(output_file, 'w', encoding='UTF-8-sig')
    for line in input_data.readlines():
        if line == "\n":
            output_data.write("\n")
        else:
            char_tag_pair = line.strip().split('\t')
            char = char_tag_pair[0]
            tag = char_tag_pair[2]
            if tag == 'B':
                output_data.write(' ' + char)
            elif tag == 'M':
                output_data.write(char)
            elif tag == 'E':
                output_data.write(char + ' ')
            else: # tag == 'S'
                output_data.write(' ' + char + ' ')
    input_data.close()
    output_data.close()
# if __name__ == '__main__':
#
#     input_file = "F:\\app\\crf++\\CRF++-0.58\\out_crftag.utf8"
#     output_file = "F:\\nlp\dataset\\fenci\\fenci-test-huanyuan.txt"
#     character_2_word(input_file, output_file)
# # python make_crf_train.py ../icwb2-data/training/msr_training.utf8 msr_training.tagging4crf.utf8

def accuracy():
    fp=open("C:\\Users\\26233\\Desktop\\nlp\\dataset\\icwb2-data\\gold\\pku_test_gold.utf8","r",encoding="UTF-8-sig")
    count=0
    count_gold=0
    count_test=0
    with open("F:\\nlp\dataset\\fenci\\fenci-test-huanyuan.txt","r",encoding="UTF-8-sig") as lines_total:
        for line1 in lines_total:
            i=0
            j=0
            if len(line1) == 0:          #为空时 结束遍历
                fp.readline()
                continue
            line1=line1.split()
            count_test+=len(line1)
            line2=fp.readline()
            if len(line2) == 0:          #为空时 结束遍历
                continue
            line2=line2.split()
            count_gold+=len(line2)
            while i< len(line1) and j<len(line2):
                if line1[i]==line2[j]:
                    count+=1                 #计数
                    i=i+1
                    j=j+1
                else:
                    len_i = len(line1[i])
                    len_j = len(line2[j])
                    while len_i != len_j:
                        if len_i>len_j:
                            j=j+1
                            len_j = len_j+len(line2[j])
                        else :
                            i=i+1
                            len_i=len_i+len(line1[i])
                    i+=1
                    j+=1
    fp.close()
    lines_total.close()
    Precision=count/count_test
    Recall=count/count_gold
    total=Precision+Recall
    F=(2*Precision*Recall)/total
    print('CRF分词结束，计算精确度')
    print('评测结果：\n')
    print('正确率：%-12.20f\n召回率：%-12.20f\nF值：%-12.20f\n'%(Precision,Recall,F))
    return(Precision,Recall,F)
accuracy()