'''
Created on 2021年5月24日
复数基
@author: gsk
'''
import numpy as np
# from PIL import Image
# import matplotlib.pyplot as plt
# import mpl_toolkits.axisartist as axisartist #导入坐标轴加工模块
import DCT
import AC
import DC
import Compress
import cv2

# 复数基的实部和虚部
A = -1
B = 1

# 将图片转化成复数
def imaToComplex(Bstr):
    P = 0
    Q = 0
    imaBinaryList = Bstr

    # 输出数组
    arryNum = len(imaBinaryList)
    # 创建一个存放r,s的数组RS,初始元素为0
    RS = np.zeros((1, 2)).astype(int).astype(str)

    # 迭代求r,s
    r0 = '1'
    s0 = '0'

    RS[0, 0] = r0
    RS[0, 1] = s0

    # 将r，s迭代存放如数组
    for i in range(1, arryNum):
        t1 = A * int(RS[i - 1, 0]) - B * int(RS[i - 1, 1])
        t2 = B * int(RS[i - 1, 0]) + A * int(RS[i - 1, 1])

        r = str(t1)
        s = str(t2)

        RS = np.append(RS, [[r, s]], axis=0)
        # print("正在计算复数的实部和虚部共:", arryNum, "第", i + 1, "个")

    # 计算复数的实部P，虚部Q
    for i in range(arryNum):
        e = imaBinaryList[arryNum-i-1:arryNum-i]
        e = int(e)
        P = int(P) + e * int(RS[i, 0])
        Q = int(Q) + e * int(RS[i, 1])

    p0 = P
    q0 = Q

    return p0, q0, arryNum

# 将高斯整数转换成0-1序列 将复数转化成图片
def convertShow(p0, q0, arryNum):
    eList = ''

    reduce = p0 - q0
    e0 = abs(reduce % 2)
    e0 = str(e0)

    # 存放二进制的数组
    eList = eList + e0
    num = 1

    # 循环求出e
    for i in range(1, arryNum):
        a = p0 * A + q0 * B - int(e0) * A
        b = -p0 * B + q0 * A + int(e0) * B
        p1 = a // 2
        q1 = b // 2
        p0 = p1
        q0 = q1

        temp = p1 - q1
        e0 = temp % 2
        e0 = str(e0)
        eList = eList + e0
        num += 1

    eList = eList[::-1]    # 反转字符串
    return eList

dct = DCT.DCT()
AC = AC.AC()
Compress = Compress.Compress()
DC = DC.DC()
img = cv2.imread('./ima/lenna8.png', 0)
# img[7][7] = 0
# 进行DCT变换并量化
quanMartixInt, C_temp, E_Array, qbits, Qstep, C_tempTran = dct.dctConvert(img)
# print("DCT变换后并量化的矩阵：", quanMartixInt)
# 哈夫曼编码
Bstr = dct.huffman(quanMartixInt, AC, Compress)
print("对量化后的矩阵进行huffman编码后的0-1序列：", Bstr)
lenStr = len(Bstr)
print("对量化后的矩阵进行huffman编码后的0-1序列的长度：", lenStr)
print("-----------------------------------------------------------------------------")

p0, q0, arryNum = imaToComplex(Bstr)
strP = str(p0)
strQ = str(q0)
if(p0<0):
    lenP = len(strP) - 1
else:
    lenP = len(strP)
if(q0<0):
    lenQ = len(strQ) - 1
else:
    lenQ = len(strQ)
print("经过DCT变换后转化成复数的实部：", p0)
print("实部的长度：", lenP)
print("-----------------------------------------------------------------------------")
print("经过DCT变换后转化成复数的虚部：", q0)
print("虚部的长度：", lenQ)

# 还原字符串
HuffStr = convertShow(p0, q0, arryNum)
# 解码
quanMart = dct.reHuffman(HuffStr, img, DC)
# 反量化
X = dct.inQuantity(E_Array, Qstep, qbits, quanMart, C_temp, C_tempTran)
# 展示图片
dct.showImg(img, quanMartixInt, X)
