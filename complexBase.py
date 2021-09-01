'''
Created on 2021年5月24日
转图片的灰度
@author: gsk
'''
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# 将图片转化成灰度图片并转化为二维数组
def pretreatment(ima):
    ima = ima.convert('L')  # 转化为灰度图像
    im = np.array(ima)  # 转化为二维数组
    # print(im.shape)
    showimg(im, True)  # 显示图片

    return im

# 显示图片
def showimg(img, isgray=False):
    plt.axis("off")
    if isgray == True:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img)
    plt.show()

# 复数基的实部和虚部
A = -1
B = 1

# 将图片转化成复数
def imaToComplex(im):
    P = 0
    Q = 0
    imaBinaryList = ''
    # 将图片数组转换成二进制序列
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            # 将图片数组转化成二进制字符串
            imaBinaryList = imaBinaryList + format(im[i, j], "b").zfill(8)

    print("普通方法将图片转换成二进制序列: ", imaBinaryList)
    # print(type(imaBinaryList))
    lengtnStr = len(imaBinaryList)
    print("普通方法将图片转换成二进制序列的长度: ", lengtnStr)
    print("-----------------------------------------------------------------------------")

    # 输出数组
    arryNum = len(imaBinaryList)
    # print("图片数组中元素的个数：", arryNum)

    # 创建一个存放r,s的数组RS,初始元素为0
    RS = np.zeros((1, 2)).astype(int).astype(str)
    # print("存放r，s的数组长度：", RS.shape)

    # 创建存放复数转化为二进制的二维数组
    # binary = np.zeros((im.shape[0], im.shape[1])).astype(int)

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
        # row = i // im.shape[1]
        # col = i % im.shape[1]
        P = int(P) + e * int(RS[i, 0])
        Q = int(Q) + e * int(RS[i, 1])

    p0 = P
    q0 = Q
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

    print("普通方法转化成复数的实部：", p0)
    print("实部的长度：", lenP)
    print("-----------------------------------------------------------------------------")
    print("普通方法转化成复数的虚部：", q0)
    print("虚部的长度：", lenQ)


    return p0, q0, arryNum

# 将图片进行猫脸变换的函数
# def arnoldConvert(imaArray, destArray, count, plen, qlen):
#     num = count
#     origin = imaArray
#     newX = 0
#     newY = 0
#     # 点对应的y坐标
#     yAxis = []
#     # 点对应的x坐标
#     xAxis = []
#
#     while (count > 0):
#         for row in range(imaArray.shape[0]):
#             for col in range(imaArray.shape[1]):
#                 newX = (col + row) % imaArray.shape[0]
#                 newY = (col * 2 + row) % imaArray.shape[0]
#                 destArray[newX][newY] = imaArray[row][col]
#
#         print("第：", num - count + 1, "次变换")
#         # showimg(destArray, True)
#         # 将图片数组转化成复数
#         p, q, arryNum = imaToComplex(destArray)
#
#         # 得到点的x，y坐标
#         # x = p/pMax
#         # y = q/qMax
#         x = p
#         y = q
#         xAxis.append(str(x))
#         yAxis.append(str(y))
#
#         imaArray = destArray
#         count = count - 1
#     a = xAxis
#     b = yAxis
#     print("x的坐标：", a)
#     print("y的坐标：", b)
#     #将点绘制到界面上
#     toGraph(a, b, plen, qlen)

# 将图片转化成复数对应的点显示出来
# def toGraph(x, squares, plen, qlen):
#     fig = plt.figure(figsize=(15, 13))  # 新建画布
#     ax = axisartist.Subplot(fig, 111)  # 使用axisartist.Subplot方法创建一个绘图区对象ax
#     fig.add_axes(ax)  # 将绘图区对象添加到画布中
#     # squares = [1, 4, 9, 16, 25]
#     # x = [1, 2, 3, 4, 5]
#     plt.title("show of arnold", fontsize=20)
#     plt.xlabel("P", fontsize=12)
#     plt.ylabel("Q", fontsize=12)
#     plt.tick_params(axis='both', labelsize=10)
#     # plt.axes([-pow(10,plen-1),pow(10,plen-1),-pow(10,plen-1),pow(10,plen-1)])
#     for i in range(len(x)):
#         ax.annotate(text=i, xy=(int(x[i]), int(squares[i])), xytext=(int(x[i]), int(squares[i])))  # 标注x轴 xy为标记点的位置，xytext为文本的位置
#         ax.annotate(text='y1', xy=(0, 1.0), xytext=(-0.5, 1.0))  # 标注y轴
#         a1x = int(x[i])
#         a1y = int(squares[i])
#         plt.scatter(a1x, a1y)
#         # plt.show()
#
#     # plt.scatter(x, squares)
#     plt.show()

# 将高斯整数转换成0-1序列 将复数转化成图片
def convertShow(p0, q0, arryNum, image):
    eList = ''

    reduce = p0 - q0
    e0 = abs(reduce % 2)
    e0 = str(e0)

    # 存放二进制的数组
    eList = eList + e0
    # eList.append(e0)
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
        # eList.append(e0)
        eList = eList + e0
        num += 1

    eList = eList[::-1]    # 反转字符串

    ArryNum = len(eList) // 8
    eIndex = 0
    AValue = 0  # 复原数组中的值

    # 创建图片复原数组
    recoverArr = np.zeros((image.shape[0], image.shape[1])).astype(int)

    for j in range(ArryNum):
        ArryVal = eList[eIndex:eIndex + 8]
        eIndex += 8
        ArrValue = int(ArryVal[0])*pow(2, 7) + int(ArryVal[1])*pow(2, 6) + int(ArryVal[2])*pow(2, 5) + int(ArryVal[3]) * pow(2, 4) + int(ArryVal[4]) * pow(2, 3) + int(ArryVal[5])*pow(2, 2) + int(ArryVal[6])*pow(2, 1) + int(ArryVal[7])*pow(2, 0)

        row = j // image.shape[1]
        col = j % image.shape[1]

        recoverArr[row, col] = ArrValue
        # print(recoverArr)
    return eList
    # showimg(recoverArr, True)

# 将数组中的元素转化为8位的二进制数字，并拼接成字符串
# ima = Image.open('./ima/dog04.png')  # 读入图像
# im = pretreatment(ima)  # 调用图像预处理函数
#
# ima1 = Image.open('./ima/dog32.png')  # 读入图像
# im1 = pretreatment(ima1)  # 调用图像预处理函数

# print(im)
# p0, q0, arryNum = imaToComplex(im)
# p1, q1, arryNum1 = imaToComplex(im1)
#
# p = p0 - p1
# q = q0 - q1

# array = convertShow(p, q, im, arryNum)
# print(array, "复原后的数组")
# showimg(array, True)



# 创建一个转换数组
# destArray = np.zeros((im.shape[0], im.shape[1])).astype(int)
# maxArray = np.zeros((im.shape[0], im.shape[1])).astype(int)
# for i in range(im.shape[0]):
#     for j in range(im.shape[1]):
#         maxArray[i][j] = 255

# # 求出图片对应最大最小复数
# pMin, qMin, arryNum1 = imaToComplex(destArray)
# pMax, qMax, arryNum2 = imaToComplex(maxArray)
# if (pMax<0):
#     p1Max = abs(pMax)
# if (qMax<0):
#     q1Max = abs(qMax)
# plen = len(str(p1Max))
# qlen = len(str(q1Max))
# arnoldConvert(im, destArray, 2, plen, qlen)

# dct = DCT.DCT()
# AC = AC.AC()
# Compress = Compress.Compress()
# DC = DC.DC()
# img = cv2.imread('./ima/lennaNose_8.png', 0)
# # 进行DCT变换并量化
# quanMartixInt, C_temp, E_Array, qbits, Qstep, C_tempTran = dct.dctConvert(img)
# # 哈夫曼编码
# Bstr = dct.huffman(quanMartixInt, AC, Compress)
# lenStr = len(Bstr)
# print('0-1序列长度', lenStr)
# p0, q0, arryNum = imaToComplex(Bstr)
#
# # 还原字符串
# HuffStr = convertShow(p0, q0, arryNum)
# # 解码
# quanMart = dct.reHuffman(HuffStr, img, DC)
# # 反量化
# X = dct.inQuantity(E_Array, Qstep, qbits, quanMart, C_temp, C_tempTran)
# # 展示图片
# dct.showImg(img, quanMartixInt, X)
img = cv2.imread('./ima/lenna8.png', 0) # 读入图像
# img = pretreatment(ima)  # 调用图像预处理函数
# img[7][7] = 0
plt.imshow(img, 'gray')
plt.show()
print(img)

imaToComplex(img)
