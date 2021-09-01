import cv2
import numpy as np
import matplotlib.pyplot as plt
import AC
import DC
import Compress

class DCT:

    # dct变换
    def dctConvert(self, img):
        # 残差矩阵中的元素
        a = 1 / (2 * np.sqrt(2))
        b = (1 / 2) * np.cos(np.pi / 16)
        c = (1 / 2) * np.cos((2 * np.pi) / 16)
        d = (1 / 2) * np.cos((3 * np.pi) / 16)
        e = (1 / 2) * np.cos((5 * np.pi) / 16)
        f = (1 / 2) * np.cos((6 * np.pi) / 16)
        g = (1 / 2) * np.cos((7 * np.pi) / 16)

        # 残差矩阵每行提取公因数后的元素
        k1 = 5
        k2 = 6
        k3 = 4
        k4 = 1
        k5 = 2

        g0 = g / k4

        V = [[a, g0, f, g0, a, g0, f, g0]]
        V = np.array(V)
        V_Tran = np.transpose(V)
        E_Array = np.dot(V_Tran, V)

        C_temp = [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [k1, k2, k3, k4, -k4, -k3, -k2, -k1],
            [k5, 1, -1, -k5, -k5, -1, 1, k5],
            [k2, -k4, -k1, -k3, k3, k1, k4, -k2],
            [1, -1, -1, 1, 1, -1, -1, 1],
            [k3, -k1, k4, k2, -k2, -k4, k1, -k3],
            [1, -k5, k5, -1, -1, k5, -k5, 1],
            [k4, -k3, k2, -k1, k1, -k2, k3, -k4]
        ]

        C_temp = np.array(C_temp)
        # print(C_temp.shape)
        m, n = img.shape
        # print(m)
        N = n

        dctIma = np.dot(C_temp, img)
        # 转置矩阵
        C_tempTran = np.transpose(C_temp)
        testP = np.dot(C_temp, C_tempTran)
        bb = np.dot(testP, E_Array)
        dctIma1 = np.dot(dctIma, np.transpose(C_temp))

        # 变换后的矩阵
        quanMartix = np.zeros((8, 8)).astype('float')

        # 对系数矩阵进行量化
        Qstep = 4
        QP = 16
        qbits = 15 + np.floor(QP / 6)
        for i in range(8):
            for j in range(8):
                MF = (E_Array[i][j] / Qstep) * pow(2, qbits)
                quanMartix[i][j] = dctIma1[i][j] * (MF / pow(2, qbits))
        quanMartixInt = np.round(quanMartix).astype('int')
        # print("进行DCT变换后的矩阵", quanMartixInt)
        return quanMartixInt, C_temp, E_Array, qbits, Qstep, C_tempTran

    #  对量化后的矩阵进行哈夫曼编码
    def huffman(self, quanMartixInt, AC, Compress):
        # zigzagArry = zigzag(quanMartixInt).astype('int')

        DCnum = quanMartixInt[0][0]

        ACnum = AC.ZScan(quanMartixInt)
        ACn = AC.RLC(ACnum)

        # print('Z : ' + str(ACnum))
        # print('DC: ' + str(DCnum))
        # print('AC: ' + str(ACn))
        # print('AClen: ' + str(len(ACn)))

        Bstr = Compress.AllCompressY(DCnum, ACn)
        # print(Bstr)
        # print(len(Bstr))
        return Bstr

    # 解码
    def reHuffman(self, Bstr, img, DC):
        DCY, ACY = Compress.encoding(Bstr, img.shape[0], img.shape[1])
        # print(DCY, "直流系数")
        # print(ACY[0], "交流系数")
        # print(len(ACY[0]))

        blocks = DC.DPCM2(DCY)
        block = np.array(blocks)
        # 解码后AC系数
        ACSer = AC.RLE(ACY[0])
        ACSer = np.array(ACSer)
        AC.Z2Tab(ACY[0], block[0])
        quanMart = block[0]
        return quanMart

    # 反量化
    def inQuantity(self, E_Array, Qstep, qbits, quanMart, C_temp, C_tempTran):
        convertDctIma1 = np.zeros((8, 8)).astype('float')
        for i in range(8):
            for j in range(8):
                MF = (E_Array[i][j] / Qstep) * pow(2, qbits)
                convertDctIma1[i][j] = quanMart[i][j] * pow(2, qbits) / MF
        # print(convertDctIma1)

        convertDctImaInt = np.round(convertDctIma1).astype('int')
        convertDctImaTran = np.transpose(convertDctImaInt)

        inverseP1 = np.linalg.inv(C_temp)
        inveraeTransP1 = np.linalg.inv(C_tempTran)
        X0 = np.dot(inverseP1, convertDctIma1)
        X1 = np.dot(X0, inveraeTransP1)
        X = np.round(X1).astype('int')
        return X

    # 绘图
    def showImg(self, img, quanMartixInt, X):
        plt.subplot(231)
        plt.imshow(img, 'gray')
        plt.title('original image')
        plt.xticks([]), plt.yticks([])

        plt.subplot(232)
        plt.imshow(quanMartixInt)
        plt.title('DCT')
        plt.xticks([]), plt.yticks([])

        plt.subplot(233)
        plt.imshow(X, 'gray')
        plt.title('recover')
        plt.xticks([]), plt.yticks([])

        # print("dct变换后的矩阵", dctIma)
        plt.show()
''''''
dct = DCT()
AC = AC.AC()
Compress = Compress.Compress()
DC = DC.DC()
img = cv2.imread('./ima/lenna8.png', 0)
# 进行DCT变换并量化
quanMartixInt, C_temp, E_Array, qbits, Qstep, C_tempTran = dct.dctConvert(img)
# 哈夫曼编码
Bstr = dct.huffman(quanMartixInt, AC, Compress)
# 解码
quanMart = dct.reHuffman(Bstr, img, DC)
# 反量化
X = dct.inQuantity(E_Array, Qstep, qbits, quanMart, C_temp, C_tempTran)
# 展示图片
# dct.showImg(img, quanMartixInt, X)



