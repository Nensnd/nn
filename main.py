import random
from training import *
import numpy as np

lst = ['', 'а', 'б', 'в', 'г', 'д', 'е', 'е', 'ж', 'з', 'и', 'й', 'к', 'л', 'м', 'н', 'о', 'п', 'р', 'с', 'т', 'у', 'ф',
       'х', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ы', 'ь', 'э', 'ю', 'я', ' ', '.', ',', ':', '?', '0', '1', '2', '3', '4', '5',
       '6', '7', '8', '9']
print(len(lst))
tokenized_inp = []


def give_token(text):
    lk = []
    for i in text:
        lk.append(float(lst.index(i)) / 49.0)
    if len(lk) < 256:
        for i in range(257 - len(lk) - 1):
            lk.append(0.0)
    return lk


def matrixs(m1, m2):
    mr = []
    for i in range(len(m1) - 1):
        mr.append(m1[i] - m2[i])
    return mr


def matrixm(m1, m2):
    mr = []
    for i in range(len(m1) - 1):
        mr.append(m1[i] * m2[i])
    return mr


def delta(errors):
    dr = []
    for i in errors:
        dr.append(i * (1.0 - i))


def g_t_answer(data):
    ans = ''
    print(data)
    for i in range(len(data)):
        try:
            ans = ans + lst[int(((data[i] * 49.0) // 1))]
        except:
            pass
    return ans


'''def tr_one(w, d, l, r):
    for j in range(len(d) - 1):
        for i in range(2048):
            w[j * i] = w[j * i] - (r[i] * d[i] * l)'''


def sig(x):
    return float(1 / (1 + np.exp(-x)))


def laymatcher(nums, weights):
    preres = []
    res = 0
    results = []
    for i in range(256):
        for j in range(256):
            preres.append(weights[i * 256 + j] * nums[j])
        for j in preres:
            res += j
        preres = []
        results.append(float(sig(res)))
    print(results)
    print(len(results))
    return results


raw_inp = input("Ваш запрос: ")

if len(raw_inp) > 256:
    print("Ваш запрос слишком большой! \n Пожалуйста, сократите запрос до 2048-ми символов.")
else:
    tokenized_inp = give_token(raw_inp)

print(tokenized_inp)
print(len(tokenized_inp))

w_1 = []
r_1 = []
w_2 = []
r_2 = []
w_3 = []
r_3 = []
w_4 = []
r_4 = []
w_5 = []
r_5 = []
w_6 = []
r_6 = []
w_7 = []
r_7 = []

for i in range(256 * 256):
    w_1.append(random.uniform(-1.0, 1.0))
    w_2.append(random.uniform(-1.0, 1.0))
    w_3.append(random.uniform(-1.0, 1.0))
    w_4.append(random.uniform(-1.0, 1.0))
    w_5.append(random.uniform(-1.0, 1.0))
    w_6.append(random.uniform(-1.0, 1.0))
    w_7.append(random.uniform(-1.0, 1.0))

# Соединяем слои нейронной сети
# 1 слой
r_1 = laymatcher(tokenized_inp, w_1)

# 2 слой
r_2 = laymatcher(r_1, w_2)
r_1 = []

# 3 слой
r_3 = laymatcher(r_2, w_3)
r_2 = []

# 4 слой
r_4 = laymatcher(r_3, w_4)
r_3 = []

# слой 5
r_5 = laymatcher(r_4, w_5)
r_4 = []

# слой 6
r_6 = laymatcher(r_5, w_6)
r_5 = []

# слой 7
r_7 = laymatcher(r_6, w_7)
r_6 = []

print(g_t_answer(r_7))
r_7 = []

def training(tinput, toutput, lr):
    for h in range(len(tinput) - 1):
        print(h)
        tokenized_inp = give_token(tinput[h])
        tokenized_out = give_token(toutput[h])
        # Соединяем слои нейронной сети
        # 1 слой
        r_1 = laymatcher(tokenized_inp, w_1)

        # 2 слой
        r_2 = laymatcher(r_1, w_2)

        # 3 слой
        r_3 = laymatcher(r_2, w_3)

        # 4 слой
        r_4 = laymatcher(r_3, w_4)

        # слой 5
        r_5 = laymatcher(r_4, w_5)

        # слой 6
        r_6 = laymatcher(r_5, w_6)

        # слой 7
        r_7 = laymatcher(r_6, w_7)

        errors_1 = matrixs(r_7, give_token(toutput[h]))
        deltas_1 = delta(errors_1)

        for de in range(256):
            print('started')
            for p in range(256):
                for d in range(256):
                    w_7[(d * 256) + p] = w_7[(d * 256) + p] - r_6[d] * deltas_1[de] * lr



training(tinput, toutput, lr)
