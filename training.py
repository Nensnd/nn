import numpy as np
import random
tinput = ['1']
toutput = ['2']
lr = 0.3
lst = ['', 'а', 'б', 'в', 'г', 'д', 'е', 'е', 'ж', 'з', 'и', 'й', 'к', 'л', 'м', 'н', 'о', 'п', 'р', 'с', 'т', 'у', 'ф',
       'х', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ы', 'ь', 'э', 'ю', 'я', ' ', '.', ',', ':', '?', '0', '1', '2', '3', '4', '5',
       '6', '7', '8', '9']



w_1 = []
w_2 = []
w_3 = []
w_4 = []
w_5 = []
w_6 = []
w_7 = []

for i in range(256 * 256):
    w_1.append(random.uniform(-1.0, 1.0))
    w_2.append(random.uniform(-1.0, 1.0))
    w_3.append(random.uniform(-1.0, 1.0))
    w_4.append(random.uniform(-1.0, 1.0))
    w_5.append(random.uniform(-1.0, 1.0))
    w_6.append(random.uniform(-1.0, 1.0))
    w_7.append(random.uniform(-1.0, 1.0))
def delta(errors):
    dr = []
    for i in errors:
        dr.append(i * (1.0 - i))
    return dr

def sdelta(errors):
    dr = []
    for i in errors:
        dr.append(sig(i) * (1.0 - sig(i)))
    return dr

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

def error(delta, w):
    ld = []
    for i in range(256):
        for j in range(255):
            ld.append(w[i * 256 + j] * delta[j])
    return ld

for h in range(len(tinput)):
    print(h)
    tokenized_inp = give_token(tinput[h])
    tokenized_out = give_token(toutput[h])
    # Соединяем слои нейронной сети
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

    #обучаю 1-й слой
    errors_1 = matrixs(r_7, give_token(toutput[h]))
    deltas_1 = delta(errors_1)
    print(deltas_1)
    for i in range(255):
        for p in range(256):
            for d in range(256):
                w_7[(d * 256) + p] = w_7[(d * 256) + p] - r_6[d] * deltas_1[i] * lr

    #обучаю второй слой
    errors_2 = error(deltas_1, w_6)
    deltas_2 = sdelta(errors_2)
    for i in range(255):
        for p in range(256):
            for d in range(256):
                w_6[(d * 256) + p] = w_6[(d * 256) + p] - r_5[d] * deltas_2[i] * lr

    #обучаю 3 слой
    errors_3 = error(deltas_2, w_5)
    deltas_3 = sdelta(errors_3)
    for i in range(255):
        for p in range(256):
            for d in range(256):
                w_5[(d * 256) + p] = w_5[(d * 256) + p] - r_4[d] * deltas_3[i] * lr

    #обучаю 4 слой
    errors_4 = error(deltas_3, w_4)
    deltas_4 = sdelta(errors_4)
    for i in range(255):
        for p in range(256):
            for d in range(256):
                w_4[(d * 256) + p] = w_4[(d * 256) + p] - r_3[d] * deltas_4[i] * lr

    #обучаю 5 слой
    errors_5 = error(deltas_4, w_3)
    deltas_5 = sdelta(errors_5)
    for i in range(255):
        for p in range(256):
            for d in range(256):
                w_3[(d * 256) + p] = w_3[(d * 256) + p] - r_2[d] * deltas_5[i] * lr

    #обучаю 6 слой
    errors_6 = error(deltas_5, w_2)
    deltas_6 = sdelta(errors_6)
    for i in range(255):
        for p in range(256):
            for d in range(256):
                w_2[(d * 256) + p] = w_2[(d * 256) + p] - r_1[d] * deltas_6[i] * lr

    #обучаю 7 слой
    errors_7 = error(deltas_6, w_1)
    deltas_7 = sdelta(errors_7)
    for i in range(255):
        for p in range(256):
            for d in range(256):
                w_1[(d * 256) + p] = w_1[(d * 256) + p] - tokenized_inp[d] * deltas_7[i] * lr


