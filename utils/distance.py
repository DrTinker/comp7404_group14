def hamming(a, b):
    # compute and return the Hamming distance between the integers
    str1 = ''
    for item in a:
        if item>0:
            str1 += '1'
        else:
            str1 += '0'
    
    str2 = ''
    for item in b:
        if item>0:
            str2 += '1'
        else:
            str2 += '0'
    return bin(int(str1) ^ int(str2)).count("1")