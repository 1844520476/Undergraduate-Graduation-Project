txt = r'..\query\1.txt'
with open(rf'{txt}', 'r') as f:
    predict_result = f.readline()
print(f'{predict_result}')