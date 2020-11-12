
lines=open('train.txt', 'r', encoding='utf-8').readlines()
fout=open('train_pro.txt','w', encoding='utf-8')

i=0
step=0
# para_size = [10, 40, 50, 30, 20]
for line in lines:
    i += 1
    words = line.strip().split()
    if len(words) < 1 or i % 50 == 0 or i % 70 == 0:
        # if step < 1:
        fout.write('\n')
        # step += 1

    else:
        if len(words) > 5 and len(words) < 128:
            # step = 0
            for word in words:
                if word == '<unk>':
                    word = '[UNK]'
                if word != '.':
                    fout.write(word)
                    fout.write(' ')
                else:
                    fout.write('.')
                    fout.write('\n')
