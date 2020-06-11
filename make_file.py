
with open('/home/yankun/spamGAN/dividedData/unlabel50_label70/train_review.txt', 'a') as txtf, open('/home/yankun/spamGAN/dividedData/unlabel50_label70/train_label.txt', 'a') as labf:
    with open('/home/yankun/spamGAN/dividedData/unlabeled50_review.txt', 'r') as f:
        texts = f.readlines()
        for line in texts:
            txtf.write(line)
            labf.writelines(str(-1) + '\n')
