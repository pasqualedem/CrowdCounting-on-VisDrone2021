import os

DIR_NAME_LENGHT = 3

sequences_end = [123, 179, 291, 331, 341, 358, 391, 410, 413, 536, 561, 564, 580,
                 587, 612, 619, 631, 634, 657, 660, 670, 1131, 1135, 1151, 1234,
                 1472, 1524, 1533, 1598, 1604, 1641, 1733, 1753, 1768, 1773, 1807]

pathRGB = '../../datasets/VisDrone2021/Train/RGB'
pathTIR = '../../datasets/VisDrone2021/Train/TIR'
pathGT = '../../datasets/VisDrone2021/Train/GT_'

img_rule = lambda x: '.jpg' in x
gt_rule = lambda x: '.xml' in x

gt_tir_split = lambda x: int(x.split('R')[0])
rgb_split = lambda x: int(x.split('.')[0])

cycle = [(pathRGB, img_rule, rgb_split), (pathTIR, img_rule, gt_tir_split), (pathGT, gt_rule, gt_tir_split)]

for c in cycle:

    low_limit = 1
    dir = 1

    for up_limit in sequences_end:

        files = os.listdir(c[0])
        files = list(filter(c[1], files))

        dir_name = str(dir).zfill(DIR_NAME_LENGHT)
        destination = os.path.join(c[0], dir_name)
        os.mkdir(destination)

        for f in files:
            file = f
            f = c[2](f)
            if low_limit <= f <= up_limit:
                os.replace(os.path.join(c[0], file), os.path.join(destination, file))

        low_limit = up_limit + 1
        dir += 1
