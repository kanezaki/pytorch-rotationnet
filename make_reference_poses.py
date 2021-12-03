import argparse
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch Pose Reference')
parser.add_argument('--input1', metavar='DIR', help='path of input1')
parser.add_argument('--input2', metavar='DIR', help='path of input2')
parser.add_argument('--output_file', help="Output npy filename.")

vcand = np.load('vcand_case2.npy')
nview = 20

def main():
    args = parser.parse_args()

    f = open(args.input1)   # class.txt
    lines1 = f.readlines()  # filename and label
    f.close()

    f = open(args.input2)   # train_c_log.txt
    lines2 = f.readlines()  # predicted label and best pose
    f.close()

    numR = len(lines1) // len(lines2)

    # align sample images, each numR image has the same pose
    with open(args.output_file,'a') as f: 
        for i in range(len(lines2)):
            max_ang = int(lines2[i].split(' ')[-1][:-1])
            for j in range(numR):
                if numR == 12:
                    idx = j + max_ang
                    if idx > (numR - 1):
                        idx = idx - numR
                else:
                    idx = vcand[max_ang][j]
                print(lines1[i * numR + idx].split(' ')[0])  
                f.writelines(lines1[i * numR + idx].split(' ')[0]+'\n')
    return


if __name__ == '__main__':
    main()
