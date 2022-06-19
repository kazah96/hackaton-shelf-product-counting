with open('datasets/PrivateTestSet/output.csv', mode='r') as file:
    lines = file.readlines()


with open('submission_1.csv', mode='w') as file:
    file.writelines([line for line in lines if line != '\n'])
