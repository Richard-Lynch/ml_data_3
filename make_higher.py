#!/usr/local/bin/python3


def readlines(filename, **kwargs):
    f = open(filename)
    feature_names = f.readline().replace(',', ';').replace(' ', '_').split(";")
    print(feature_names)
    feature_names[-1] = feature_names[-1].replace('\n', '')
    print(feature_names)
    shortFile = []
    i = 0
    while True:
        row = f.readline().replace(',', ';')
        limit = i
        if row == "":
            print("hit limit:", i)
            break
        shortFile.append(row.split(';'))
        i += 1
    if shortFile:
        with open("squared_" + filename, 'w') as fo:
            new_features = []
            for value in feature_names:
                new_features.append(value)
                new_features.append(value + '_squared')
            del new_features[-1]
            fo.write(", ".join(new_features) + "\n")
            for row in shortFile:
                print('row', row)
                new_row = []
                for value in row:
                    print('v', value)
                    print(type(value))
                    new_row.append(str(float(value)))
                    new_row.append(str(float(value)**2))
                del new_row[-1]
                new_row_s = ", ".join(new_row)
                fo.write(new_row_s + '\n')


if __name__ == "__main__":
    readlines("winequality-red.csv")
