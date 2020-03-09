

with open("testing/test_split/scales.txt", "r") as infile:
    for line in infile.readlines():
        case, scale = (line.split(","))
        case_number = int(case.split(":")[1])
        scale = float(scale.split(":")[1])
        print(case_number, scale)
