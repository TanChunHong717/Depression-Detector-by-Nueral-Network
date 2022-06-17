import depressDetect

if __name__ == "__main__":
    print("Now we testing on the whole dataset(including traing set and validation set)")
    print("Testing all the user in codition group, they all should be classified as depress")
    for i in range(1, 24):
        print(depressDetect.deep_depression_detector("data\condition\condition_{}.csv".format(i)))
    print("\nTesting all the user in control group, they all should be classified as non depress")
    for i in range(1, 32):
        print(depressDetect.deep_depression_detector("data\control\control_{}.csv".format(i)))