from Tools import InteractionDataset

if __name__ == "__main__":
    itc = InteractionDataset("./data/split_s0_s1_s2/val.csv")
    print(itc.scenario)
