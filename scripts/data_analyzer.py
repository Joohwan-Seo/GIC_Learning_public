import pandas as pd
import matplotlib.pyplot as plt

def main(file_name):
    print(file_name)
    df = pd.read_csv(file_name)
    
    df.head()
    t1 = 'eval/Returns Mean'
    col = df[t1]

    filtered = col.rolling(window=50).mean()

    print(col.shape)

    plt.plot(col)
    plt.plot(filtered)
    plt.show()

    return 0


if __name__ == "__main__":
    log = '~/deeprl/research/GIC-RL/data/GIC-RL-nominal-expert-long/GIC_RL_nominal_expert_long_2023_01_27_22_27_18_0000--s-0/'
    file_name = log + 'progress.csv'

    main(file_name)