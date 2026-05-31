import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    df = pd.read_csv('./checkpoints/record.csv')
    cols = df.columns.tolist()
    cols[0] = 'epoch'
    df.columns = cols
    x = df[cols[0]]
    for i in range(1, len(cols)):
        plt.plot(x, df[cols[i]], label=cols[i])
    plt.legend()
    plt.show()

