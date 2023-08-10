import pandas as pd
import matplotlib.pyplot as plt

def main(file_name, window):
    print(file_name)
    df = pd.read_csv(file_name)
    
    df.head()
    # t1 = 'eval/Rewards Mean'
    t1 = 'eval/Returns Mean'
    t2 = 'expl/Returns Mean'
    col = df[t1]
    col2 = df[t2]

    filtered = col.rolling(window=window).mean()
    filtered2 = col2.rolling(window=window).mean() # 30 works well

    print(col.shape)

    qf1 = 'trainer/QF1 Loss'
    qf2 = 'trainer/QF2 Loss'

    success = 'eval/env_infos/success Mean'

    mu_loss = 'trainer/Policy Loss'

    col_qf1 = df[qf1]
    col_qf2 = df[qf2]

    col_mu_loss = df[mu_loss]

    col_success = df[success]
    filtered_success = col_success.rolling(window=window).mean()

    plt.figure()
    plt.plot(col)
    plt.plot(filtered)
    plt.legend(['Return','filtered Returns'])
    plt.ylim([-100,120])

    plt.figure()
    plt.plot(col2)
    plt.plot(filtered2)
    plt.legend(['Expl Return','Expl filtered Returns'])
    plt.ylim([-100,120])

    plt.figure()
    plt.plot(col_qf1)
    plt.plot(col_qf2)
    plt.legend(['Loss Q1','Loss Q2'])

    plt.figure()
    plt.plot(col_mu_loss)
    plt.legend(['Loss policy'])

    plt.figure()
    plt.plot(col_success)
    plt.plot(filtered_success)
    plt.legend(['eval success','filtered'])
    
    plt.show()

    return 0

def main_multi(f1, f2, window):
    df1 = pd.read_csv(f1)
    df2 = pd.read_csv(f2)

    t1 = 'eval/Returns Mean'
    mu_loss = 'trainer/Policy Loss'

    col_mu_loss_1 = df1[mu_loss]
    col_mu_loss_2 = df2[mu_loss]

    col_return_1 = df1[t1]
    col_return_2 = df2[t1]

    filtered_1 = col_return_1.rolling(window=window).mean()
    filtered_2 = col_return_2.rolling(window=window).mean()

    success = 'eval/env_infos/success Mean'

    col_success1 = df1[success]
    filtered_success1 = col_success1.rolling(window=window).mean()
    col_success2 = df2[success]
    filtered_success2 = col_success2.rolling(window=window).mean()

    plt.figure()
    # plt.plot(col_return_1,'b')
    # plt.plot(col_return_2,'r')
    plt.plot(filtered_1)
    plt.plot(filtered_2)
    # plt.legend(['pos', 'pos_vel','filtered 1', 'filtered 2'])
    plt.legend(['case 1', 'case 2'])

    plt.figure()
    plt.plot(col_mu_loss_1)
    plt.plot(col_mu_loss_2)
    plt.legend(['case 1', 'case 2'])

    plt.figure()
    plt.plot(filtered_success1)
    plt.plot(filtered_success2)
    plt.legend(['filtered 1', 'filtered 2'])

    plt.show()



if __name__ == "__main__":
    window = 10

    log_GIC_final = 'data/Fanuc_success/final_GIC_minimal_separated_pos_3x128_reward2/'
    log_CIC_final = 'data/Fanuc_success/final_CIC_minimal_separated_pos_3x128_reward2/'
    
    main(log_GIC_final + 'progress.csv', window)

    main_multi(log_GIC_final + 'progress.csv', log_CIC_final + 'progress.csv', window)