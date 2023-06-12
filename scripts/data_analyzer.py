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

    log1 = 'data/Fanuc-GIC-ws-1-3x128-pos/Fanuc_GIC_ws_1_3x128_pos_2023_05_16_12_09_13_0000--s-0/'
    log2 = 'data/Fanuc-GIC-ws-1-3x128/Fanuc_GIC_ws_1_3x128_2023_05_15_10_47_14_0000--s-0/'
    # log = 'data/Expert-pos-bump/success5/'
    file_name1 = log1 + 'progress.csv'
    file_name2 = log2 + 'progress.csv'

    log_now1 = 'data/Fanuc-GIC-ws-1-2x32-pos-GIC/Fanuc_GIC_ws_1_2x32_pos_GIC_2023_05_23_17_21_39_0000--s-0/'
    log_now2 = 'data/Fanuc-GIC-ws-1-2x32-pos-GIC-noForce/Fanuc_GIC_ws_1_2x32_pos_GIC_noForce_2023_05_24_20_35_15_0000--s-0/'
    log_now3 = 'data/Fanuc-GIC-ws-1-3x128-pos-vel-GIC/Fanuc_GIC_ws_1_3x128_pos_vel_GIC_2023_05_26_19_13_06_0000--s-0/'
    log_now4 = 'data/Fanuc-GIC-ws-1-3x128-pos-GIC-pretrained/Fanuc_GIC_ws_1_3x128_pos_GIC_pretrained_2023_05_29_13_39_28_0000--s-0/'
    log_now5 = 'data/Fanuc-GIC-ws-1-7x128-pos-GIC-pretrained/Fanuc_GIC_ws_1_7x128_pos_GIC_pretrained_2023_05_30_10_11_04_0000--s-0/'
    log_now6 = 'data/Fanuc-GIC-ws-1-3x128-pos-force-GIC/Fanuc_GIC_ws_1_3x128_pos_force_GIC_2023_05_31_12_56_22_0000--s-0/'
    log_now7 = 'data/Fanuc-GIC-ws-1-3x128-feature-GIC/Fanuc_GIC_ws_1_3x128_feature_GIC_2023_06_01_13_15_10_0000--s-0/'
    log_now8 = 'data/Fanuc-GIC-ws-1-2x32-feature-GIC/Fanuc_GIC_ws_1_2x32_feature_GIC_2023_06_01_14_36_11_0000--s-0/'
    log_now9 = 'data/Fanuc-GIC-ws-1-2x256-feature-GIC/Fanuc_GIC_ws_1_2x256_feature_GIC_2023_06_02_10_36_39_0000--s-0/'
    log_now10 = 'data/Fanuc-GIC-ws-1-2x256-pos-feature-GIC/Fanuc_GIC_ws_1_2x256_pos_feature_GIC_2023_06_02_19_04_06_0000--s-0/'
    log_now11 = 'data/Fanuc-GIC-ws-1-2x256-pos-feature-GIC-rerun/Fanuc_GIC_ws_1_2x256_pos_feature_GIC_rerun_2023_06_03_14_43_30_0000--s-0/'
    log_now12 = 'data/Fanuc-GIC-ws-1-2x256-pos-feature2-GIC/Fanuc_GIC_ws_1_2x256_pos_feature2_GIC_2023_06_03_23_02_31_0000--s-0/'

    log_now13 = 'data/Fanuc-GIC-ws-1-2x256-pos-feature3-GIC/Fanuc_GIC_ws_1_2x256_pos_feature3_GIC_2023_06_04_09_50_42_0000--s-0/'
    
    log_pos1 = 'data/Fanuc-GIC-ws-1-2x256-pos-seed1-GIC/Fanuc_GIC_ws_1_2x256_pos_seed1_GIC_2023_06_04_17_56_16_0000--s-0/'

    log_pos1_redo = 'data/Fanuc-GIC-ws-1-3x200-pos-seed1-GIC/Fanuc_GIC_ws_1_3x200_pos_seed1_GIC_2023_06_05_11_18_04_0000--s-0/'

    log_pos_minimal = 'data/Fanuc-GIC-ws-1-3x128-pos-miminal-GIC-no-force/Fanuc_GIC_ws_1_3x128_pos_miminal_GIC_no_force_2023_06_05_17_08_03_0000--s-0/'
    log_pos_minimal2 = 'data/Fanuc-GIC-ws-1-3x128-pos-miminal-GIC/Fanuc_GIC_ws_1_3x128_pos_miminal_GIC_2023_06_05_19_07_54_0000--s-0/'

    log_pos_minimal_noforce = 'data/Fanuc-GIC-ws-1-3x128-pos-miminal-GIC-no-force/Fanuc_GIC_ws_1_3x128_pos_miminal_GIC_no_force_2023_06_05_23_43_31_0000--s-0/'
    log_pos_minimal_noforce2 = 'data/Fanuc-GIC-ws-1-3x128-pos-miminal-GIC-no-force/Fanuc_GIC_ws_1_3x128_pos_miminal_GIC_no_force_2023_06_06_10_03_26_0000--s-0/'
    log_pos_minimal2_noforce = 'data/Fanuc-GIC-ws-1-3x128-pos-miminal2-GIC-no-force/Fanuc_GIC_ws_1_3x128_pos_miminal2_GIC_no_force_2023_06_06_21_55_16_0000--s-0/'

    log_pos_minimal_noforce_sepa = 'data/Fanuc-Separated-GIC-ws-1-3x128-pos-miminal-GIC-no-force/Fanuc_Separated_GIC_ws_1_3x128_pos_miminal_GIC_no_force_2023_06_07_19_48_52_0000--s-0/'
    log_pos_minimal2_noforce_sepa = 'data/Fanuc-Separated-GIC-ws-1-3x128-pos-miminal2-GIC-no-force/Fanuc_Separated_GIC_ws_1_3x128_pos_miminal2_GIC_no_force_2023_06_08_12_44_26_0000--s-0/'
    log_pos_minimal2_noforce_sepa_reward2 = 'data/Fanuc-Separated-GIC-ws-1-3x128-pos-miminal2-GIC-no-force-force-penalty/Fanuc_Separated_GIC_ws_1_3x128_pos_miminal2_GIC_no_force_force_penalty_2023_06_08_19_54_16_0000--s-0/'
    log_pos_minimal3_noforce_sepa_reward3 = 'data/Fanuc-Separated-GIC-ws-1-3x128-pos-miminal3-GIC-no-force-force-penalty/Fanuc_Separated_GIC_ws_1_3x128_pos_miminal3_GIC_no_force_force_penalty_2023_06_09_20_39_27_0000--s-0/'

    log_td3_2 = 'data/TD3-GIC-default/TD3_GIC_default_2023_05_30_17_39_41_0000--s-0/'
    log_td3_1 = 'data/TD3-GIC-default/TD3_GIC_default_2023_05_30_10_08_04_0000--s-0/'
    log_td3_residual = 'data/TD3-GIC-residual-default/TD3_GIC_residual_default_2023_06_01_11_20_19_0000--s-0/'

    log_residual = 'data/Fanuc-GIC-ws-1-3x128-pos-GIC-residual/Fanuc_GIC_ws_1_3x128_pos_GIC_residual_2023_05_31_12_55_38_0000--s-0/'

    log_success = 'data/Fanuc_success/GIC_window1_pos_vel_3x128/'
    file_name_success = 'data/Fanuc_success/GIC_window1_pos_vel_3x128/progress.csv'
    log_keep = 'data/Fanuc-GIC-ws-1-2x256-pos-feature-GIC/Fanuc_GIC_ws_1_2x256_pos_feature_GIC_2023_06_02_19_04_06_0000--s-0/'
    log_keep2 = 'data/Fanuc-GIC-ws-1-3x128-pos-miminal-GIC/Fanuc_GIC_ws_1_3x128_pos_miminal_GIC_2023_06_05_19_07_54_0000--s-0/'
    log_success2 = 'data/Fanuc-GIC-ws-1-3x128-pos-miminal-GIC-no-force/Fanuc_GIC_ws_1_3x128_pos_miminal_GIC_no_force_2023_06_06_10_03_26_0000--s-0/'

    log_keep4 = 'data/Fanuc-GIC-ws-1-3x128-pos-miminal2-GIC-no-force/Fanuc_GIC_ws_1_3x128_pos_miminal2_GIC_no_force_2023_06_06_21_55_16_0000--s-0/'

    ### On the separated environment
    log_success3 = 'data/Fanuc-Separated-GIC-ws-1-3x128-pos-miminal-GIC-no-force/Fanuc_Separated_GIC_ws_1_3x128_pos_miminal_GIC_no_force_2023_06_07_19_48_52_0000--s-0/' #itr 230

    ###
    log_bm_pos_minimal_noforce = 'data/Fanuc-Separated-CIC-ws-1-3x128-pos-miminal-no-force/Fanuc_Separated_CIC_ws_1_3x128_pos_miminal_no_force_2023_06_11_11_37_37_0000--s-0/'

    main(log_bm_pos_minimal_noforce + 'progress.csv', window)

    # main_multi(log_pos_minimal_noforce_sepa + 'progress.csv', log_pos_minimal3_noforce_sepa_reward3 + 'progress.csv', window)