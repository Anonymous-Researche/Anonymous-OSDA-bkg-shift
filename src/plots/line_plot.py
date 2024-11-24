import matplotlib.pyplot as plt
import pandas as pd

# Creating a dataframe from the given data
data = {
    'FPR threshold': [0.01, 0.02, 0.03, 0.04, 0.05],
    'AUROC_OOD Class=0 - water/ice/snow': [0.9809, 0.9783, 0.9799, 0.9808, 0.9765],
    'AUROC_OOD Class=1 - mountains/hills/desert/sky': [0.9903, 0.9959, 0.9959, 0.9578, 0.9895],
    'AUROC_OOD Class=2 - forest/field/jungle': [0.9853, 0.9823, 0.9793, 0.9781, 0.9765],
    'AUPRC_OOD Class=0 - water/ice/snow': [0.9322, 0.9207, 0.923, 0.9225, 0.8988],
    'AUPRC_OOD Class=1 - mountains/hills/desert/sky': [0.9628, 0.9763, 0.9795, 0.9898, 0.9569],
    'AUPRC_OOD Class=2 - forest/field/jungle': [0.9496, 0.9449, 0.9278, 0.9241, 0.9169],
    'OSCR_OOD Class=0 - water/ice/snow': [0.8182, 0.8126, 0.8206, 0.8184, 0.8176],
    'OSCR_OOD Class=1 - mountains/hills/desert/sky': [0.8327, 0.8686, 0.8656, 0.8563, 0.8414],
    'OSCR_OOD Class=2 - forest/field/jungle': [0.8315, 0.8167, 0.8238, 0.8254, 0.821]
}

df = pd.DataFrame(data)

# Plotting the data
def plot_metric(df, metric):
    plt.figure(figsize=(10, 6))
    plt.plot(df['FPR threshold'], df[f'{metric}_OOD Class=0 - water/ice/snow'], label='OOD Class=0 - water/ice/snow')
    plt.plot(df['FPR threshold'], df[f'{metric}_OOD Class=1 - mountains/hills/desert/sky'], label='OOD Class=1 - mountains/hills/desert/sky')
    plt.plot(df['FPR threshold'], df[f'{metric}_OOD Class=2 - forest/field/jungle'], label='OOD Class=2 - forest/field/jungle')
    plt.xlabel('FPR threshold')
    plt.ylabel(metric)
    plt.title(f'{metric} vs FPR threshold')
    plt.legend()
    plt.grid(True)
    plt.xticks(df['FPR threshold'])
    plt.show()
    plt.savefig(f'../../plots/{metric}_vs_FPR_threshold.png')

# Plotting AUROC
plot_metric(df, 'AUROC')

# Plotting AUPRC
plot_metric(df, 'AUPRC')

# Plotting OSCR
plot_metric(df, 'OSCR')
