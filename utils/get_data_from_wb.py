import wandb
import numpy as np
import matplotlib.pyplot as plt

wandb.login(key='c6350accf3d3ceacf6585d4d9515f3ff37db8712')
MY_WB_NAME = 'yobo'
dataset = 'zinc'

'''
Experiments:
Additional features (without one)
'''
all = ['run-20230712_112605-chfc1oz3', 'run-20230712_112202-nj5p1ab6', 'run-20230712_112200-zhpgjgcx']
no_random = ['run-20230715_100048-dtx6lf4k', 'run-20230715_094949-gvlyufrf', 'run-20230715_094949-457rcjsb']
no_path = ['run-20230715_225208-j1zd9ckw', 'run-20230715_094655-ugwj9xd2', 'run-20230715_094655-45hcuj27']
no_spectre = ['run-20230715_225248-33aov2jh', 'run-20230715_225238-mc3cbsjs', 'run-20230715_225224-hbsp4xyu']
no_cycles = ['run-20230714_134608-svpk2n9w', 'run-20230714_220902-p6j7yfvi', 'run-20230714_220902-ulszvp3z']

only_random = ['run-20230716_100633-mreoftpp', 'run-20230716_100633-oz7tl7o6', 'run-20230716_100634-ponk033j']
only_path = ['run-20230715_225349-8i3fywt8', 'run-20230715_225349-y08pgigx', 'run-20230715_225349-wnr21g68']
only_spectre = ['run-20230716_101724-46lih1og', 'run-20230716_101724-lsk3kasv', 'run-20230716_101725-d74nczlc']
only_cycles = ['run-20230716_100754-zqjckt1b', 'run-20230716_100754-m93hosfg', 'run-20230716_100727-brpu15wa']
no_feat = ['kfn9y954', 'nj8yb7h3', 'bf5uzd6h']

no_init = ['4xjguyqj', 'e5cxxft7', 'q7hswnck', 'ctr38k1g', 'wadw4oe7']
init = ['fq2d32yz', 'fn8fmnno', 'smakvjif', 'nj5p1ab6', 'zhpgjgcx']
'''
Experiments:
Autoencoder 256 cat.
'''

c1_k256 = ['run-20230717_144804-4og3n7sq', 'run-20230717_152726-qd6k3xyl', 'Running run-20230717_152726-bl3xlqvp']
# done run-20230717_144804-4og3n7sq
# Running run-20230717_152726-bl3xlqvp, run-20230717_152726-qd6k3xyl
c4_k4 = ['run-20230717_143533-q0nq1erk', 'run-20230717_143533-eb3l210m', 'run-20230717_143534-j8m9c3bw'] # prior done: tous
c2_k16 = ['run-20230717_141903-3b77svim', 'run-20230717_141903-zpl2y0d7', 'run-20230717_141904-5hhorcvy'] # all running

c1_k1024 = ['qn49jv5n', 'cm21uten', '5kwrr1mi']
c2_k32 = ['chfc1oz3', 'nj5p1ab6', 'zhpgjgcx']
c3_k10 = ['ydss5xaa', 'zcnfatoa', '3foayiva']

c1_k4096 = ['52xnm6cg', 'qg819tuz', 'dbg45s1u']
c2_k64 = ['juilva3i', 'wux34zl3', 'xemhp8ym']
c3_k16 = ['14kium8x', '89pdp8yo', 'zbahxo0c']
c4_k8 = ['zbahxo0c', '47o3ia9h', 'qj33aun2']

c1_k256p = ['8j75283s', 'topaz6b3', '0j7vmv2k']
c2_k16p = ['pec22t5f', 'ef527bzi', 'kti67ejw']
c4_k4p = ['6ayi9f24', 'rvhyscus', 'focrvj0e']

c1_k1024p =['3cr38y0n', 'jmr9itls', 'j8awvsik']
c2_k32p = ['e7egstpw', 'uulihk6g', 'yogc8g6e']
c3_k10p = ['jveei13n', 'appjltq9', '1c9wjzmd']

c1_k4096p = ['gwnl0yc0', '1up9acml', 'sko0z5e7']
c2_k64p = ['vuuebyr6', 'hwiwnsxj', 'xb1jpgyw']
c3_k16p = ['6uvttsk3', 'u00jgri6', 'boemcnq0']
c4_k8p = ['vscsq7h8', 'bwh7dle4', '1xp8oa7m']



def get_timeseries(run_id_list, key, prior = False):
    timeseries = []
    for folder_name in run_id_list:
        api = wandb.Api()
        run_id = folder_name[-8:]
        if prior:
            train = 'prior'
        else:
            train = 'autoencoder'
        run_name = f'{MY_WB_NAME}/VQ-GAE_{dataset}_train_{train}/{run_id}'
        run = api.run(run_name)
        data = run.history(keys=[key])[:70]
        timeseries.append(data[key].to_numpy())
    return timeseries


def plot_errplot_list(run_list, run_name, key, prior=False, up=False):
    """
    Function to plot averages of several time series with standard deviation.

    Args:
    timeseries: list of numpy arrays, where each array represents a timeseries
    colors: colormap for the plots
    """
    # Calculate the average and standard deviation of all timeseries
    length = 70
    n = len(run_list)
    # Create figure and axes
    fig, ax = plt.subplots()
    means = []
    stds = []
    # Getting the colormap
    cm = plt.get_cmap('viridis')
    for i, run_id_list in enumerate(run_list):
        timeseries = get_timeseries(run_id_list, key, prior=prior)
        value = []
        for serie in timeseries:
            if up:
                val = np.max(serie[:length])
            else:
                val = np.min(serie[:length])
            value.append(val)
        means.append(np.asarray(value).mean())
        stds.append(np.asarray(value).std())
    print(stds)
    print(means)
    plt.bar(x=run_name,
            color=cm([0, 0, 0, 0.35, 0.5, 0.4, 1, 1, 1, 1]),
            height=means, yerr=stds, capsize=10)
    labels = ['$M^C=256$', '$M^C=1024$', '$M^C=1000$', '$M^C=4096$']
    colors = [0, 0.5, 0.4, 1.]
    handles = [plt.Rectangle((0, 0), 1, 1, color=cm(color)) for color in colors]
    plt.legend(handles, labels)
    plt.title('Effect of the codebook sizes \nand partitioning on reconstruction', fontsize=18)
    #'Fréchet Chemical Distance'
    plt.ylabel('Reconstruction loss', fontsize=16)
    plt.xlabel('Codebook size $m$ and number of vectors $C$: $m^C$', fontsize=16)
    plt.show()


def plot_timeseries_list(run_list, timesteps, names, key, title=None, y_max=None):
    """
    Function to plot averages of several time series with standard deviation.

    Args:
    timeseries: list of numpy arrays, where each array represents a timeseries
    colors: colormap for the plots
    """
    # Getting the colormap
    cm = plt.get_cmap('magma')

    # Calculate the average and standard deviation of all timeseries
    length = len(timesteps)
    n = len(run_list)
    # Create figure and axes
    fig, ax = plt.subplots()
    for i, run_id_list in enumerate(run_list):
        timeseries = get_timeseries(run_id_list, key)
        timeseries, min_size = concatenate_arrays(timeseries)
        avg_timeseries = np.mean(timeseries, axis=0)
        std_timeseries = np.std(timeseries, axis=0)

        # Plot the average timeseries
        ax.plot(timesteps, avg_timeseries[:length], color=cm(0.8 * i/(n-1) + 0.1), linestyle='-', label=f'{names[i]}')

        # Plot the standard deviation
        ax.fill_between(timesteps, avg_timeseries[:length] - std_timeseries[:length],
                        avg_timeseries[:length] + std_timeseries[:length],
                        color=cm(0.8 * i/(n-1) + 0.1), alpha=0.2)

    # Setting the legend
    ax.legend(fontsize=10, loc=1)

    # Show the plot
    plt.ylim(0, y_max)
    plt.ylabel('Reconstruction loss', fontsize=14)
    plt.xlabel('Epochs', fontsize=14)
    plt.title(title, fontsize=20)
    plt.show()


def plot_average_timeseries(timeseries, timesteps):
    """
    Function to plot averages of several time series with standard deviation.

    Args:
    timeseries: list of numpy arrays, where each array represents a timeseries
    colors: colormap for the plots
    """
    # Getting the colormap
    cm = plt.get_cmap('plasma')

    # Calculate the average and standard deviation of all timeseries
    avg_timeseries = np.mean(timeseries, axis=0)
    std_timeseries = np.std(timeseries, axis=0)

    # Create figure and axes
    fig, ax = plt.subplots()

    # Plot the average timeseries
    ax.plot(timesteps//1000, avg_timeseries, color=cm(0.2), linestyle='-', label='Average')

    # Plot the standard deviation
    ax.fill_between(timesteps/1000, avg_timeseries - std_timeseries, avg_timeseries + std_timeseries,
                    color=cm(0.2), alpha=0.3)

    # Setting the legend
    ax.legend()

    # Show the plot
    plt.ylim(0, 0.005)
    plt.show()


def concatenate_arrays(arrays):
    """
    Function to concatenate several numpy arrays along a new axis (axis=0).
    If the arrays are not of the same size, truncate the arrays to the size of the smallest one.

    Args:
    arrays: list of numpy arrays to concatenate

    Returns:
    concatenated_array: concatenated numpy array
    """
    # Find the size of the smallest array
    min_size = min(arr.shape[0] for arr in arrays)

    # Truncate all arrays to the size of the smallest one
    truncated_arrays = [arr[:min_size] for arr in arrays]

    # Concatenate the truncated arrays along a new axis
    concatenated_array = np.stack(truncated_arrays, axis=0)

    return concatenated_array, min_size


# plot_timeseries_list([all, no_path, no_spectre, no_random, no_cycles], np.arange(20)+1,
#                      ['all', 'no p-paths', 'no spectral embedding', 'no random features', 'no c-cycles'],
#                      'val.recon_loss', title='Effect of the feature augmentation', y_max=0.005)
#
# plot_timeseries_list([no_feat, only_path, only_spectre, only_random, only_cycles], np.arange(20)+1,
#                      ['no feature augmentation', 'p-paths', 'spectral embedding', 'random features', 'c-cycles'],
#                      'val.recon_loss', title='Effect of the feature augmentation', y_max=0.01)

#plot_timeseries_list([init, no_init], np.arange(20)+1,
#                     ['K-means++', 'No init.'], 'val.recon_loss')

label = ['$256^1$', '$16^2$', '$4^4$', '$1024^1$', '$32^2$','$10^3$', '$4096^1$', '$64^2$', '$16^3$', '$8^4$']
#plot_errplot_list([c1_k256, c2_k16, c4_k4, c1_k1024, c2_k32, c3_k10, c1_k4096, c2_k64, c3_k16, c4_k8], label,
#                  'val.recon_loss')

plot_errplot_list([c1_k256p, c2_k16p, c4_k4p, c1_k1024p, c2_k32p, c3_k10p, c1_k4096p, c2_k64p, c3_k16p, c4_k8p], label,
                 'Mol eval iter.nspdk', prior=True, up=True)