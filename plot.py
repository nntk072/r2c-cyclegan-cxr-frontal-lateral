# Using npy files to plot the results
# # Save the loss data for each iteration into a separate file
# np.save(py.join(plot_dir, f'loss_A2B_g_loss_{ep}.npy'), A2B_g_loss)
# np.save(py.join(plot_dir, f'loss_B2A_g_loss_{ep}.npy'), B2A_g_loss)
# np.save(py.join(plot_dir, f'loss_A2B2A_cycle_loss_{ep}.npy'), A2B2A_cycle_loss)
# np.save(py.join(plot_dir, f'loss_B2A2B_cycle_loss_{ep}.npy'), B2A2B_cycle_loss)
# np.save(py.join(plot_dir, f'loss_A2A_id_loss_{ep}.npy'), A2A_id_loss)
# np.save(py.join(plot_dir, f'loss_B2B_id_loss_{ep}.npy'), B2B_id_loss)
# np.save(py.join(plot_dir, f'loss_A_d_loss_{ep}.npy'), A_d_loss)
# np.save(py.join(plot_dir, f'loss_B_d_loss_{ep}.npy'), B_d_loss)
# np.save(py.join(plot_dir, f'iterations_{ep}.npy'), iterations)

import numpy as np
import matplotlib.pyplot as plt
import pylib as py


def main():
    method = "unet"
    A2B_g_loss_list = np.array([])
    B2A_g_loss_list = np.array([])
    A2B2A_cycle_loss_list = np.array([])
    B2A2B_cycle_loss_list = np.array([])
    A2A_id_loss_list = np.array([])
    B2B_id_loss_list = np.array([])
    A_d_loss_list = np.array([])
    B_d_loss_list = np.array([])
    ssim_A2B_list = np.array([])
    psnr_A2B_list = np.array([])
    ssim_B2A_list = np.array([])
    psnr_B2A_list = np.array([])
    ep_list = np.array([])
    for ep in range(0, 1000):
        A2B_g_loss = np.load(
            f'output/{method}/plot_data/training/loss_A2B_g_loss_{ep}.npy')
        B2A_g_loss = np.load(
            f'output/{method}/plot_data/training/loss_B2A_g_loss_{ep}.npy')
        A2B2A_cycle_loss = np.load(
            f'output/{method}/plot_data/training/loss_A2B2A_cycle_loss_{ep}.npy')
        B2A2B_cycle_loss = np.load(
            f'output/{method}/plot_data/training/loss_B2A2B_cycle_loss_{ep}.npy')
        A2A_id_loss = np.load(
            f'output/{method}/plot_data/training/loss_A2A_id_loss_{ep}.npy')
        B2B_id_loss = np.load(
            f'output/{method}/plot_data/training/loss_B2B_id_loss_{ep}.npy')
        A_d_loss = np.load(
            f'output/{method}/plot_data/training/loss_A_d_loss_{ep}.npy')
        B_d_loss = np.load(
            f'output/{method}/plot_data/training/loss_B_d_loss_{ep}.npy')
        ssim_A2B = np.load(
            f'output/{method}/plot_data/training/ssim_A2B_value_list_{ep}.npy')
        psnr_A2B = np.load(
            f'output/{method}/plot_data/training/psnr_A2B_value_list_{ep}.npy')
        ssim_B2A = np.load(
            f'output/{method}/plot_data/training/ssim_B2A_value_list_{ep}.npy')
        psnr_B2A = np.load(
            f'output/{method}/plot_data/training/psnr_B2A_value_list_{ep}.npy')
        iterations = np.load(
            f'output/{method}/plot_data/training/iterations_{ep}.npy')

        # Calculate the mean of the loss data for each iteration and save into the list
        A2B_g_loss_list = np.append(A2B_g_loss_list, np.mean(A2B_g_loss))
        B2A_g_loss_list = np.append(B2A_g_loss_list, np.mean(B2A_g_loss))
        A2B2A_cycle_loss_list = np.append(
            A2B2A_cycle_loss_list, np.mean(A2B2A_cycle_loss))
        B2A2B_cycle_loss_list = np.append(
            B2A2B_cycle_loss_list, np.mean(B2A2B_cycle_loss))
        A2A_id_loss_list = np.append(A2A_id_loss_list, np.mean(A2A_id_loss))
        B2B_id_loss_list = np.append(B2B_id_loss_list, np.mean(B2B_id_loss))
        A_d_loss_list = np.append(A_d_loss_list, np.mean(A_d_loss))
        B_d_loss_list = np.append(B_d_loss_list, np.mean(B_d_loss))
        ssim_A2B_list = np.append(ssim_A2B_list, np.mean(ssim_A2B))
        psnr_A2B_list = np.append(psnr_A2B_list, np.mean(psnr_A2B))
        ssim_B2A_list = np.append(ssim_B2A_list, np.mean(ssim_B2A))
        psnr_B2A_list = np.append(psnr_B2A_list, np.mean(psnr_B2A))
        ep_list = np.append(ep_list, ep)

    # Do the same with the valid
    A2B_g_loss_list_valid = np.array([])
    B2A_g_loss_list_valid = np.array([])
    A2B2A_cycle_loss_list_valid = np.array([])
    B2A2B_cycle_loss_list_valid = np.array([])
    A2A_id_loss_list_valid = np.array([])
    B2B_id_loss_list_valid = np.array([])
    A_d_loss_list_valid = np.array([])
    B_d_loss_list_valid = np.array([])
    ssim_A2B_list_valid = np.array([])
    ssim_B2A_list_valid = np.array([])
    psnr_A2B_list_valid = np.array([])
    psnr_B2A_list_valid = np.array([])
    ep_list_valid = np.array([])
    for ep in range(0, 1000):  # the name of the folder is validation
        A2B_g_loss = np.load(
            f'output/{method}/plot_data/validation/loss_A2B_g_loss_{ep}.npy')
        B2A_g_loss = np.load(
            f'output/{method}/plot_data/validation/loss_B2A_g_loss_{ep}.npy')
        A2B2A_cycle_loss = np.load(
            f'output/{method}/plot_data/validation/loss_A2B2A_cycle_loss_{ep}.npy')
        B2A2B_cycle_loss = np.load(
            f'output/{method}/plot_data/validation/loss_B2A2B_cycle_loss_{ep}.npy')
        A2A_id_loss = np.load(
            f'output/{method}/plot_data/validation/loss_A2A_id_loss_{ep}.npy')
        B2B_id_loss = np.load(
            f'output/{method}/plot_data/validation/loss_B2B_id_loss_{ep}.npy')
        A_d_loss = np.load(
            f'output/{method}/plot_data/validation/loss_A_d_loss_{ep}.npy')
        B_d_loss = np.load(
            f'output/{method}/plot_data/validation/loss_B_d_loss_{ep}.npy')
        ssim_A2B = np.load(
            f'output/{method}/plot_data/validation/ssim_A2B_value_list_{ep}.npy')
        psnr_A2B = np.load(
            f'output/{method}/plot_data/validation/psnr_A2B_value_list_{ep}.npy')
        ssim_B2A = np.load(
            f'output/{method}/plot_data/validation/ssim_B2A_value_list_{ep}.npy')
        psnr_B2A = np.load(
            f'output/{method}/plot_data/validation/psnr_B2A_value_list_{ep}.npy')
        iterations = np.load(
            f'output/{method}/plot_data/validation/iterations_{ep}.npy')

        # Calculate the mean of the loss data for each iteration and save into the list
        A2B_g_loss_list_valid = np.append(
            A2B_g_loss_list_valid, np.mean(A2B_g_loss))
        B2A_g_loss_list_valid = np.append(
            B2A_g_loss_list_valid, np.mean(B2A_g_loss))
        A2B2A_cycle_loss_list_valid = np.append(
            A2B2A_cycle_loss_list_valid, np.mean(A2B2A_cycle_loss))
        B2A2B_cycle_loss_list_valid = np.append(
            B2A2B_cycle_loss_list_valid, np.mean(B2A2B_cycle_loss))
        A2A_id_loss_list_valid = np.append(
            A2A_id_loss_list_valid, np.mean(A2A_id_loss))
        B2B_id_loss_list_valid = np.append(
            B2B_id_loss_list_valid, np.mean(B2B_id_loss))
        A_d_loss_list_valid = np.append(A_d_loss_list_valid, np.mean(A_d_loss))
        B_d_loss_list_valid = np.append(B_d_loss_list_valid, np.mean(B_d_loss))
        ssim_A2B_list_valid = np.append(ssim_A2B_list_valid, np.mean(ssim_A2B))
        psnr_A2B_list_valid = np.append(psnr_A2B_list_valid, np.mean(psnr_A2B))
        ssim_B2A_list_valid = np.append(ssim_B2A_list_valid, np.mean(ssim_B2A))
        psnr_B2A_list_valid = np.append(psnr_B2A_list_valid, np.mean(psnr_B2A))
        ep_list_valid = np.append(ep_list_valid, ep)

    # Plot into 4 figures, g_loss, cycle_loss, id_loss, d_loss
    plt.figure(figsize=(20, 12))  # Increase figure size
    plt.plot(ep_list, A2B_g_loss_list, label='A2B_g_loss', linewidth=1)
    plt.plot(ep_list, B2A_g_loss_list, label='B2A_g_loss', linewidth=1)
    plt.plot(ep_list_valid, A2B_g_loss_list_valid,
             label='A2B_g_loss_valid', linewidth=1)
    plt.plot(ep_list_valid, B2A_g_loss_list_valid,
             label='B2A_g_loss_valid', linewidth=1)
    plt.legend(fontsize='large')
    plt.title('Generator Losses', fontsize='x-large')
    plt.xlabel('Epochs', fontsize='large')
    plt.ylabel('Loss', fontsize='large')
    plt.grid(which='major', color='black', linewidth=0.5)
    plt.minorticks_on()
    plt.grid(which='minor', color='gray', linewidth=0.5)
    plt.xlim(ep_list[0], ep_list[-1])
    plt.xticks(np.arange(ep_list[0], ep_list[-1], 50))
    if max(max(A2B_g_loss_list), max(B2A_g_loss_list), max(A2B_g_loss_list_valid), max(B2A_g_loss_list_valid)) < 0.25:
        plt.ylim(0, 0.25)
        plt.yticks(np.arange(0, 0.25, 0.025))
    elif max(max(A2B_g_loss_list), max(B2A_g_loss_list), max(A2B_g_loss_list_valid), max(B2A_g_loss_list_valid)) < 0.5:
        plt.ylim(0, 0.5)
        plt.yticks(np.arange(0, 0.5, 0.05))
    elif max(max(A2B_g_loss_list), max(B2A_g_loss_list), max(A2B_g_loss_list_valid), max(B2A_g_loss_list_valid)) < 0.75:
        plt.ylim(0, 0.75)
        plt.yticks(np.arange(0, 0.75, 0.05))
    elif max(max(A2B_g_loss_list), max(B2A_g_loss_list), max(A2B_g_loss_list_valid), max(B2A_g_loss_list_valid)) < 1:
        plt.ylim(0, 1)
        plt.yticks(np.arange(0, 1, 0.05))
    else:
        plt.ylim(0, max(max(A2B_g_loss_list), max(B2A_g_loss_list),
                 max(A2B_g_loss_list_valid), max(B2A_g_loss_list_valid)))
        plt.yticks(np.arange(0, max(max(A2B_g_loss_list), max(B2A_g_loss_list), max(
            A2B_g_loss_list_valid), max(B2A_g_loss_list_valid)), 0.05))
    plt.savefig(f'output/{method}/generator_losses.png',
                dpi=300)  # Save as high-res image
    plt.close()

    # Cycle Losses Plot
    plt.figure(figsize=(20, 12))  # Increase figure size
    plt.plot(ep_list, A2B2A_cycle_loss_list,
             label='A2B2A_cycle_loss', linewidth=1)
    plt.plot(ep_list, B2A2B_cycle_loss_list,
             label='B2A2B_cycle_loss', linewidth=1)
    plt.plot(ep_list_valid, A2B2A_cycle_loss_list_valid,
             label='A2B2A_cycle_loss_valid', linewidth=1)
    plt.plot(ep_list_valid, B2A2B_cycle_loss_list_valid,
             label='B2A2B_cycle_loss_valid', linewidth=1)
    plt.legend(fontsize='large')
    plt.title('Cycle Losses', fontsize='x-large')
    plt.xlabel('Epochs', fontsize='large')
    plt.ylabel('Loss', fontsize='large')
    plt.grid(which='major', color='black', linewidth=0.5)
    plt.minorticks_on()
    plt.grid(which='minor', color='gray', linewidth=0.5)
    plt.xlim(ep_list[0], ep_list[-1])
    plt.xticks(np.arange(ep_list[0], ep_list[-1], 50))
    if max(max(A2B2A_cycle_loss_list), max(B2A2B_cycle_loss_list), max(A2B2A_cycle_loss_list_valid), max(B2A2B_cycle_loss_list_valid)) < 0.25:
        plt.ylim(0, 0.25)
        plt.yticks(np.arange(0, 0.25, 0.025))
    elif max(max(A2B2A_cycle_loss_list), max(B2A2B_cycle_loss_list), max(A2B2A_cycle_loss_list_valid), max(B2A2B_cycle_loss_list_valid)) < 0.5:
        plt.ylim(0, 0.5)
        plt.yticks(np.arange(0, 0.5, 0.05))
    elif max(max(A2B2A_cycle_loss_list), max(B2A2B_cycle_loss_list), max(A2B2A_cycle_loss_list_valid), max(B2A2B_cycle_loss_list_valid)) < 0.75:
        plt.ylim(0, 0.75)
        plt.yticks(np.arange(0, 0.75, 0.05))
    elif max(max(A2B2A_cycle_loss_list), max(B2A2B_cycle_loss_list), max(A2B2A_cycle_loss_list_valid), max(B2A2B_cycle_loss_list_valid)) < 1:
        plt.ylim(0, 1)
        plt.yticks(np.arange(0, 1, 0.05))
    else:
        plt.ylim(0, max(max(A2B2A_cycle_loss_list), max(B2A2B_cycle_loss_list), max(
            A2B2A_cycle_loss_list_valid), max(B2A2B_cycle_loss_list_valid)))
        plt.yticks(np.arange(0, max(max(A2B2A_cycle_loss_list), max(B2A2B_cycle_loss_list), max(
            A2B2A_cycle_loss_list_valid), max(B2A2B_cycle_loss_list_valid)), 0.05))
    plt.savefig(f'output/{method}/cycle_losses.png',
                dpi=300)  # Save as high-res image
    plt.close()

    # Identity Losses Plot
    plt.figure(figsize=(20, 12))  # Increase figure size
    plt.plot(ep_list, A2A_id_loss_list, label='A2A_id_loss', linewidth=1)
    plt.plot(ep_list, B2B_id_loss_list, label='B2B_id_loss', linewidth=1)
    plt.plot(ep_list_valid, A2A_id_loss_list_valid,
             label='A2A_id_loss_valid', linewidth=1)
    plt.plot(ep_list_valid, B2B_id_loss_list_valid,
             label='B2B_id_loss_valid', linewidth=1)
    plt.legend(fontsize='large')
    plt.title('Identity Losses', fontsize='x-large')
    plt.xlabel('Epochs', fontsize='large')
    plt.ylabel('Loss', fontsize='large')
    plt.grid(which='major', color='black', linewidth=0.5)
    plt.minorticks_on()
    plt.grid(which='minor', color='gray', linewidth=0.5)
    plt.xlim(ep_list[0], ep_list[-1])
    plt.xticks(np.arange(ep_list[0], ep_list[-1], 50))
    if max(max(A2A_id_loss_list), max(B2B_id_loss_list), max(A2A_id_loss_list_valid), max(B2B_id_loss_list_valid)) < 0.25:
        plt.ylim(0, 0.25)
        plt.yticks(np.arange(0, 0.25, 0.025))
    elif max(max(A2A_id_loss_list), max(B2B_id_loss_list), max(A2A_id_loss_list_valid), max(B2B_id_loss_list_valid)) < 0.5:
        plt.ylim(0, 0.5)
        plt.yticks(np.arange(0, 0.5, 0.05))
    elif max(max(A2A_id_loss_list), max(B2B_id_loss_list), max(A2A_id_loss_list_valid), max(B2B_id_loss_list_valid)) < 0.75:
        plt.ylim(0, 0.75)
        plt.yticks(np.arange(0, 0.75, 0.05))
    elif max(max(A2A_id_loss_list), max(B2B_id_loss_list), max(A2A_id_loss_list_valid), max(B2B_id_loss_list_valid)) < 1:
        plt.ylim(0, 1)
        plt.yticks(np.arange(0, 1, 0.05))
    else:
        plt.ylim(0, max(max(A2A_id_loss_list), max(B2B_id_loss_list),
                 max(A2A_id_loss_list_valid), max(B2B_id_loss_list_valid)))
        plt.yticks(np.arange(0, max(max(A2A_id_loss_list), max(B2B_id_loss_list), max(
            A2A_id_loss_list_valid), max(B2B_id_loss_list_valid)), 0.05))
    plt.savefig(f'output/{method}/identity_losses.png',
                dpi=300)  # Save as high-res image
    plt.close()

    # Discriminator Losses Plot
    plt.figure(figsize=(20, 12))  # Increase figure size
    plt.plot(ep_list, A_d_loss_list, label='A_d_loss', linewidth=1)
    plt.plot(ep_list, B_d_loss_list, label='B_d_loss', linewidth=1)
    plt.plot(ep_list_valid, A_d_loss_list_valid,
             label='A_d_loss_valid', linewidth=1)
    plt.plot(ep_list_valid, B_d_loss_list_valid,
             label='B_d_loss_valid', linewidth=1)
    plt.legend(fontsize='large')
    plt.title('Discriminator Losses', fontsize='x-large')
    plt.xlabel('Epochs', fontsize='large')
    plt.ylabel('Loss', fontsize='large')
    plt.grid(which='major', color='black', linewidth=0.5)
    plt.minorticks_on()
    plt.grid(which='minor', color='gray', linewidth=0.5)
    plt.xlim(ep_list[0], ep_list[-1])
    plt.xticks(np.arange(ep_list[0], ep_list[-1], 50))
    if max(max(A_d_loss_list), max(B_d_loss_list), max(A_d_loss_list_valid), max(B_d_loss_list_valid)) < 0.25:
        plt.ylim(0, 0.25)
        plt.yticks(np.arange(0, 0.25, 0.025))
    elif max(max(A_d_loss_list), max(B_d_loss_list), max(A_d_loss_list_valid), max(B_d_loss_list_valid)) < 0.5:
        plt.ylim(0, 0.5)
        plt.yticks(np.arange(0, 0.5, 0.05))
    elif max(max(A_d_loss_list), max(B_d_loss_list), max(A_d_loss_list_valid), max(B_d_loss_list_valid)) < 0.75:
        plt.ylim(0, 0.75)
        plt.yticks(np.arange(0, 0.75, 0.05))
    elif max(max(A_d_loss_list), max(B_d_loss_list), max(A_d_loss_list_valid), max(B_d_loss_list_valid)) < 1:
        plt.ylim(0, 1)
        plt.yticks(np.arange(0, 1, 0.05))
    else:
        plt.ylim(0, max(max(A_d_loss_list), max(B_d_loss_list),
                 max(A_d_loss_list_valid), max(B_d_loss_list_valid)))
        plt.yticks(np.arange(0, max(max(A_d_loss_list), max(B_d_loss_list), max(
            A_d_loss_list_valid), max(B_d_loss_list_valid)), 0.05))
    # Save as high-res image
    plt.savefig(f'output/{method}/discriminator_losses.png', dpi=300)
    plt.close()

    # Plot the SSIM and PSNR data
    plt.figure(figsize=(20, 12))  # Increase figure size
    plt.plot(ep_list, psnr_A2B_list, label='psnr_A2B', linewidth=1)
    plt.plot(ep_list, psnr_B2A_list, label='psnr_B2A',
             linewidth=1)
    plt.plot(ep_list_valid, psnr_A2B_list_valid,
                label='psnr_A2B_valid', linewidth=1)
    plt.plot(ep_list_valid, psnr_B2A_list_valid,
                label='psnr_B2A_valid', linewidth=1)
    plt.legend(fontsize='large')
    plt.title('PSNR', fontsize='x-large')
    plt.xlabel('Epochs', fontsize='large')
    plt.ylabel('PSNR', fontsize='large')
    plt.grid(which='major', color='black', linewidth=0.5)
    plt.minorticks_on()
    plt.grid(which='minor', color='gray', linewidth=0.5)
    plt.xlim(ep_list[0], ep_list[-1])
    plt.xticks(np.arange(ep_list[0], ep_list[-1], 50))
    if max(max(psnr_A2B_list), max(psnr_B2A_list), max(psnr_A2B_list_valid), max(psnr_B2A_list_valid)) < 15:
        plt.ylim(0, 15)
        plt.yticks(np.arange(0, 15, 1.5))
    elif max(max(psnr_A2B_list), max(psnr_B2A_list), max(psnr_A2B_list_valid), max(psnr_B2A_list_valid)) < 20:
        plt.ylim(0, 20)
        plt.yticks(np.arange(0, 20, 2))
    else:
        plt.ylim(0, max(max(psnr_A2B_list), max(psnr_B2A_list), max(
            psnr_A2B_list_valid), max(psnr_B2A_list_valid)))
        plt.yticks(np.arange(0, max(max(psnr_A2B_list), max(psnr_B2A_list), max(
            psnr_A2B_list_valid), max(psnr_B2A_list_valid)), 2.5))
    plt.savefig(f'output/{method}/psnr.png',
                dpi=300)  # Save as high-res image
    plt.close()

    plt.figure(figsize=(20, 12))  # Increase figure size
    plt.plot(ep_list, ssim_A2B_list, label='ssim_A2B', linewidth=1)
    plt.plot(ep_list, ssim_B2A_list, label='ssim_B2A',
                linewidth=1)
    plt.plot(ep_list_valid, ssim_A2B_list_valid,
                label='ssim_A2B_valid', linewidth=1)
    plt.plot(ep_list_valid, ssim_B2A_list_valid,
                label='ssim_B2A_valid', linewidth=1)
    plt.legend(fontsize='large')
    plt.title('SSIM', fontsize='x-large')
    plt.xlabel('Epochs', fontsize='large')
    plt.ylabel('SSIM', fontsize='large')
    plt.grid(which='major', color='black', linewidth=0.5)
    plt.minorticks_on()
    plt.grid(which='minor', color='gray', linewidth=0.5)
    plt.xlim(ep_list[0], ep_list[-1])
    plt.xticks(np.arange(ep_list[0], ep_list[-1], 50))
    # plt.ylim(-1, 1)
    # plt.yticks(np.arange(-1, 1, 0.1))
    plt.savefig(f'output/{method}/ssim.png',
                dpi=300)  # Save as high-res image
        
    # # Asking for the starting and ending epoch and concatenating the data for the plot
    # # Start epoch
    # start_epoch = int(input('Enter the starting epoch: '))
    # # End epoch
    # end_epoch = int(input('Enter the ending epoch: '))
    # if start_epoch < 0 or end_epoch < 0:
    #     raise ValueError('The starting and ending epoch must be greater than 0')
    # if start_epoch > end_epoch:
    #     raise ValueError('The starting epoch must be less than the ending epoch')

    # # Concatenate the data for the plot
    # A2B_g_loss = np.concatenate([np.load(f'output/plot_data/loss_A2B_g_loss_{ep}.npy') for ep in range(start_epoch, end_epoch)])
    # B2A_g_loss = np.concatenate([np.load(f'output/plot_data/loss_B2A_g_loss_{ep}.npy') for ep in range(start_epoch, end_epoch)])
    # A2B2A_cycle_loss = np.concatenate([np.load(f'output/plot_data/loss_A2B2A_cycle_loss_{ep}.npy') for ep in range(start_epoch, end_epoch)])
    # B2A2B_cycle_loss = np.concatenate([np.load(f'output/plot_data/loss_B2A2B_cycle_loss_{ep}.npy') for ep in range(start_epoch, end_epoch)])
    # A2A_id_loss = np.concatenate([np.load(f'output/plot_data/loss_A2A_id_loss_{ep}.npy') for ep in range(start_epoch, end_epoch)])
    # B2B_id_loss = np.concatenate([np.load(f'output/plot_data/loss_B2B_id_loss_{ep}.npy') for ep in range(start_epoch, end_epoch)])
    # A_d_loss = np.concatenate([np.load(f'output/plot_data/loss_A_d_loss_{ep}.npy') for ep in range(start_epoch, end_epoch)])
    # B_d_loss = np.concatenate([np.load(f'output/plot_data/loss_B_d_loss_{ep}.npy') for ep in range(start_epoch, end_epoch)])
    # iterations = np.concatenate([np.load(f'output/plot_data/iterations_{ep}.npy') for ep in range(start_epoch, end_epoch)])

    # # Plot the loss data for each iteration
    # plt.plot(iterations, A2B_g_loss, label='A2B_g_loss')
    # plt.plot(iterations, B2A_g_loss, label='B2A_g_loss')
    # plt.plot(iterations, A2B2A_cycle_loss, label='A2B2A_cycle_loss')
    # plt.plot(iterations, B2A2B_cycle_loss, label='B2A2B_cycle_loss')
    # plt.plot(iterations, A2A_id_loss, label='A2A_id_loss')
    # plt.plot(iterations, B2B_id_loss, label='B2B_id_loss')
    # plt.plot(iterations, A_d_loss, label='A_d_loss')
    # plt.plot(iterations, B_d_loss, label='B_d_loss')
    # plt.xlabel('Iterations')
    # plt.ylabel('Loss')
    # plt.title('Loss vs Iterations')
    # plt.legend()
    # plt.show()
    # plt.savefig(f'output/plot_figure/loss_vs_iterations_{start_epoch}_to_{end_epoch}.png')


def save_plot_data(iterations, A2B_g_loss, B2A_g_loss, A2B2A_cycle_loss, B2A2B_cycle_loss, A2A_id_loss, B2B_id_loss, A_d_loss, B_d_loss, ep, name, method):
    """Save the loss data for each iteration into a separate file."""
    np.save(
        f'output/{method}/plot_data/{name}/loss_A2B_g_loss_{ep}.npy', A2B_g_loss)
    np.save(
        f'output/{method}/plot_data/{name}/loss_B2A_g_loss_{ep}.npy', B2A_g_loss)
    np.save(
        f'output/{method}/plot_data/{name}/loss_A2B2A_cycle_loss_{ep}.npy', A2B2A_cycle_loss)
    np.save(
        f'output/{method}/plot_data/{name}/loss_B2A2B_cycle_loss_{ep}.npy', B2A2B_cycle_loss)
    np.save(
        f'output/{method}/plot_data/{name}/loss_A2A_id_loss_{ep}.npy', A2A_id_loss)
    np.save(
        f'output/{method}/plot_data/{name}/loss_B2B_id_loss_{ep}.npy', B2B_id_loss)
    np.save(
        f'output/{method}/plot_data/{name}/loss_A_d_loss_{ep}.npy', A_d_loss)
    np.save(
        f'output/{method}/plot_data/{name}/loss_B_d_loss_{ep}.npy', B_d_loss)
    np.save(
        f'output/{method}/plot_data/{name}/iterations_{ep}.npy', iterations)


def temporary_plot(g_loss_dir, d_loss_dir, cycle_loss_dir, id_loss_dir, iterations, A2B_g_loss, B2A_g_loss, A2B2A_cycle_loss, B2A2B_cycle_loss, A2A_id_loss, B2B_id_loss, A_d_loss, B_d_loss, ep):
    """Temporary plot."""
    # plot the loss
    plt.figure()
    plt.plot(iterations, A2B_g_loss, label='A2B_g_loss')
    plt.plot(iterations, B2A_g_loss, label='B2A_g_loss')
    plt.legend()
    plt.title('Generator Losses')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.savefig(py.join(g_loss_dir, f'generator_losses_{ep}.png'))
    plt.close()

    plt.figure()
    plt.plot(iterations, A2B2A_cycle_loss, label='A2B2A_cycle_loss')
    plt.plot(iterations, B2A2B_cycle_loss, label='B2A2B_cycle_loss')
    plt.legend()
    plt.title('Cycle Losses')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.savefig(py.join(cycle_loss_dir, f'cycle_losses_{ep}.png'))
    plt.close()

    plt.figure()
    plt.plot(iterations, A2A_id_loss, label='A2A_id_loss')
    plt.plot(iterations, B2B_id_loss, label='B2B_id_loss')
    plt.legend()
    plt.title('Identity Losses')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.savefig(py.join(id_loss_dir, f'identity_losses_{ep}.png'))
    plt.close()

    plt.figure()
    plt.plot(iterations, A_d_loss, label='A_d_loss')
    plt.plot(iterations, B_d_loss, label='B_d_loss')
    plt.legend()
    plt.title('Discriminator Losses')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.savefig(py.join(d_loss_dir, f'discriminator_losses_{ep}.png'))
    plt.close()


def save_psnr_and_ssim_data(iterations, ssim_A2B_value_list, psnr_A2B_value_list, ssim_B2A_value_list, psnr_B2A_value_list, ep, name, method):
    """Save the PSNR and SSIM data for each iteration into a separate file."""
    np.save(
        f'output/{method}/plot_data/{name}/ssim_A2B_value_list_{ep}.npy', ssim_A2B_value_list)
    np.save(
        f'output/{method}/plot_data/{name}/psnr_A2B_value_list_{ep}.npy', psnr_A2B_value_list
    )
    np.save(
        f'output/{method}/plot_data/{name}/ssim_B2A_value_list_{ep}.npy', ssim_B2A_value_list)
    np.save(
        f'output/{method}/plot_data/{name}/psnr_B2A_value_list_{ep}.npy', psnr_B2A_value_list)

    # np.save(f'output/{method}/plot_data/{name}/iterations_{ep}.npy', iterations)


def temporary_plot_psnr_ssim(ssim_dir, psnr_dir, iterations, ssim_A2B_value_list, psnr_A2B_value_list, ssim_B2A_value_list, psnr_B2A_value_list, ep):
    """Plot the PSNR and SSIM data for each iteration."""
    # 2 Plots for psnr/ssim data
    plt.figure()
    plt.plot(iterations, ssim_A2B_value_list, label='ssim_A2B')
    plt.plot(iterations, ssim_B2A_value_list, label='ssim_B2A')
    plt.legend()
    plt.title('SSIM')
    plt.xlabel('Iterations')
    plt.ylabel('SSIM')
    plt.savefig(py.join(ssim_dir, f'ssim_{ep}.png'))
    plt.close()

    plt.figure()
    plt.plot(iterations, psnr_A2B_value_list, label='psnr_A2B')
    plt.plot(iterations, psnr_B2A_value_list, label='psnr_B2A')
    plt.legend()
    plt.title('PSNR')
    plt.xlabel('Iterations')
    plt.ylabel('PSNR')
    plt.savefig(py.join(psnr_dir, f'psnr_{ep}.png'))
    plt.close()


if __name__ == '__main__':
    main()
