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
    # Make an input
    method = input('Enter the method: ')
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
    epoch = int(input('Enter the number of epochs: '))

    for ep in range(0, epoch):
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
    for ep in range(0, epoch):  # the name of the folder is validation
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
    plt.xlim(ep_list[0], ep_list[-1])
    plt.xticks(np.arange(ep_list[0], ep_list[-1], epoch/10))
    for i in range(100, 0, -1):
        if max(max(A2B_g_loss_list), max(B2A_g_loss_list), max(A2B_g_loss_list_valid), max(B2A_g_loss_list_valid)) <= 0.25 * i:
            if max(max(A2B_g_loss_list), max(B2A_g_loss_list), max(A2B_g_loss_list_valid), max(B2A_g_loss_list_valid)) <= 0.25 * (i-1):
                i -= 1
                continue
            plt.ylim(0, 0.25 * i)
            plt.yticks(np.arange(0, 0.25 * i, 0.25*i/20))
            break
    # Save as high-res image
    plt.savefig(f'output/{method}/generator_losses.png')
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
    # plt.minorticks_on()
    # plt.grid(which='minor', color='gray', linewidth=0.5)
    plt.xlim(ep_list[0], ep_list[-1])
    plt.xticks(np.arange(ep_list[0], ep_list[-1], epoch/10))
    for i in range(100, 0, -1):
        if max(max(A2B2A_cycle_loss_list), max(B2A2B_cycle_loss_list), max(A2B2A_cycle_loss_list_valid), max(B2A2B_cycle_loss_list_valid)) <= 0.25 * i:
            if max(max(A2B2A_cycle_loss_list), max(B2A2B_cycle_loss_list), max(A2B2A_cycle_loss_list_valid), max(B2A2B_cycle_loss_list_valid)) <= 0.25 * (i-1):
                i -= 1
                continue
            plt.ylim(0, 0.25 * i)
            plt.yticks(np.arange(0, 0.25 * i, 0.25*i/20))
            break
    plt.savefig(f'output/{method}/cycle_losses.png')  # Save as high-res image
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
    # plt.minorticks_on()
    # plt.grid(which='minor', color='gray', linewidth=0.5)
    plt.xlim(ep_list[0], ep_list[-1])
    plt.xticks(np.arange(ep_list[0], ep_list[-1], epoch/10))
    for i in range(100, 0, -1):
        if max(max(A2A_id_loss_list), max(B2B_id_loss_list), max(A2A_id_loss_list_valid), max(B2B_id_loss_list_valid)) <= 0.25 * i:
            if max(max(A2A_id_loss_list), max(B2B_id_loss_list), max(A2A_id_loss_list_valid), max(B2B_id_loss_list_valid)) <= 0.25 * (i-1):
                i -= 1
                continue
            plt.ylim(0, 0.25 * i)
            plt.yticks(np.arange(0, 0.25 * i, 0.25*i/20))
            break
    # Save as high-res image
    plt.savefig(f'output/{method}/identity_losses.png')
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
    # plt.minorticks_on()
    # plt.grid(which='minor', color='gray', linewidth=0.5)
    plt.xlim(ep_list[0], ep_list[-1])
    plt.xticks(np.arange(ep_list[0], ep_list[-1], epoch/10))
    for i in range(100, 0, -1):
        if max(max(A_d_loss_list), max(B_d_loss_list), max(A_d_loss_list_valid), max(B_d_loss_list_valid)) <= 0.25 * i:
            if max(max(A_d_loss_list), max(B_d_loss_list), max(A_d_loss_list_valid), max(B_d_loss_list_valid)) <= 0.25 * (i-1):
                i -= 1
                continue
            plt.ylim(0, 0.25 * i)
            plt.yticks(np.arange(0, 0.25 * i, 0.25*i/20))
            break
    # Save as high-res image
    plt.savefig(f'output/{method}/discriminator_losses.png')
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
    plt.xlim(ep_list[0], ep_list[-1])
    plt.xticks(np.arange(ep_list[0], ep_list[-1], epoch/10))
    for i in range(1000, 0, -1):
        if max(max(psnr_A2B_list), max(psnr_B2A_list), max(psnr_A2B_list_valid), max(psnr_B2A_list_valid)) <= 0.25 * i:
            if max(max(psnr_A2B_list), max(psnr_B2A_list), max(psnr_A2B_list_valid), max(psnr_B2A_list_valid)) <= 0.25 * (i-1):
                i -= 1
                continue
            plt.ylim(0, 0.25 * i)
            plt.yticks(np.arange(0, 0.25 * i, 0.25*i/20))
            break
    plt.savefig(f'output/{method}/psnr.png')  # Save as high-res image
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
    plt.xlim(ep_list[0], ep_list[-1])
    plt.xticks(np.arange(ep_list[0], ep_list[-1], epoch/10))
    for i in range(100, 0, -1):
        if max(max(ssim_A2B_list), max(ssim_B2A_list), max(ssim_A2B_list_valid), max(ssim_B2A_list_valid)) <= 0.25 * i:
            if max(max(ssim_A2B_list), max(ssim_B2A_list), max(ssim_A2B_list_valid), max(ssim_B2A_list_valid)) <= 0.25 * (i-1):
                i -= 1
                continue
            plt.ylim(0, 0.25 * i)
            plt.yticks(np.arange(0, 0.25 * i, 0.25*i/20))
            break
    plt.savefig(f'output/{method}/ssim.png')  # Save as high-res image


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
    # Make the plot with the same format as the plot in main function
    plt.figure(figsize=(20, 12))  # Increase figure size
    plt.plot(iterations, A2B_g_loss, label='A2B_g_loss', linewidth=1)
    plt.plot(iterations, B2A_g_loss, label='B2A_g_loss', linewidth=1)
    plt.legend(fontsize='large')
    plt.title('Generator Losses', fontsize='x-large')
    plt.xlabel('Iterations', fontsize='large')
    plt.ylabel('Loss', fontsize='large')
    plt.grid(which='major', color='black', linewidth=0.5)
    plt.xlim(iterations[0], iterations[-1])
    plt.xticks(np.arange(iterations[0], iterations[-1], 1000))
    for i in range(100, 0, -1):
        if max(max(A2B_g_loss), max(B2A_g_loss)) <= 0.25 * i:
            if max(max(A2B_g_loss), max(B2A_g_loss)) <= 0.25 * (i-1):
                i -= 1
                continue
            plt.ylim(0, 0.25 * i)
            plt.yticks(np.arange(0, 0.25 * i, 0.25*i/20))
            break
    # Save as high-res image
    plt.savefig(py.join(g_loss_dir, f'generator_losses_{ep}.png'))
    plt.close()

    plt.figure(figsize=(20, 12))  # Increase figure size
    plt.plot(iterations, A2B2A_cycle_loss,
             label='A2B2A_cycle_loss', linewidth=1)
    plt.plot(iterations, B2A2B_cycle_loss,
             label='B2A2B_cycle_loss', linewidth=1)
    plt.legend(fontsize='large')
    plt.title('Cycle Losses', fontsize='x-large')
    plt.xlabel('Iterations', fontsize='large')
    plt.ylabel('Loss', fontsize='large')
    plt.grid(which='major', color='black', linewidth=0.5)
    plt.xlim(iterations[0], iterations[-1])
    plt.xticks(np.arange(iterations[0], iterations[-1], 1000))
    for i in range(100, 0, -1):
        if max(max(A2B2A_cycle_loss), max(B2A2B_cycle_loss)) <= 0.25 * i:
            if max(max(A2B2A_cycle_loss), max(B2A2B_cycle_loss)) <= 0.25 * (i-1):
                i -= 1
                continue
            plt.ylim(0, 0.25 * i)
            plt.yticks(np.arange(0, 0.25 * i, 0.25*i/20))
            break
    plt.savefig(py.join(cycle_loss_dir, f'cycle_losses_{ep}.png'))
    plt.close()

    plt.figure(figsize=(20, 12))  # Increase figure size
    plt.plot(iterations, A2A_id_loss, label='A2A_id_loss', linewidth=1)
    plt.plot(iterations, B2B_id_loss, label='B2B_id_loss', linewidth=1)
    plt.legend(fontsize='large')
    plt.title('Identity Losses', fontsize='x-large')
    plt.xlabel('Iterations', fontsize='large')
    plt.ylabel('Loss', fontsize='large')
    plt.grid(which='major', color='black', linewidth=0.5)
    plt.xlim(iterations[0], iterations[-1])
    plt.xticks(np.arange(iterations[0], iterations[-1], (iterations[-1]-iterations[0])/10))
    for i in range(100, 0, -1):
        if max(max(A2A_id_loss), max(B2B_id_loss)) <= 0.25 * i:
            if max(max(A2A_id_loss), max(B2B_id_loss)) <= 0.25 * (i-1):
                i -= 1
                continue
            plt.ylim(0, 0.25 * i)
            plt.yticks(np.arange(0, 0.25 * i, 0.25*i/20))
            break
    # Save as high-res image
    plt.savefig(py.join(id_loss_dir, f'identity_losses_{ep}.png'))
    plt.close()

    plt.figure(figsize=(20, 12))  # Increase figure size
    plt.plot(iterations, A_d_loss, label='A_d_loss', linewidth=1)
    plt.plot(iterations, B_d_loss, label='B_d_loss', linewidth=1)
    plt.legend(fontsize='large')
    plt.title('Discriminator Losses', fontsize='x-large')
    plt.xlabel('Iterations', fontsize='large')
    plt.ylabel('Loss', fontsize='large')
    plt.grid(which='major', color='black', linewidth=0.5)
    plt.xlim(iterations[0], iterations[-1])
    plt.xticks(np.arange(iterations[0], iterations[-1], (iterations[-1]-iterations[0])/10))
    for i in range(100, 0, -1):
        if max(max(A_d_loss), max(B_d_loss)) <= 0.25 * i:
            if max(max(A_d_loss), max(B_d_loss)) <= 0.25 * (i-1):
                i -= 1
                continue
            plt.ylim(0, 0.25 * i)
            plt.yticks(np.arange(0, 0.25 * i, 0.25*i/20))
            break
    # Save as high-res image
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
    # Make the plot with the same format as the plot in main function
    plt.figure(figsize=(20, 12))  # Increase figure size
    plt.plot(iterations, psnr_A2B_value_list, label='psnr_A2B', linewidth=1)
    plt.plot(iterations, psnr_B2A_value_list, label='psnr_B2A', linewidth=1)
    plt.legend(fontsize='large')
    plt.title('PSNR', fontsize='x-large')
    plt.xlabel('Iterations', fontsize='large')
    plt.ylabel('PSNR', fontsize='large')
    plt.grid(which='major', color='black', linewidth=0.5)
    plt.xlim(iterations[0], iterations[-1])
    plt.xticks(np.arange(iterations[0], iterations[-1], 1000))
    for i in range(1000, 0, -1):
        if max(max(psnr_A2B_value_list), max(psnr_B2A_value_list)) <= 0.1 * i:
            if max(max(psnr_A2B_value_list), max(psnr_B2A_value_list)) <= 0.1 * (i-1):
                i -= 1
                continue
            min_psnr = int(min(min(psnr_A2B_value_list),
                           min(psnr_B2A_value_list)))
            min_psnr = min_psnr - min_psnr % 5
            plt.ylim(min_psnr, 0.1 * i)
            plt.yticks(np.arange(min_psnr, 0.1 * i, (0.1*i-min_psnr)/20))
            break
    # Save as high-res image
    plt.savefig(py.join(psnr_dir, f'psnr_{ep}.png'))
    plt.close()

    plt.figure(figsize=(20, 12))  # Increase figure size
    plt.plot(iterations, ssim_A2B_value_list, label='ssim_A2B', linewidth=1)
    plt.plot(iterations, ssim_B2A_value_list, label='ssim_B2A', linewidth=1)
    plt.legend(fontsize='large')
    plt.title('SSIM', fontsize='x-large')
    plt.xlabel('Iterations', fontsize='large')
    plt.ylabel('SSIM', fontsize='large')
    plt.grid(which='major', color='black', linewidth=0.5)
    plt.xlim(iterations[0], iterations[-1])
    plt.xticks(np.arange(iterations[0], iterations[-1], 1000))
    for i in range(1000, 0, -1):
        if max(max(ssim_A2B_value_list), max(ssim_B2A_value_list)) <= 0.1 * i:
            if max(max(ssim_A2B_value_list), max(ssim_B2A_value_list)) <= 0.1 * (i-1):
                i -= 1
                continue
            min_ssim = int(min(min(ssim_A2B_value_list),
                           min(ssim_B2A_value_list))*1000)
            # print(min_ssim)
            min_ssim = min_ssim - min_ssim % 50
            # print(min_ssim)
            plt.ylim(min_ssim/1000, 0.1 * i)
            plt.yticks(np.arange(min_ssim/1000, 0.1 * i, (0.1*i-min_ssim/1000)/20))
            break
    plt.savefig(py.join(ssim_dir, f'ssim_{ep}.png'))  # Save as high-res image
    plt.close()


if __name__ == '__main__':
    main()
