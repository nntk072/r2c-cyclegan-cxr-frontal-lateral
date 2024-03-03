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

def main():
    for ep in range(0, 1000):
        A2B_g_loss = np.load(f'output/plot_data/loss_A2B_g_loss_{ep}.npy')
        B2A_g_loss = np.load(f'output/plot_data/loss_B2A_g_loss_{ep}.npy')
        A2B2A_cycle_loss = np.load(f'output/plot_data/loss_A2B2A_cycle_loss_{ep}.npy')
        B2A2B_cycle_loss = np.load(f'output/plot_data/loss_B2A2B_cycle_loss_{ep}.npy')
        A2A_id_loss = np.load(f'output/plot_data/loss_A2A_id_loss_{ep}.npy')
        B2B_id_loss = np.load(f'output/plot_data/loss_B2B_id_loss_{ep}.npy')
        A_d_loss = np.load(f'output/plot_data/loss_A_d_loss_{ep}.npy')
        B_d_loss = np.load(f'output/plot_data/loss_B_d_loss_{ep}.npy')
        iterations = np.load(f'output/plot_data/iterations_{ep}.npy')

        # Plot the loss data for each iteration
        plt.plot(iterations, A2B_g_loss, label='A2B_g_loss')
        plt.plot(iterations, B2A_g_loss, label='B2A_g_loss')
        plt.plot(iterations, A2B2A_cycle_loss, label='A2B2A_cycle_loss')
        plt.plot(iterations, B2A2B_cycle_loss, label='B2A2B_cycle_loss')
        plt.plot(iterations, A2A_id_loss, label='A2A_id_loss')
        plt.plot(iterations, B2B_id_loss, label='B2B_id_loss')
        plt.plot(iterations, A_d_loss, label='A_d_loss')
        plt.plot(iterations, B_d_loss, label='B_d_loss')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('Loss vs Iterations')
        plt.legend()
        plt.show()
        plt.savefig(f'output/plot_figure/loss_vs_iterations_{ep}.png')
        plt.close()
    
    
    # Asking for the starting and ending epoch and concatenating the data for the plot
    # Start epoch
    start_epoch = int(input('Enter the starting epoch: '))
    # End epoch
    end_epoch = int(input('Enter the ending epoch: '))
    if start_epoch < 0 or end_epoch < 0:
        raise ValueError('The starting and ending epoch must be greater than 0')
    if start_epoch > end_epoch:
        raise ValueError('The starting epoch must be less than the ending epoch')

    # Concatenate the data for the plot
    A2B_g_loss = np.concatenate([np.load(f'output/plot_data/loss_A2B_g_loss_{ep}.npy') for ep in range(start_epoch, end_epoch)])
    B2A_g_loss = np.concatenate([np.load(f'output/plot_data/loss_B2A_g_loss_{ep}.npy') for ep in range(start_epoch, end_epoch)])
    A2B2A_cycle_loss = np.concatenate([np.load(f'output/plot_data/loss_A2B2A_cycle_loss_{ep}.npy') for ep in range(start_epoch, end_epoch)])
    B2A2B_cycle_loss = np.concatenate([np.load(f'output/plot_data/loss_B2A2B_cycle_loss_{ep}.npy') for ep in range(start_epoch, end_epoch)])
    A2A_id_loss = np.concatenate([np.load(f'output/plot_data/loss_A2A_id_loss_{ep}.npy') for ep in range(start_epoch, end_epoch)])
    B2B_id_loss = np.concatenate([np.load(f'output/plot_data/loss_B2B_id_loss_{ep}.npy') for ep in range(start_epoch, end_epoch)])
    A_d_loss = np.concatenate([np.load(f'output/plot_data/loss_A_d_loss_{ep}.npy') for ep in range(start_epoch, end_epoch)])
    B_d_loss = np.concatenate([np.load(f'output/plot_data/loss_B_d_loss_{ep}.npy') for ep in range(start_epoch, end_epoch)])
    iterations = np.concatenate([np.load(f'output/plot_data/iterations_{ep}.npy') for ep in range(start_epoch, end_epoch)])

    # Plot the loss data for each iteration
    plt.plot(iterations, A2B_g_loss, label='A2B_g_loss')
    plt.plot(iterations, B2A_g_loss, label='B2A_g_loss')
    plt.plot(iterations, A2B2A_cycle_loss, label='A2B2A_cycle_loss')
    plt.plot(iterations, B2A2B_cycle_loss, label='B2A2B_cycle_loss')
    plt.plot(iterations, A2A_id_loss, label='A2A_id_loss')
    plt.plot(iterations, B2B_id_loss, label='B2B_id_loss')
    plt.plot(iterations, A_d_loss, label='A_d_loss')
    plt.plot(iterations, B_d_loss, label='B_d_loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Loss vs Iterations')
    plt.legend()
    plt.show()
    plt.savefig(f'output/plot_figure/loss_vs_iterations_{start_epoch}_to_{end_epoch}.png')

def save_plot_data(iterations, A2B_g_loss, B2A_g_loss, A2B2A_cycle_loss, B2A2B_cycle_loss, A2A_id_loss, B2B_id_loss, A_d_loss, B_d_loss, ep):
    """Save the loss data for each iteration into a separate file."""
    np.save(f'output/plot_data/loss_A2B_g_loss_{ep}.npy', A2B_g_loss)
    np.save(f'output/plot_data/loss_B2A_g_loss_{ep}.npy', B2A_g_loss)
    np.save(f'output/plot_data/loss_A2B2A_cycle_loss_{ep}.npy', A2B2A_cycle_loss)
    np.save(f'output/plot_data/loss_B2A2B_cycle_loss_{ep}.npy', B2A2B_cycle_loss)
    np.save(f'output/plot_data/loss_A2A_id_loss_{ep}.npy', A2A_id_loss)
    np.save(f'output/plot_data/loss_B2B_id_loss_{ep}.npy', B2B_id_loss)
    np.save(f'output/plot_data/loss_A_d_loss_{ep}.npy', A_d_loss)
    np.save(f'output/plot_data/loss_B_d_loss_{ep}.npy', B_d_loss)
    np.save(f'output/plot_data/iterations_{ep}.npy', iterations)

def temporary_plot(iterations, A2B_g_loss, B2A_g_loss, A2B2A_cycle_loss, B2A2B_cycle_loss, A2A_id_loss, B2B_id_loss, A_d_loss, B_d_loss):
    """Temporary plot."""
    plt.plot(iterations, A2B_g_loss, label='A2B_g_loss')
    plt.plot(iterations, B2A_g_loss, label='B2A_g_loss')
    plt.plot(iterations, A2B2A_cycle_loss, label='A2B2A_cycle_loss')
    plt.plot(iterations, B2A2B_cycle_loss, label='B2A2B_cycle_loss')
    plt.plot(iterations, A2A_id_loss, label='A2A_id_loss')
    plt.plot(iterations, B2B_id_loss, label='B2B_id_loss')
    plt.plot(iterations, A_d_loss, label='A_d_loss')
    plt.plot(iterations, B_d_loss, label='B_d_loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Loss vs Iterations')
    plt.legend()
    plt.show()
    plt.savefig(f'output/plot_figure/loss_vs_iterations_{start_epoch}_to_{end_epoch}.png')
    plt.close()
    
if __name__ == '__main__':
    main()