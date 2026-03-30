import matplotlib.pyplot as plt

def plot_dIdV(backgate, dIdV):
    plt.figure()
    plt.plot(backgate, dIdV, '.-')
    plt.show()

