import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn.neighbors import NearestNeighbors
from datasets import generate_data
from ican import causal_inference

class Application(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.title("ICAN")

        # Frames for inputting dataset and datapoints
        self.input_frame = tk.Frame(self)
        self.input_frame.pack(side=tk.TOP)

        self.dataset_label = tk.Label(self.input_frame, text="Dataset:")
        self.dataset_label.pack(side=tk.LEFT)
        self.dataset_entry = tk.Entry(self.input_frame)
        self.dataset_entry.pack(side=tk.LEFT)

        self.datapoints_label = tk.Label(self.input_frame, text="Datapoints:")
        self.datapoints_label.pack(side=tk.LEFT)
        self.datapoints_entry = tk.Entry(self.input_frame)
        self.datapoints_entry.pack(side=tk.LEFT)

        # Button for running the algorithm
        self.run_button = tk.Button(self.input_frame, text="Run ICAN", command=self.run_ican)
        self.run_button.pack(side=tk.LEFT)

        # Frame for outputting causal structure
        self.structure_frame = tk.Frame(self)
        self.structure_frame.pack(side=tk.TOP)
        self.structure_label = tk.Label(self.structure_frame, text="Structure:")
        self.structure_label.pack(side=tk.LEFT)
        self.structure_output = tk.Text(self.structure_frame, height=1, width=20)
        self.structure_output.pack()

        # Frame for outputting variance
        self.var_frame = tk.Frame(self)
        self.var_frame.pack(side=tk.TOP)
        self.var_label = tk.Label(self.var_frame, text="Var(X) / Var(Y):")
        self.var_label.pack(side=tk.LEFT)
        self.var_output = tk.Text(self.var_frame, height=1, width=20)
        self.var_output.pack()

        # Frame for plots
        self.plot_frame = tk.Frame(self)
        self.plot_frame.pack(side=tk.BOTTOM)

        # Create empty plots
        self.create_empty_plots()

    def run_ican(self):
        # Get inputs
        dataset = int(self.dataset_entry.get())
        datapoints = int(self.datapoints_entry.get())

        # Run ICAN
        T, X, Y, X_true, Y_true = generate_data(datapoints, dataset)
        T_hat, var, s1_hat, s2_hat, result, structure = causal_inference(X.reshape(-1, 1), Y.reshape(-1, 1))
        
        # Generate plots
        self.clear_plot()
        self.create_plots(T, T_hat, X, X_true, Y, Y_true, s1_hat, s2_hat, structure)

        # Display causal structure
        self.structure_output.delete('1.0', tk.END)
        self.structure_output.insert(tk.END, structure)

        # Display variance
        self.var_output.delete('1.0', tk.END)
        self.var_output.insert(tk.END, var)

    def create_plots(self, T, T_hat, X, X_true, Y, Y_true, s1_hat, s2_hat, structure):
        # Plot curve using Nearest Neighbors (because s=(u,v) is not always injective)
        def plotCurves(X, Y, style):
            data = np.column_stack([X, Y])  # Needed for NN

            # Use NearestNeighbors to find two closest points
            nbrs = NearestNeighbors(n_neighbors=3, algorithm="ball_tree").fit(data)
            distances, indices = nbrs.kneighbors(data)

            # Draw lines to two closest neighbors for each point (assuming they are the left and right neighbors)
            for i in range(len(data)):
                for j in range(1, 3):
                    axs[1, 0].plot([data[i, 0], data[indices[i, j], 0]], [data[i, 1], data[indices[i, j], 1]], style)

        Nx = X.reshape(-1, 1) - s1_hat.predict(T_hat.reshape(-1, 1)).reshape(-1, 1)
        Ny = Y.reshape(-1, 1) - s2_hat.predict(T_hat.reshape(-1, 1)).reshape(-1, 1)

        Nx = Nx.reshape(-1, 1)
        Ny = Ny.reshape(-1, 1)
        T_hat = T_hat.reshape(-1, 1)

        fig, axs = plt.subplots(3, 2, figsize=(10, 10))

        # Scatter plot for X and Y
        axs[0, 0].scatter(X, Y)
        axs[0, 0].set_title('X vs Y')

        # Plot curves
        axs[1, 0].set_title("true (black) and estimated (red) curves")
        axs[1, 0].scatter(X, Y)

        T_fine = np.linspace(T_hat[0], T_hat[-1], 2000)

        predicted_X = s1_hat.predict(T_fine).reshape(-1, 1)
        predicted_Y = s2_hat.predict(T_fine).reshape(-1, 1)

        plotCurves(predicted_X, predicted_Y, "r-")
        plotCurves(X_true, Y_true, "k-")

        if structure == "X<-T->Y":
            # Scatter plot for X and Y
            axs[2, 0].scatter(T, T_hat)
            axs[2, 0].set_title('true confounder vs estimated confounder')

        # Scatter plot for Nx and Ny
        axs[0, 1].scatter(Nx, Ny)
        axs[0, 1].set_title('estimated Nx vs estimated Ny')

        # Scatter plot for T and Nx
        axs[1, 1].scatter(T_hat, Nx)
        axs[1, 1].set_title('estimated T vs estimated Nx')

        # Scatter plot for T and Ny
        axs[2, 1].scatter(T_hat, Ny)
        axs[2, 1].set_title('estimated T vs estimated Ny')

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

    def clear_plot(self):
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

    # Create empty plots
    def create_empty_plots(self):
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        fig, axs = plt.subplots(3, 2, figsize=(10, 10))
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()


if __name__ == "__main__":
    app = Application()
    app.mainloop()

