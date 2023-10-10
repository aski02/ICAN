import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datasets import generate_data
from ican import causal_inference
import os

class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ICAN")
        self.geometry("1200x800")
        self.configure(bg="white")

        self.variance_value = tk.StringVar()
        self.hsic_value = tk.StringVar()
        self.neighbor_value = tk.StringVar()

        self._configure_grids()
        self._create_main_frame()

        self.plot_save_dir = "generated_plots"
        if not os.path.exists(self.plot_save_dir):
            os.makedirs(self.plot_save_dir)


    def _configure_grids(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)


    def _create_main_frame(self):
        self.canvas = tk.Canvas(self)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.canvas.configure(yscrollcommand=scrollbar.set)

        main_frame = ttk.Frame(self.canvas, padding="20")
        self.canvas.create_window((0, 0), window=main_frame, anchor="nw")
        main_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        self.input_frame = self._create_input_frame(main_frame)
        self.input_frame.grid(row=1, pady=10, sticky="ew")

        self.output_frame = self._create_output_frame(main_frame)
        self.output_frame.grid(row=3, pady=10, sticky="ew")

        self.plot_frame = ttk.LabelFrame(main_frame, text="Plots", padding="10")
        self.plot_frame.grid(row=5, sticky="ew")


    def _create_input_frame(self, parent):
        frame = ttk.LabelFrame(parent, text="Inputs", padding=(20, 10))
        dataset_frame = ttk.Frame(frame)
        dataset_frame.grid(row=0, column=0, padx=5, sticky="ew")
        _, self.dataset_combobox = self._create_label_and_combobox(dataset_frame, "Dataset:", [f"Dataset {i}" for i in range(0, 20)], row=0, column=0)
        _, self.datapoints_entry = self._create_label_and_entry(dataset_frame, "Datapoints:", row=1, column=0)
        self.datapoints_entry.insert(0, "50")
        _, self.iterations_entry = self._create_label_and_entry(dataset_frame, "Iterations:", row=2, column=0)
        self.iterations_entry.insert(0, "3")

        other_frame = ttk.Frame(frame)
        other_frame.grid(row=0, column=1, padx=5, sticky="ew")
        _, self.dim_reduction_combobox = self._create_label_and_combobox(other_frame, "Dim. reduction:", ["Isomap", "TSNE", "LLE", "PCA", "KernelPCA"], row=0, column=0)
        _, self.independence_combobox = self._create_label_and_combobox(other_frame, "Independence test:", ["HSIC", "MI", "Pearson", "Spearman", "Kendalltau"], row=1, column=0)
        _, self.regression_combobox = self._create_label_and_combobox(other_frame, "Regression:", ["GPR", "NuSVR", "DecisionTree", "RandomForest", "XGBoost"], row=2, column=0)
                
        _, self.minFitting_combobox = self._create_label_and_combobox(other_frame, "Minimize distance:", ["Nelder-Mead", "L-BFGS-B", "BFGS", "powell", "SLSQP", "TNC", "CG", "trust-constr", "COBYLA"], row=0, column=2)
        _, self.minProjection_combobox = self._create_label_and_combobox(other_frame, "Minimize dependence:", ["Nelder-Mead", "L-BFGS-B", "BFGS", "powell", "SLSQP", "TNC", "CG", "trust-constr", "COBYLA"], row=1, column=2)
        _, self.kernel_combobox = self._create_label_and_combobox(other_frame, "Kernel for GPR:", ["RBF", "RBF + White", "Matern", "Matern + White", "Matern + ExpSineSquared", "DotProduct", "RationalQuadratic", "RationalQuadratic + White", "ExpSineSquared", "ExpSineSquared + White"], row=2, column=2)
        
        thresholds_frame = ttk.Frame(frame)
        thresholds_frame.grid(row=0, column=2, padx=5, sticky="ew")
        self._create_label_and_slider(thresholds_frame, "Variance threshold:", row=0, column=0, slider_range=(1.0, 5.0), value_var=self.variance_value, init_value=2.0)
        self._create_label_and_slider(thresholds_frame, "Significance level:", row=1, column=0, slider_range=(0.01, 0.2), value_var=self.hsic_value, init_value=0.05)
        self._create_label_and_slider(thresholds_frame, "Neighbors for dim. red.:", row=2, column=0, slider_range=(0.05, 0.5), value_var=self.neighbor_value, init_value=0.1)
        
        self.run_button = ttk.Button(frame, text="Run ICAN", command=self.run_ican)
        self.run_button.grid(row=1, columnspan=3, pady=10)

        return frame


    def _create_label_and_combobox(self, parent, text, values, row, column):
        label = ttk.Label(parent, text=text)
        label.grid(row=row, column=column, padx=5, pady=5)
        
        combobox = ttk.Combobox(parent, values=values)
        combobox.grid(row=row, column=column+1, padx=5, pady=5)
        combobox.set(values[0])

        return label, combobox


    def _create_label_and_entry(self, parent, text, row, column):
        label = ttk.Label(parent, text=text)
        label.grid(row=row, column=column, padx=5, pady=5)

        entry = ttk.Entry(parent)
        entry.grid(row=row, column=column+1, padx=5, pady=5)

        return label, entry


    def _create_label_and_slider(self, parent, text, row, column, slider_range=(1.0, 5.0), value_var=None, value_column_offset=2, init_value=None):
        label = ttk.Label(parent, text=text)
        label.grid(row=row, column=column, padx=5, pady=5)

        slider = ttk.Scale(parent, from_=slider_range[0], to=slider_range[1], orient=tk.HORIZONTAL, command=lambda val, var=value_var: self._update_slider_value(val, var))
        slider.grid(row=row, column=column+1, padx=5, pady=5)
        
        if init_value:
        	slider.set(init_value)
        else:
        	slider.set(slider_range[1] / 2.0)

        value_label = ttk.Label(parent, textvariable=value_var)
        value_label.grid(row=row, column=column+value_column_offset, padx=5, pady=5)

        return label, slider, value_label


    def _update_slider_value(self, val, value_var):
        value_var.set(f"{float(val):.2f}")


    def _create_output_frame(self, parent):
        frame = ttk.LabelFrame(parent, text="Outputs", padding=(20, 10))
        frame.grid(row=1, sticky="ew")

        # Variance quotient Output
        self._create_output_label_pair(frame, "Variance quotient:", row=0)
        _, self.var_output_value = self._create_output_label_pair(frame, "Var(Nx) / Var(Ny)", row=0)

        # P-values Output
        self._create_output_label_pair(frame, "p-value for X->Y:", row=1)
        _, self.p1_output_value = self._create_output_label_pair(frame, "p-value for X->Y:", row=1)
        self._create_output_label_pair(frame, "p-value for Y->X:", row=2)
        _, self.p2_output_value = self._create_output_label_pair(frame, "p-value for Y->X:", row=2)

        # Structure Output
        self._create_output_label_pair(frame, "Structure:", row=3)
        _, self.structure_output_value = self._create_output_label_pair(frame, "Structure:", row=3)

        return frame


    def _create_output_label_pair(self, parent, text, row):
        label = ttk.Label(parent, text=text)
        label.grid(row=row, column=0, padx=5, pady=5)

        value_label = ttk.Label(parent, text="N/A")
        value_label.grid(row=row, column=1, padx=5, pady=5)

        return label, value_label


    def run_ican(self):
        dataset_value = int(self.dataset_combobox.get().split()[-1])
        datapoints = int(self.datapoints_entry.get())
        iterations = int(self.iterations_entry.get())
        
        dim_reduction = self.dim_reduction_combobox.get()
        independence_method = self.independence_combobox.get()
        regression_method = self.regression_combobox.get()
        kernel = self.kernel_combobox.get()
        
        min_distance = self.minFitting_combobox.get()
        min_projection = self.minProjection_combobox.get()
        
        variance_threshold = float(self.variance_value.get())
        independence_threshold = float(self.hsic_value.get())
        neighbor_percentage = float(self.neighbor_value.get())
        
        # Run ICAN
        T, X, Y = generate_data(datapoints, dataset_value)
        X,Y = X.reshape(-1,1), Y.reshape(-1,1)
        T_hat, var, s1_hat, s2_hat, result, structure, p1, p2 = causal_inference(X.reshape(-1, 1), Y.reshape(-1, 1), dim_reduction, neighbor_percentage, iterations, kernel, variance_threshold, independence_threshold, regression_method, independence_method, min_distance, min_projection)
        
        # Normalize to variance 1
        X = (X - np.mean(X)) / np.std(X)
        Y = (Y - np.mean(Y)) / np.std(Y)
        
        self.var_output_value["text"] = str(var)
        self.structure_output_value["text"] = structure
        self.p1_output_value["text"] = str(p1)
        self.p2_output_value["text"] = str(p2)

        self.clear_plot()
        self.create_plots(T, T_hat, X, Y, s1_hat, s2_hat, structure)


    def create_plots(self, T, T_hat, X, Y, s1_hat, s2_hat, structure):
        Nx = (X.reshape(-1, 1) - s1_hat.predict(T_hat.reshape(-1, 1)).reshape(-1, 1)).reshape(-1, 1)
        Ny = (Y.reshape(-1, 1) - s2_hat.predict(T_hat.reshape(-1, 1)).reshape(-1, 1)).reshape(-1, 1)
        
        T_hat = T_hat.reshape(-1, 1)
        T_fine = np.sort(T_hat, axis=0)
        predicted_X = s1_hat.predict(T_fine).reshape(-1, 1)
        predicted_Y = s2_hat.predict(T_fine).reshape(-1, 1)

        def save_plot(func, title, filename, xlabel="", ylabel=""):
            fig, ax = plt.subplots()
            func(ax)
            #ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            fig.tight_layout()
            fig.savefig(os.path.join(self.plot_save_dir, filename))
            plt.close(fig)

        save_plot(lambda ax: ax.scatter(X, Y, edgecolors="b", facecolors="none", s=30, linewidth=0.5), 
                "X vs Y", "X_vs_Y.png", "X", "Y")
        
        save_plot(lambda ax: (ax.scatter(X, Y, edgecolors="b", facecolors="none", s=30, linewidth=0.5), 
                            ax.plot(predicted_X, predicted_Y, "r--", linewidth=1.2)), 
                "Estimated curve", "Estimated_curve.png", "X", "Y")
        
        if structure == "X<-T->Y":
            save_plot(lambda ax: ax.scatter(T, T_hat, edgecolors="b", facecolors="none", s=30, linewidth=0.5), 
                    "True confounder vs Estimated confounder", "True_vs_Estimated_confounder.png", "True confounder", "Estimated confounder")
        
        save_plot(lambda ax: ax.scatter(Nx, Ny, edgecolors="b", facecolors="none", s=30, linewidth=0.5), 
                "Estimated Nx vs Estimated Ny", "Nx_vs_Ny.png", "Estimated Nx", "Estimated Ny")
        
        save_plot(lambda ax: ax.scatter(T_hat, Nx, edgecolors="b", facecolors="none", s=30, linewidth=0.5), 
                "Estimated T vs Estimated Nx", "T_vs_Nx.png", "Estimated T", "Estimated Nx")
        
        save_plot(lambda ax: ax.scatter(T_hat, Ny, edgecolors="b", facecolors="none", s=30, linewidth=0.5), 
                "Estimated T vs Estimated Ny", "T_vs_Ny.png", "Estimated T", "Estimated Ny")
        
        # Now, display the original combined plot in the GUI
        fig, axs = plt.subplots(3, 2, figsize=(10, 10))
        axs[0, 0].scatter(X, Y, edgecolors="b", facecolors="none", s=30, linewidth=0.5)
        axs[0, 0].set_title("X vs Y")
        axs[0, 0].set_xlabel("X")
        axs[0, 0].set_ylabel("Y")
        
        axs[1, 0].scatter(X, Y, edgecolors="b", facecolors="none", s=30, linewidth=0.5)
        axs[1, 0].plot(predicted_X, predicted_Y, "r--", linewidth=1.2)
        axs[1, 0].set_title("estimated curve")
        axs[1, 0].set_xlabel("X")
        axs[1, 0].set_ylabel("Y")
        
        if structure == "X<-T->Y":
            axs[2, 0].scatter(T, T_hat, edgecolors="b", facecolors="none", s=30, linewidth=0.5)
            axs[2, 0].set_title("true confounder vs estimated confounder")
            axs[2, 0].set_xlabel("true confounder")
            axs[2, 0].set_ylabel("estimated confounder")
        
        axs[0, 1].scatter(Nx, Ny, edgecolors="b", facecolors="none", s=30, linewidth=0.5)
        axs[0, 1].set_title("estimated Nx vs estimated Ny")
        axs[0, 1].set_xlabel("estimated Nx")
        axs[0, 1].set_ylabel("estimated Ny")
        
        axs[1, 1].scatter(T_hat, Nx, edgecolors="b", facecolors="none", s=30, linewidth=0.5)
        axs[1, 1].set_title("estimated T vs estimated Nx")
        axs[1, 1].set_xlabel("estimated T")
        axs[1, 1].set_ylabel("estimated Nx")
        
        axs[2, 1].scatter(T_hat, Ny, edgecolors="b", facecolors="none", s=30, linewidth=0.5)
        axs[2, 1].set_title("estimated T vs estimated Ny")
        axs[2, 1].set_xlabel("estimated T")
        axs[2, 1].set_ylabel("estimated Ny")   
        
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()


    def clear_plot(self):
        for widget in self.plot_frame.winfo_children():
            widget.destroy()


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
