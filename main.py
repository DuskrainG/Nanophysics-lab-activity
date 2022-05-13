import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


data = pd.read_csv("bulk gold dieletric functions.txt", sep="\t", header=None)
data.columns = ["lambda", "epsilon1", "epsilon2"]
wavelength = np.array(data["lambda"])
epsilon1_bulk = np.array(data["epsilon1"])
epsilon2_bulk = np.array(data["epsilon2"])


plt.figure(figsize=(10, 6), dpi=100)
plt.plot(wavelength, epsilon1_bulk, color="green", label=r"$\epsilon_1$")
plt.plot(wavelength, epsilon2_bulk, color="red", label=r"$\epsilon_2$")

plt.title("Bulk gold dielectric functions", fontdict={"fontname": "Calibri", "fontsize": 20})
plt.xlabel(r"$\lambda$", fontdict={"fontsize": 14})
plt.ylabel(r"$\epsilon$", fontdict={"fontsize": 14})
plt.legend(fontsize=10, ncol=2)
plt.tight_layout()

#plt.savefig("grafico.png", dpi=200)
