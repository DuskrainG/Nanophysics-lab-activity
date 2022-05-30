import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from matplotlib import ticker

# VARIABLES
fs = 20
wp = 1.37*10**16  #[Hz]
vf = 1.40*10**15  #[nm/s]
Gamma_bulk = 0.476*10**14  #[Hz] # from Ashcroft-Mermin (at 100°C)
c = 299792458*10**9 # [nm/s]
z = 10**7 # [nm]
# guesses of best fit values from chi-squared maps: use those as initial guesses for the size dependent fits
epsilonm_guess = 2.3
rho_guess = 3.5*10**-9
R_guess = 5
f_guess = 2*10**-6
epsilonm_fit = epsilonm_guess
rho_fit = rho_guess
R_fit = R_guess
e_guess = 0.001 # to get from ImageJ

bulk_data = pd.read_csv("bulk gold dieletric functions.txt", sep="\t", header=None)
bulk_data.columns = ["lambda", "epsilon1", "epsilon2"]
epsilon1_bulk = np.array(bulk_data["epsilon1"]) # from 200nm to 900nm
epsilon2_bulk = np.array(bulk_data["epsilon2"]) # from 200nm to 900nm

np_data = pd.read_csv("G01-NPs.dat", sep="\t", header=None)
np_data.columns = ["lambda", "absorbance"]
absorbance = np.array(np_data["absorbance"])
absorbance_r = absorbance[50:201]  # _r variables are restricted to the selected fit region # I've also tried [70:171]

l = np.array(np_data["lambda"])  # from 400nm to 800nm
l_bulk = np.array(bulk_data["lambda"]) # from 200nm to 900nm
l_r = l[50:201]

#%%
# FUNCTION DEFINITIONS

def Gamma(R): # [Hz]
    return Gamma_bulk * (1+(np.pi*vf/(4*Gamma_bulk*R)))

def omega(l):  #[Hz]
    return 2*np.pi*c/l

def epsilon1(l, R):
    l_index = l.astype(int)-200
    # try:
    #     l_index = [int(i) for i in l]
    # except:
    #     l_index = int(l)-200
    try:
        e1 = epsilon1_bulk[l_index]+wp**2 * (1/(omega(l)**2+Gamma_bulk**2)-1/(omega(l)**2+Gamma(R)**2))
        return e1
    except:
        print("Invalid wavelength")
          
def epsilon2(l, R):
    l_index = l.astype(int)-200
    # try:
    #     l_index = [int(i) for i in l]
    # except:
    #     l_index = int(l)-200
    try:
        e2 = epsilon2_bulk[l_index]-wp**2/omega(l) * (Gamma_bulk/(omega(l)**2+Gamma_bulk**2)-Gamma(R)/(omega(l)**2+Gamma(R)**2))
        return e2
    except:
        print("Invalid wavelength")

def Absorbance(l, R, epsilonm, rho): # rho = [nm**(-3)]
    return np.log10(np.e)*9*z*omega(l)/c*epsilonm**(3/2)*4/3*np.pi*R**3*rho*epsilon2(l, R)/((epsilon1(l, R)+2*epsilonm)**2+(epsilon2(l, R))**2)

def Absorbance_f(l, R, epsilonm, f):
    return np.log10(np.e)*9*z*omega(l)/c*epsilonm**(3/2)*f*epsilon2(l, R)/((epsilon1(l, R)+2*epsilonm)**2+(epsilon2(l, R))**2)

def Absorbance_epsilonmfix(l, R, rho):
    return Absorbance(l, R, epsilonm_guess, rho)

def Absorbance_f_epsilonmfix(l, R, f):
    return Absorbance_f(l, R, par_fit_rhofix[1], f)

def Absorbance_epsilonmfixfit(l, R, rho):
    return Absorbance(l, R, par_fit_rhofix[1], rho)

def Absorbance_rhofix(l, R, epsilonm):
    return Absorbance(l, R, epsilonm, par_fit_epsilonmfix[1])

def Absorbance_JC(l, epsilonm, f):
    l=l.astype(int)-200
    return np.log10(np.e)*9*z*omega(l)/c*epsilonm**(3/2)*f*epsilon2_bulk[l]/((epsilon1_bulk[l]+2*epsilonm)**2+(epsilon2_bulk[l])**2)

def Chi(observed, expected):
    return ((observed-expected)**2/expected).sum()

def Chi_R_rho(R, rho):
    return Chi(absorbance_r, Absorbance(l_r, R, epsilonm_fit, rho))

def Chi_R_epsilonm(R, epsilonm): # only accounts for the chi squared in the selected fit range
    return Chi(absorbance_r, Absorbance(l_r, R, epsilonm, rho_fit))

def Gans_prolate(e): # prolate dipolarization factor
    return (1-e**2)/e**2 * (1/(2*e)*np.log((1+e)/(1-e))-1)

def Gans_ec_prolate_f(l, R, epsilonm, f, L1): # extinction coefficient [nm**-1]
    c1 = epsilon2(l, R)/(L1**2* (epsilon1(l, R)+epsilonm*((1-L1)/L1))**2+epsilon2(l, R)**2)
    L2 = (1-L1)/2
    c2 = epsilon2(l, R)/(L2**2*(epsilon1(l, R)+epsilonm*((1-L2)/L2))**2+epsilon2(l, R)**2)
    return 1/3*omega(l)/c*epsilonm**(3/2)*f*(c1 + 2*c2)

def Gans_ec_prolate_f_L1_fixed(l, R, epsilonm, f):
    L1 = Gans_prolate(e_guess)
    return Gans_ec_prolate_f(l, R, epsilonm, f, L1)

def Gans_absorbance_prolate(l, R, epsilonm, f, L1):
    ec = Gans_ec_prolate_f(l, R, epsilonm, f, L1)
    return np.log10(np.e)*z*ec

def Gans_absorbance_prolate_L1_fixed(l, R, epsilonm, f):
    L1 = Gans_prolate(e_guess)
    return Gans_absorbance_prolate(l, R, epsilonm, f, L1)

def Gans_absorbance_prolate_f_fixed(l, R, epsilonm, e):
    L1 = Gans_prolate(e)
    return Gans_absorbance_prolate(l, R, epsilonm, f_fit, L1)

def Chi_Gans(R, epsilonm, f, L1):
    return Chi(absorbance_r, Gans_absorbance_prolate(l_r, R, epsilonm, f, L1))

def Chi_Gans_L1_fixed(R, epsilonm, f):
    return Chi(absorbance_r, Gans_absorbance_prolate_L1_fixed(l_r, R, epsilonm, f))

def multiwrite(outfile, string):
    outfile.write(string + "\n")
    print(string)
    
#%%
# FITTING: Johnson and Christy

par_fit_JC, par_cov_JC = curve_fit(Absorbance_JC, l, absorbance, p0=(epsilonm_guess, 2*10**-6))
Absorbance_JC_fitted = Absorbance_JC(l, epsilonm=par_fit_JC[0], f=par_fit_JC[1])

par_fit_JC_r, par_cov_JC_r = curve_fit(Absorbance_JC, l_r, absorbance_r, p0=(epsilonm_guess, 2*10**-6))
Absorbance_JC_fitted_r = Absorbance_JC(l, epsilonm=par_fit_JC_r[0], f=par_fit_JC_r[1])

with open("outputfile.txt", "w") as outfile:
    multiwrite(outfile, "Full curve JC fit:")
    multiwrite(outfile, "espilonm_JC = " + str(par_fit_JC[0]) + " with error " + str(np.sqrt(par_cov_JC[0,0])))
    multiwrite(outfile, "f_JC = " + str(par_fit_JC[1]) + " with error " + str(np.sqrt(par_cov_JC[1,1])))
    multiwrite(outfile, "Chi-squared = " + str(Chi(absorbance, Absorbance_JC_fitted)))
    multiwrite(outfile, "")
    
    multiwrite(outfile, "Restricting the JC fit to 470nm-570nm:")
    multiwrite(outfile, "espilonm_JC = " + str(par_fit_JC_r[0]) + " with error " + str(np.sqrt(par_cov_JC_r[0,0])))
    multiwrite(outfile, "f_JC = " + str(par_fit_JC_r[1]) + " with error " + str(np.sqrt(par_cov_JC_r[1,1])))
    multiwrite(outfile, "Chi-squared = " + str(Chi(absorbance, Absorbance_JC_fitted_r)))
    multiwrite(outfile, "")

#%%
# PLOTTING

epsilon1_2 = epsilon1(l, R=2)
epsilon1_5 = epsilon1(l, R=5)
epsilon1_10 = epsilon1(l, R=10)
epsilon2_2 = epsilon2(l, R=2)
epsilon2_5 = epsilon2(l, R=5)
epsilon2_10 = epsilon2(l, R=10)

# Bulk gold dielectric functions
plt.figure(figsize=(10, 6), dpi=100)
plt.plot(l_bulk, epsilon1_bulk, label=r"$\epsilon_1$")
plt.plot(l_bulk, epsilon2_bulk, label=r"$\epsilon_2$")
plt.plot(l, epsilon1_2, label=r"$\epsilon_1(R=2nm)$")
plt.plot(l, epsilon1_5, label=r"$\epsilon_1(R=5nm)$")
plt.plot(l, epsilon1_10, label=r"$\epsilon_1(R=10nm)$")
plt.plot(l, epsilon2_2, label=r"$\epsilon_2(R=2nm)$")
plt.plot(l, epsilon2_5, label=r"$\epsilon_2(R=5nm)$")
plt.plot(l, epsilon2_10, label=r"$\epsilon_2(R=10nm)$")
plt.title("Bulk gold dielectric functions", fontdict={"fontname": "Calibri", "fontsize": fs})
plt.xlabel(r"$\lambda$ (nm)", fontdict={"fontsize": fs})
plt.xticks(fontsize=fs)
plt.ylabel(r"$\epsilon$", fontdict={"fontsize": fs})
plt.yticks(fontsize=fs)
plt.legend(fontsize=fs//4*3, ncol=2)
plt.tight_layout()

# JC plots
plt.figure(figsize=(10, 6), dpi=100)
plt.plot(l, absorbance, color="green", label=r"Experimental data")
plt.plot(l, Absorbance_JC_fitted, color="blue", label=r"Johnson-Christy, full fit")
plt.plot(l, Absorbance_JC_fitted_r, color="red", label=r"Johnson-Christy, fit of the peak")
plt.title("Absorbance", fontdict={"fontname": "Calibri", "fontsize": fs+5})
plt.xlabel(r"$\lambda$ (nm)", fontdict={"fontsize": fs})
plt.xticks(fontsize=fs)
plt.ylabel("Absorbance", fontdict={"fontsize": fs})
plt.yticks(fontsize=fs)
plt.legend(fontsize=fs//4*3, ncol=1)
plt.tight_layout()

#%%
# FITTING: Size dependent, initial triple fit

# Initial 3 parameter fit: this can give the initial guess for the more refined fit later, but results are not totally realiable
# Indeed rho*R**3 is constant, so the dependency on R and rho is very weak and there probably are a lot of local minima
par_fit, par_cov = curve_fit(Absorbance, l_r, absorbance_r, p0=(R_guess, epsilonm_guess, rho_guess))
Absorbance_fitted = Absorbance(l, R=par_fit[0], epsilonm=par_fit[1], rho=par_fit[2])

with open("outputfile.txt", "a") as outfile:
    multiwrite(outfile, "Fit restricted to 470nm-570nm:")
    multiwrite(outfile, "R = " + str(par_fit[0]) + " nm, with error " + str(np.sqrt(par_cov[0,0])) + " nm")
    multiwrite(outfile, "espilonm = " + str(par_fit[1]) + " with error " + str(np.sqrt(par_cov[1,1])))
    multiwrite(outfile, "rho = " + str(par_fit[2]) + " nm**-3, with error " + str(np.sqrt(par_cov[2,2])) + " nm**-3")
    multiwrite(outfile, "Chi-squared = " + str(Chi(absorbance, Absorbance_fitted)))
    multiwrite(outfile, "")

#%%
# PLOTTING: Size dependent, initial triple fit

plt.figure(figsize=(10, 6), dpi=100)
plt.plot(l, absorbance, color="green", label="Experimental data")
plt.plot(l, Absorbance_fitted, color="blue", label="Fit as a function of R, $\epsilon_{m}$, ρ ")
plt.title("Absorbance (R, $\epsilon_{m}$, ρ ) vs $\lambda$ ", fontsize=fs+5)
plt.xlabel(r"$\lambda$ (nm)", fontdict={"fontsize": fs})
plt.xticks(fontsize=fs)
plt.ylabel("Absorbance", fontdict={"fontsize": fs})
plt.yticks(fontsize=fs)
plt.legend(fontsize=fs//4*3)
plt.tight_layout()

#%%
# FITTING: size dependent, double fits

# fixing epsilonm to our guess (assuming epsilonm=n^2 with n=1.333 knowing we are in water)
par_fit_epsilonmfix, par_cov_epsilonmfix = curve_fit(Absorbance_epsilonmfix, l_r, absorbance_r, p0=(R_guess, rho_guess))
Absorbance_fitted_epsilonmfix = Absorbance_epsilonmfix(l, R=par_fit_epsilonmfix[0], rho=par_fit_epsilonmfix[1])

with open("outputfile.txt", "a") as outfile:
    multiwrite(outfile, "Fit restricted to 470nm-570nm: R, rho; first try")
    multiwrite(outfile, "R = " + str(par_fit_epsilonmfix[0]) + " nm, with error " + str(np.sqrt(par_cov_epsilonmfix[0,0])) + " nm")
    multiwrite(outfile, "rho = " + str(par_fit_epsilonmfix[1]) + " nm**-3, with error " + str(np.sqrt(par_cov_epsilonmfix[1,1])) + " nm**-3")
    multiwrite(outfile, "Chi-squared = " + str(Chi(absorbance, Absorbance_fitted_epsilonmfix)))
    multiwrite(outfile, "")

# fixing rho to the one obtained previously
par_fit_rhofix, par_cov_rhofix = curve_fit(Absorbance_rhofix, l_r, absorbance_r, p0=(R_guess, epsilonm_guess))
Absorbance_fitted_rhofix = Absorbance_rhofix(l, R=par_fit_rhofix[0], epsilonm=par_fit_rhofix[1])

with open("outputfile.txt", "a") as outfile:
    multiwrite(outfile, "Fit restricted to 470nm-570nm: R, epsilonm")
    multiwrite(outfile, "R = " + str(par_fit_rhofix[0]) + " nm, with error " + str(np.sqrt(par_cov_rhofix[0,0])) + " nm")
    multiwrite(outfile, "epsilonm = " + str(par_fit_rhofix[1]) + " with error " + str(np.sqrt(par_cov_rhofix[1,1])))
    multiwrite(outfile, "Chi-squared = " + str(Chi(absorbance, Absorbance_fitted_rhofix)))
    multiwrite(outfile, "")

#fixing epsilonm to the previous value
par_fit_epsilonmfixfit, par_cov_epsilonmfixfit = curve_fit(Absorbance_epsilonmfixfit, l_r, absorbance_r, p0=(R_guess, rho_guess))
Absorbance_fitted_epsilonmfixfit = Absorbance_epsilonmfixfit(l, R=par_fit_epsilonmfixfit[0], rho=par_fit_epsilonmfixfit[1])

R_fit = par_fit_epsilonmfixfit[0]
rho_fit = par_fit_epsilonmfixfit[1]

with open("outputfile.txt", "a") as outfile:
    multiwrite(outfile, "Fit restricted to 470nm-570nm: R, rho; second try")
    multiwrite(outfile, "R = " + str(R_fit) + " nm, with error " + str(np.sqrt(par_cov_epsilonmfixfit[0,0])) + " nm")
    multiwrite(outfile, "rho = " + str(rho_fit) + " nm**-3, with error " + str(np.sqrt(par_cov_epsilonmfixfit[1,1])) + " nm**-3")
    multiwrite(outfile, "Chi-squared = " + str(Chi(absorbance, Absorbance_fitted_epsilonmfixfit)))
    multiwrite(outfile, "")
    

# using f as a parameter
par_fit_f_epsilonmfix, par_cov_f_epsilonmfix = curve_fit(Absorbance_f_epsilonmfix, l_r, absorbance_r, p0=(R_guess, f_guess))
Absorbance_fitted_f_epsilonmfix = Absorbance_f_epsilonmfix(l, R=par_fit_f_epsilonmfix[0], f=par_fit_f_epsilonmfix[1])

with open("outputfile.txt", "a") as outfile:
    multiwrite(outfile, "Fit restricted to 470nm-570nm: R, f")
    multiwrite(outfile, "R = " + str(par_fit_f_epsilonmfix[0]) + " nm, with error " + str(np.sqrt(par_cov_f_epsilonmfix[0,0])) + " nm")
    multiwrite(outfile, "f = " + str(par_fit_f_epsilonmfix[1]) + " with error " + str(np.sqrt(par_cov_f_epsilonmfix[1,1])))
    multiwrite(outfile, "Chi-squared = " + str(Chi(absorbance, Absorbance_fitted_f_epsilonmfix)))
    multiwrite(outfile, "")

#%%
# FITTING: Chi squared maps

R_domain_1 = np.arange(2, 15, 0.1)
rho_domain = np.arange(1*10**-10, 15*10**-9, 10**-10)
epsilonm_domain = np.arange(1.4, 3, 0.01)
R_domain_2 = np.arange(2, 7, 0.1)

Chi_R_rho_values = np.zeros((len(R_domain_1), len(rho_domain)))    
for i in range(len(R_domain_1)):
    for j in range(len(rho_domain)):
        Chi_R_rho_values[i, j] = Chi_R_rho(R_domain_1[i], rho_domain[j])
        
fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
contour_plot = plt.contourf(R_domain_1, rho_domain, Chi_R_rho_values.transpose(),
                            np.logspace(np.log10(Chi_R_rho_values.min()), np.log10(Chi_R_rho_values.max()), 20), locator=ticker.LogLocator(), cmap="plasma")
colbar = plt.colorbar()
colbar.set_label(label=r"$\chi^2$",size=fs)
colbar.ax.tick_params(labelsize=fs)
plt.title(r"$\chi^2$ map of absorbance: (R, ρ)", fontsize=fs+5)
plt.xlabel("R (nm)", fontdict={"fontsize": fs})
plt.xticks(fontsize=fs)
plt.ylabel(r"$\rho \ (nm^{-3})$", fontdict={"fontsize": fs})
plt.yticks(fontsize=fs)
t = ax.yaxis.get_offset_text()
t.set_size(fs)
plt.tight_layout()

argmin_R_rho = np.argwhere(Chi_R_rho_values == Chi_R_rho_values.min())[0]
R_min = R_domain_1[argmin_R_rho[0]]
rho_min = rho_domain[argmin_R_rho[1]]


Chi_R_epsilonm_values = np.zeros((len(R_domain_2), len(epsilonm_domain)))
for i in range(len(R_domain_2)):
    for j in range(len(epsilonm_domain)):
        Chi_R_epsilonm_values[i, j] = Chi_R_epsilonm(R_domain_2[i], epsilonm_domain[j])

fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
contour_plot = plt.contourf(R_domain_2, epsilonm_domain, Chi_R_epsilonm_values.transpose(),
                            np.logspace(np.log10(Chi_R_epsilonm_values.min()), np.log10(Chi_R_epsilonm_values.max()), 30),
                            locator=ticker.LogLocator(), cmap="plasma")
colbar = plt.colorbar()
colbar.set_label(label=r"$\chi^2$",size=fs)
colbar.ax.tick_params(labelsize=fs)
plt.title(r"$\chi^2$ map of absorbance: (R, $\epsilon_m$)", fontsize=fs+5)
plt.xlabel("R (nm)", fontdict={"fontsize": fs})
plt.xticks(fontsize=fs)
plt.ylabel(r"$\epsilon_m$", fontdict={"fontsize": fs})
plt.yticks(fontsize=fs)
t = ax.yaxis.get_offset_text()
t.set_size(fs)
plt.tight_layout()

argmin_R_epsilonm = np.argwhere(Chi_R_epsilonm_values == Chi_R_epsilonm_values.min())[0]
R_min_2 = R_domain_2[argmin_R_epsilonm[0]]
epsilonm_min = epsilonm_domain[argmin_R_epsilonm[1]]

with open("outputfile.txt", "a") as outfile:
    multiwrite(outfile, "Minimum of chi squared (coarse sample)")
    multiwrite(outfile, "fit1, R = " + str(R_min) + " nm")
    multiwrite(outfile, "fit1, rho = " + str(rho_min) + " nm**-3")
    multiwrite(outfile, "fit2, R = " + str(R_min_2) + " nm")
    multiwrite(outfile, "fit2, epsilonm = " + str(epsilonm_min))
    multiwrite(outfile, "")
#%%
# PLOTTING: size dependent, double fits

plt.figure(figsize=(10, 6), dpi=100)
plt.plot(l, absorbance, color="green", label="Experimental data")
plt.plot(l, Absorbance_fitted_epsilonmfix, color="blue", label="Fit as a function of R, ρ ")
plt.title("Absorbance vs $\lambda$: (R, ρ) fit fixing $\epsilon_{m}$ = 2.3", fontsize=fs+5)
plt.xlabel(r"$\lambda$ (nm)", fontdict={"fontsize": fs})
plt.xticks(fontsize=fs)
plt.ylabel("Absorbance", fontdict={"fontsize": fs})
plt.yticks(fontsize=fs)
plt.legend(fontsize=fs//4*3)
plt.tight_layout()

plt.figure(figsize=(10, 6), dpi=100)
plt.plot(l, absorbance, color="green", label="Experimental data")
plt.plot(l, Absorbance_fitted_rhofix, color="blue", label="Fit as a function of R, $\epsilon_{m}$ ")
plt.title("Absorbance vs $\lambda$: (R, $\epsilon_{m}$) fit fixing ρ = 1.1*$10^{-8} \ nm^{-3}$", fontsize=fs+5)
plt.xlabel(r"$\lambda$ (nm)", fontdict={"fontsize": fs})
plt.xticks(fontsize=fs)
plt.ylabel("Absorbance", fontdict={"fontsize": fs})
plt.yticks(fontsize=fs)
plt.legend(fontsize=fs//4*3)
plt.tight_layout()

plt.figure(figsize=(10, 6), dpi=100)
plt.plot(l, absorbance, color="green", label="Experimental data")
plt.plot(l, Absorbance_fitted_epsilonmfixfit, color="blue", label="Fit as a function of R, ρ ")
plt.title("Absorbance vs $\lambda$: (R, ρ) fit fixing $\epsilon_{m}$ = 2.1", fontsize=fs+5)
plt.xlabel(r"$\lambda$ (nm)", fontdict={"fontsize": fs})
plt.xticks(fontsize=fs)
plt.ylabel("Absorbance", fontdict={"fontsize": fs})
plt.yticks(fontsize=fs)
plt.legend(fontsize=fs//4*3)
plt.tight_layout()

plt.figure(figsize=(10, 6), dpi=100)
plt.plot(l, absorbance, color="green", label="Experimental data")
plt.plot(l, Absorbance_fitted_f_epsilonmfix, color="blue", label="Fit as a function of R, f ")
plt.title("Absorbance vs $\lambda$: (R, f) fit fixing $\epsilon_{m}$ = 2.1", fontsize=fs+5)
plt.xlabel(r"$\lambda$ (nm)", fontdict={"fontsize": fs})
plt.xticks(fontsize=fs)
plt.ylabel("Absorbance", fontdict={"fontsize": fs})
plt.yticks(fontsize=fs)
plt.legend(fontsize=fs//4*3)
plt.tight_layout()

#%%
# GANS THEORY
f_fit = rho_fit * 4/3 * np.pi * R_fit**3

bounds_ = ([0, 1, 1*10**(-7), 0], [np.inf, np.inf, 1*10**(-5), 1])
par_fit_Gans_, par_cov_Gans_ = curve_fit(Gans_absorbance_prolate,
    l_r, absorbance_r, p0=(R_fit, epsilonm_fit, f_fit, 1/3), bounds = bounds_, maxfev=5000)
Absorbance_fitted_Gans_ = Gans_absorbance_prolate(l,
    R=par_fit_Gans_[0], epsilonm=par_fit_Gans_[1], f=par_fit_Gans_[2], L1=par_fit_Gans_[3])

with open("outputfile.txt", "a") as outfile:
    multiwrite(outfile, "Gans fit restricted to 470nm-570nm:")
    multiwrite(outfile, "R = " + str(par_fit_Gans_[0]) + " nm, with error " + str(np.sqrt(par_cov_Gans_[0,0])) + " nm")
    multiwrite(outfile, "epsilonm = " + str(par_fit_Gans_[1]) + "with error " + str(np.sqrt(par_cov_Gans_[1,1])))
    multiwrite(outfile, "f = " + str(par_fit_Gans_[2]) + "with error " + str(np.sqrt(par_cov_Gans_[2,2])))
    multiwrite(outfile, "e = " + str(par_fit_Gans_[3]) + "with error " + str(np.sqrt(par_cov_Gans_[3,3])))
    multiwrite(outfile, "Chi-squared = " + str(Chi(absorbance, Absorbance_fitted_Gans_)))
    multiwrite(outfile, "")
    
plt.figure(figsize=(10, 6), dpi=100)
plt.plot(l, absorbance, color="green", label="Experimental data")
plt.plot(l, Absorbance_fitted_Gans_, color="blue", label="Gans theory fit")
plt.title("Absorbance vs $\lambda$: Gans theory", fontsize=fs+5)
plt.xlabel(r"$\lambda$ (nm)", fontdict={"fontsize": fs})
plt.xticks(fontsize=fs)
plt.ylabel("Absorbance", fontdict={"fontsize": fs})
plt.yticks(fontsize=fs)
plt.legend(fontsize=fs//4*3)
plt.tight_layout()

# Manual fit by definition and optimization of a proper chi squared

R_domain_Gans = np.arange(2, 15, 0.2)
epsilonm_domain_Gans = np.arange(1.4, 3, 0.05)
f_domain_Gans = np.arange(1*10**-7, 5*10**-6, 5*10**-7)

Chi_Gans_values = np.zeros((len(R_domain_Gans), len(epsilonm_domain_Gans),
                     len(f_domain_Gans)))    
for i in range(len(R_domain_Gans)):
    for j in range(len(epsilonm_domain_Gans)):
        for k in range(len(f_domain_Gans)):
                Chi_Gans_values[i, j, k] = Chi_Gans_L1_fixed(R_domain_Gans[i],
                                                       epsilonm_domain_Gans[j], f_domain_Gans[k])


argmin_Gans = np.argwhere(Chi_Gans_values == Chi_Gans_values.min())[0]
R_Gans = R_domain_Gans[argmin_Gans[0]]
epsilonm_Gans = epsilonm_domain_Gans[argmin_Gans[1]]
f_Gans = f_domain_Gans[argmin_Gans[2]]

# It turns out that the results are the same as with the simple fit
# On one side this is reassuring, on the other results remain to be bad

#%%
'''
# 4 dimensional chi squared map: being 4d, to obtain a decent resolution a lot of time is needed
Chi_Gans_values = np.zeros((len(R_domain_Gans), len(epsilonm_domain_Gans),
                     len(f_domain_Gans), len(L1_domain_Gans)))    
for i in range(len(R_domain_Gans)):
    for j in range(len(epsilonm_domain_Gans)):
        for k in range(len(f_domain_Gans)):
            for h in range(len(L1_domain_Gans)):
                Chi_Gans_values[i, j, k ,h] = Chi_Gans(R_domain_Gans[i], epsilonm_domain_Gans[j],
                                                       f_domain_Gans[k], L1_domain_Gans[h])


argmin_Gans = np.argwhere(Chi_Gans_values == Chi_Gans_values.min())[0]
R_Gans = R_domain_Gans[argmin_Gans[0]]
epsilonm_Gans = epsilonm_domain_Gans[argmin_Gans[1]]
f_Gans = f_domain_Gans[argmin_Gans[2]]
L1_Gans = L1_domain_Gans[argmin_Gans[3]]

# Attempts at fixing some of the variables; does not help
# The problem is that the curve is very well fitted by bulk values for the dielectric function
# Thus we can not infer the value of R, which is the one we're most interested in
# The only saving grace is that the ellipticity values are extremely small
# So we might say that Gans theory is not necessary and keep using our previous results

bounds = ([0, 1, 1*10**(-7)], [20, 3, 1*10**(-5)])
par_fit_Gans, par_cov_Gans = curve_fit(Gans_absorbance_prolate_L1_fixed,
    l_r, absorbance_r, p0=(R_fit, epsilonm_fit, f_fit), bounds = bounds, maxfev=5000)
Absorbance_fitted_Gans = Gans_absorbance_prolate_L1_fixed(l,
    R=par_fit_Gans[0], epsilonm=par_fit_Gans[1], f=par_fit_Gans[2])

with open("outputfile.txt", "a") as outfile:
    multiwrite(outfile, "Gans fit restricted to 470nm-570nm, e fixed:")
    multiwrite(outfile, "R = " + str(par_fit_Gans[0]) + " nm, with error " + str(np.sqrt(par_cov_Gans[0,0])) + " nm")
    multiwrite(outfile, "epsilonm = " + str(par_fit_Gans[1]) + "with error " + str(np.sqrt(par_cov_Gans[1,1])))
    multiwrite(outfile, "f = " + str(par_fit_Gans[2]) + "with error " + str(np.sqrt(par_cov_Gans[2,2])))
    multiwrite(outfile, "Chi-squared = " + str(Chi(absorbance, Absorbance_fitted_Gans)))
    multiwrite(outfile, "")
    
plt.figure(figsize=(10, 6), dpi=100)
plt.plot(l, absorbance, color="green", label="Experimental data")
plt.plot(l, Absorbance_fitted_Gans, color="blue", label="Gans theory fit")
plt.title("Absorbance vs $\lambda$: Gans theory, e fixed", fontsize=fs+5)
plt.xlabel(r"$\lambda$ (nm)", fontdict={"fontsize": fs})
plt.xticks(fontsize=fs)
plt.ylabel("Absorbance", fontdict={"fontsize": fs})
plt.yticks(fontsize=fs)
plt.legend(fontsize=fs//4*3)
plt.tight_layout()


bounds_f = ([0, 1, 0], [20, 3, 1])
par_fit_Gans_f, par_cov_Gans_f = curve_fit(Gans_absorbance_prolate_f_fixed,
    l_r, absorbance_r, p0=(R_fit, epsilonm_fit, e_guess), bounds = bounds_f, maxfev=5000)
Absorbance_fitted_Gans_f = Gans_absorbance_prolate_f_fixed(l,
    R=par_fit_Gans_f[0], epsilonm=par_fit_Gans_f[1], e=par_fit_Gans_f[2])

with open("outputfile.txt", "a") as outfile:
    multiwrite(outfile, "Gans fit restricted to 470nm-570nm, f fixed:")
    multiwrite(outfile, "R = " + str(par_fit_Gans_f[0]) + " nm, with error " + str(np.sqrt(par_cov_Gans_f[0,0])) + " nm")
    multiwrite(outfile, "epsilonm = " + str(par_fit_Gans_f[1]) + "with error " + str(np.sqrt(par_cov_Gans_f[1,1])))
    multiwrite(outfile, "e = " + str(par_fit_Gans_f[2]) + "with error " + str(np.sqrt(par_cov_Gans_f[2,2])))
    multiwrite(outfile, "Chi-squared = " + str(Chi(absorbance, Absorbance_fitted_Gans_f)))
    multiwrite(outfile, "")
    
plt.figure(figsize=(10, 6), dpi=100)
plt.plot(l, absorbance, color="green", label="Experimental data")
plt.plot(l, Absorbance_fitted_Gans_f, color="blue", label="Gans theory fit")
plt.title("Absorbance vs $\lambda$: Gans theory, f fixed", fontsize=fs+5)
plt.xlabel(r"$\lambda$ (nm)", fontdict={"fontsize": fs})
plt.xticks(fontsize=fs)
plt.ylabel("Absorbance", fontdict={"fontsize": fs})
plt.yticks(fontsize=fs)
plt.legend(fontsize=fs//4*3)
plt.tight_layout()
'''

#%%

# TODO: understand why this gives me R = 6.6nm

def cross_section_rho(x,a,b,bool_lim=False):  # a = R, b=rho
    omega = 2 * np.pi * c / x
    epsm = 1.33**2#1.43*1.43 #1.33*1.33         
    eps1 = epsilon1(l,a)
    eps2 = epsilon2(l,a)
    const1  = 1/3 * 4/3 * np.pi * 1/c
    const2 = np.log10(np.e) * np.power(10.,7) * np.power(1/10,9)
    a1 = 23 # Major axis
    a2 = 20  # Minor axis
    ecc = np.sqrt( 1 - (a2/a1)**2 )
    L1  = (1-ecc**2)/ecc**2 * ( 1/(2*ecc) * np.log((1+ecc)/(1-ecc)) - 1)
    L2  = (1-L1)/2
    summation = eps2/L1**2 / ( ( eps1 + epsm * (1-L1)/L1)**2 + eps2**2)   +  2 * eps2/L2**2 / ( ( eps1 + epsm * (1-L2)/L2)**2 + eps2**2)
    sigma = const1 * const2 * b * omega * np.power(epsm,3/2) * np.power(a,3) * summation
    return sigma

R_list = np.linspace(3, 20, num=200)
rho_list = np.linspace(1.,2., num=100)
x = l
y = absorbance
error = 1/y
chi_list = [] 
a_list   = []
b_list   = []

for i in range(len(R_list)):
    for j in range(len(rho_list)):
        chisq = np.sum( np.power( y - cross_section_rho(x,R_list[i],rho_list[j],bool_lim=True), 2)  /(np.power(error,2)) )
        
        chi_list.append(chisq)
        a_list.append(R_list[i])
        b_list.append(rho_list[j])

min_chi       =  min(chi_list)
index_min_chi = np.argmin(chi_list)
a             = a_list[index_min_chi]
b             = b_list[index_min_chi]

print('min chi_square:', min_chi)
print('index min chi_square index:', index_min_chi)
print('a=',a)
print('b=',b)
