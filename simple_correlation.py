import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.optimize import curve_fit
from scipy.stats.stats import pearsonr


plt.rcParams.update({'font.size': 14})


def lin_func(x, k):
    return k * x


def quad_func(x, k):
    return k * x ** 2


def power_func(x, k, ind):
    return k * x ** ind


def chisq(y_vals, y_expected, y_errs, num_params=1):
    #returns chi2, dof, red_chi2
    #  for reduced chisq test, under the assumption of Poisson counting
    #  we have lnL = const - (1/2.)*chi2
    if y_vals.shape[0] != y_expected.shape[0]:
        print("Inconsistent input sizes")
        return
    #z = (y_vals[i] - y_expected[i]) / y_errs[i]
    z = (y_vals - y_expected) / y_errs
    chi2 = np.sum(z ** 2)
    chi2dof = chi2 / (y_vals.shape[0] - num_params)
    return chi2, (y_vals.shape[0] - num_params), chi2dof


def fit_factory_boundMC(model, x, y, dy=None, p0=[1.2e-11, -3.6], Nsim=500, plotEs=None):
    if dy is not None:
        paramsPLlog, covariacesPLlog = curve_fit(model, x,
                                                 y, p0=p0,
                                                 sigma=dy,
                                                 absolute_sigma=True)  # approx abs
    else:
        paramsPLlog, covariacesPLlog = curve_fit(model, x,
                                                 y, p0=p0)

    for i in range(len(p0)):
        print("best-fit param %d = %.4g +/- %.4g" % (i, paramsPLlog[i], np.sqrt(abs(covariacesPLlog[i, i]))))

    chi2LC, dofLC, redchi2LC = chisq(y,
                                     model(x, *paramsPLlog),
                                     dy,
                                     len(p0))
    lnL_LC = -0.5 * chi2LC
    print("fit Chisq is %.3f, dof is %d, reduced Chisq is %.2f" % (chi2LC, dofLC, redchi2LC))
    print("Log likelihood lnL={0}".format(lnL_LC))

    # Use some MC
    if plotEs is not None:
        x_values = plotEs
    else:
        x_values = np.logspace(np.log10(float(x[0])), np.log10(float(x[-1])), 100)
    rng = np.random.RandomState(seed=76)
    parameter_samples = rng.multivariate_normal(paramsPLlog, covariacesPLlog, Nsim)

    realizations = np.array([model(x_values, *pars) for pars in parameter_samples])

    qlo = 100 * scipy.stats.norm.cdf(-1)  # 1 is the 1 sigma
    y_low = np.percentile(realizations, qlo, axis=0)
    qhi = 100 * scipy.stats.norm.cdf(1)  # 1 is the 1 sigma
    y_high = np.percentile(realizations, qhi, axis=0)

    # print(y_low.shape)

    return paramsPLlog, np.diag(covariacesPLlog) ** 0.5, chi2LC, dofLC, redchi2LC, x_values, y_low, y_high


def add_CL(x, lin_func, fit, cov_, quad_func, fit_quad, cov_quad, ax):

    ax.fill_between(x, lin_func(x, fit[0] - np.sqrt(cov_[0, 0])),
                    lin_func(x, fit[0] + np.sqrt(cov_[0, 0])), facecolor='k', alpha=0.1)

    ax.fill_between(x, quad_func(x, fit_quad[0] - np.sqrt(cov_quad[0, 0])),
                    quad_func(x, fit_quad[0] + np.sqrt(cov_quad[0, 0])), facecolor='k', alpha=0.1)


def toyMC(x, dxp, dxn, y, dy, n=100):
    assert len(x) == len(y)
    N = len(x)
    rng = np.random.RandomState(seed=76)
    rdata =  pearsonr(x, y)[0]
    rs = np.zeros(n)
    for isim in range(n):
        xs_ = rng.normal(size=N)
        samplex = np.zeros_like(xs_)
        for i, x_ in enumerate(xs_):
            if x_>=0:
                samplex[i] = x_*dxp[i]+x[i]
            else:
                samplex[i] = x_*dxn[i]+x[i]
        sampley = rng.normal(size=N)*dy+y
        rs[isim] = pearsonr(samplex, sampley)[0]
    qlo = 100 * scipy.stats.norm.cdf(-1)    #1 is the 1 sigma
    rlow = np.percentile(rs, qlo, axis=0)
    qhi = 100 * scipy.stats.norm.cdf(1)     #1 is the 1 sigma
    rhi = np.percentile(rs, qhi, axis=0)
    print("pearson r 1-sigma CL is {:.2f} -- {:.2f}".format(rlow, rhi))
    print("pearson r is {:.2f} + {:.2f} - {:.2f}".format(rdata, rhi-rdata, rdata-rlow))

    return rdata, rs, rlow, rhi



if __name__ == "__main__":
    # some data
    xdata = np.array([0.78873, 0.65666, 2.4601 , 2.3517 , 2.178  , 1.4122 , 1.9267 ,
           2.9907 , 1.0286 , 1.356  , 1.7321 , 1.4385 , 1.7409 , 0.94367,
           0.79195, 1.9959 ])
    ydata = np.array([0.10094804, 0.24631636, 0.323214  , 0.333583  , 0.540316  ,
           0.370015  , 0.378274  , 0.876731  , 0.0958182 , 0.12149   ,
           0.23814   , 0.352008  , 0.344041  , 0.366126  , 0.14196   ,
           0.181861  ])
    xerr_pos = np.array([0.04547, 0.03334, 0.0939 , 0.1403 , 0.067  , 0.0598 , 0.0763 ,
           0.1093 , 0.0864 , 0.117  , 0.0599 , 0.3925 , 0.0991 , 0.07233,
           0.05655, 0.0951 ])
    xerr_neg = np.array([0.05423, 0.03426, 0.1111 , 0.1147 , 0.066  , 0.0682 , 0.0787 ,
           0.0877 , 0.0883 , 0.072  , 0.0861 , 0.1875 , 0.0739 , 0.04937,
           0.04885, 0.0769 ])
    yerr = np.array([0.04870452, 0.0513109 , 0.0329474 , 0.046216  , 0.0463591 ,
           0.0417949 , 0.0416079 , 0.0405101 , 0.0446851 , 0.0469011 ,
           0.0519923 , 0.097362  , 0.0536178 , 0.0553851 , 0.083871  ,
           0.0455736 ])

    # plotting correlation
    fig, ax = plt.subplots()
    ax.errorbar(xdata, ydata,
                xerr=[xerr_neg, xerr_pos], yerr=yerr,
                fmt='o', color='k', ecolor='k', mec='k', ms=5, capthick=0, ls='')

    # fit linear
    fit, cov_ = curve_fit(lin_func, xdata, ydata, p0=[1.0],
                          sigma=yerr, absolute_sigma=True)
    # fit quadratic
    fit_quad, cov_quad = curve_fit(quad_func, xdata, ydata, p0=[1.0],
                                   sigma=yerr, absolute_sigma=True)
    # fit power law
    fit_p, cov_p = curve_fit(power_func, xdata, ydata, p0=[1.0, 1.5],
                             sigma=yerr, absolute_sigma=True)

    # check covariance matrices
    print(fit, np.diag(cov_) ** 0.5)
    print(fit_quad, np.diag(cov_quad) ** 0.5)
    print(fit_p, np.diag(cov_p) ** 0.5)

    # chi-square
    chi2_lin, dof_lin, redchi2_lin = chisq(ydata,
                                           lin_func(xdata, fit[0]),
                                           yerr,
                                           1)
    print("Linear fit Chisq is %.3f, dof is %d, reduced Chisq is %.2f" % (chi2_lin, dof_lin, redchi2_lin))

    chi2_quad, dof_quad, redchi2_quad = chisq(ydata,
                                              quad_func(xdata, fit_quad[0]),
                                              yerr,
                                              1)
    print("Quadratic fit Chisq is %.3f, dof is %d, reduced Chisq is %.2f" % (chi2_quad, dof_quad, redchi2_quad))

    chi2_p, dof_p, redchi2_p = chisq(ydata,
                                     power_func(xdata, *fit_p),
                                     yerr,
                                     1)
    print("Power law fit Chisq is %.3f, dof is %d, reduced Chisq is %.2f" % (chi2_p, dof_p, redchi2_p))

    x = np.arange(0, 3.2, 0.01)

    ax.plot(x, lin_func(x, fit[0]), '--k', alpha=0.3)

    ax.plot(x, quad_func(x, fit_quad[0]), ':k', alpha=0.3)

    # ax.plot(np.arange(0,3,0.01), power_func(np.arange(0,3,0.01), *fit_p), '--b')


    ax.set_ylabel(r'Flux > 200 GeV (10$^{-6}$ ph m$^{-2}$ s$^{-1}$)', color='k')
    ax.set_xlabel(r'Energy flux 0.3 -- 10 keV (10$^{-11}$ erg cm$^{-2}$ s$^{-1}$)', color='k')


    add_CL(x, lin_func, fit, cov_, quad_func, fit_quad, cov_quad, ax)

    fit_p, cov_p = curve_fit(power_func, xdata, ydata, p0=[1.0, 1.5])

    ps_, dps_, chi2_, dof_, redchi2_, es_, ylo_, yhi_ = fit_factory_boundMC(power_func,
                                                                            xdata,
                                                                            ydata,
                                                                            dy=np.array(yerr),
                                                                            p0=[1.0, 1.5],
                                                                            plotEs=x)
    model_ = power_func(x, *ps_)
    ax.plot(x, model_, color='b', ls='-')
    ax.fill_between(x, ylo_, yhi_, alpha=0.3, color='b')

    plt.tight_layout()
    # plt.savefig("XV_flux_corr_NHtotal.pdf")

    # plotting toy MC pearson r
    fig, ax = plt.subplots()

    rdata, rs, rlow, rhi = toyMC(xdata, xerr_pos, xerr_neg, ydata,yerr)
    _ = plt.hist(rs, bins=25)
    plt.axvline(rlow, color='r', ls='--')
    plt.axvline(rhi, color='r', ls='--')
    plt.axvline(rdata, color='r', ls='-',
                label="pearson r is \n{:.2f} + {:.2f} - {:.2f}".format(rdata, rhi - rdata, rdata - rlow))
    plt.xlabel(r"Pearson $r$")
    plt.legend(frameon=False)
    plt.tight_layout()
    #plt.savefig("VERJ0521_pearson_MC.png")

    plt.show()
    # df_x_v=df_x_v1.copy()





