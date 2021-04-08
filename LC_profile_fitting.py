import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from astropy.time import Time


import seaborn as sns
try:
    sns.set_style("ticks")
    sns.set_style({"xtick.direction": "in" ,"ytick.direction": "in"})
except:
    print("sns problem")

# for MCMC
import emcee
# for posterior estimatation
import scipy.stats as stats
from sklearn.neighbors.kde import KernelDensity


plt.rcParams.update({'font.size': 14})


# defining some functions for fitting

def flare_func(ts, amp, tpeak, trise, tdecay):
    rise_slice = np.where(ts<=tpeak)
    decay_slice = np.where(ts>tpeak)
    fs = np.zeros_like(ts).astype('float')
    fs[rise_slice] = amp*np.exp((ts[rise_slice]-tpeak)/trise)
    fs[decay_slice] = amp*np.exp(-(ts[decay_slice]-tpeak)/tdecay)
    return fs


def flare_func_two_peak(ts, amp1, tpeak1, trise1, tdecay1, amp2, tpeak2, trise2, tdecay2):
    rise_slice1 = np.where(ts<=tpeak1)
    decay_slice1 = np.where(ts>tpeak1)
    rise_slice2 = np.where(ts<=tpeak2)
    decay_slice2 = np.where(ts>tpeak2)

    fs = np.zeros_like(ts).astype('float')
    fs[rise_slice1] = amp1*np.exp((ts[rise_slice1]-tpeak1)/trise1)
    fs[decay_slice1] = amp1*np.exp(-(ts[decay_slice1]-tpeak1)/tdecay1)
    fs[rise_slice2] += amp2*np.exp((ts[rise_slice2]-tpeak2)/trise2)
    fs[decay_slice2] += amp2*np.exp(-(ts[decay_slice2]-tpeak2)/tdecay2)

    return fs


def flare_func_log(ts, amp, tpeak, trise, tdecay):
    rise_slice = np.where(ts<=tpeak)
    decay_slice = np.where(ts>tpeak)
    fs = np.zeros_like(ts).astype('float')
    fs[rise_slice] = amp*np.exp((ts[rise_slice]-tpeak)/trise)
    fs[decay_slice] = amp*np.exp(-(ts[decay_slice]-tpeak)/tdecay)
    return np.log10(fs)


def flare_func_w_const(ts, amp, tpeak, trise, tdecay, const):
    rise_slice = np.where(ts<=tpeak)
    decay_slice = np.where(ts>tpeak)
    fs = np.zeros_like(ts).astype('float')
    fs[rise_slice] = const+amp*np.exp((ts[rise_slice]-tpeak)/trise)
    fs[decay_slice] = const+amp*np.exp(-(ts[decay_slice]-tpeak)/tdecay)
    return fs

def flare_func_w_const_log(ts, amp, tpeak, trise, tdecay, const):
    rise_slice = np.where(ts<=tpeak)
    decay_slice = np.where(ts>tpeak)
    fs = np.zeros_like(ts).astype('float')
    fs[rise_slice] = const+amp*np.exp((ts[rise_slice]-tpeak)/trise)
    fs[decay_slice] = const+amp*np.exp(-(ts[decay_slice]-tpeak)/tdecay)
    return np.log10(fs)


def flare_func_w_const_higher_order(ts, amp, tpeak, trise, tdecay, const, k):
    rise_slice = np.where(ts<=tpeak)
    decay_slice = np.where(ts>tpeak)
    fs = np.zeros_like(ts).astype('float')
    fs[rise_slice] = const+amp*np.exp(-(abs(ts[rise_slice]-tpeak)/trise)**k)
    fs[decay_slice] = const+amp*np.exp(-(abs(ts[decay_slice]-tpeak)/tdecay)**k)
    return fs


def flare_func_w_const_higher_order_log(ts, amp, tpeak, trise, tdecay, const, k):
    rise_slice = np.where(ts<=tpeak)
    decay_slice = np.where(ts>tpeak)
    fs = np.zeros_like(ts).astype('float')
    fs[rise_slice] = const+amp*np.exp(-(abs(ts[rise_slice]-tpeak)/trise)**k)
    fs[decay_slice] = const+amp*np.exp(-(abs(ts[decay_slice]-tpeak)/tdecay)**k)
    return np.log10(fs)


# chi-square for fit quality estimation
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


# for intra-night light curves, convert MJD into minutes elapsed on that MJD
def min_of_mjd(t, mjd0=None):
    #t=sorted(t)
    t=np.sort(t)
    if mjd0 is None:
        mjd0 = t[0]
    return (t-mjd0)*24.*60


#ML fitting
from scipy import stats
from scipy import optimize



#log likelihood

def log_prior(params):
    if len(params)==4 and params[0]>0 and params[0] < 20 and params[1]>0 and params[1] < 200.0 \
        and params[2]> 0 and params[2] < 200.0 and params[3]>0 and params[3] < 100.0:
        return 0.0
        #return 200 ** -len(params)
    elif len(params)==5 and params[0]>0 and params[0] < 20 and params[1]>0 and params[1] < 200.0 \
        and params[2]> 0 and params[2] < 200.0 and params[3]>0 and params[3] < 100.0 and params[4]>0 and params[4] < 10.0:
        return 0.0
        #return 200 ** -len(params)
    return -np.inf

def lnL_poisson(params, model=flare_func, data=None):
    #  for reduced chisq test, under the assumption of Poisson counting
    #  we have lnL = const - (1/2.)*chi2
    x, y, dy = data
    y_fit = model(x, *params)
    z = (y - y_fit) / dy
    chi2 = np.sum(z ** 2)
    return -0.5*chi2

def logL(params, model=flare_func, data=None):
    """Gaussian log-likelihood of the model at params"""
    x, y, dy = data
    y_fit = model(x, *params)
    #return sum(stats.norm.logpdf(*args)
    #           for args in zip(y, y_fit, dy))
    return -0.5 * np.sum(np.log(2 * np.pi * dy ** 2)
                         + (y - y_fit) ** 2 / dy ** 2)

def log_posterior(params, model=flare_func, data=None, loglikelihood=lnL_poisson):
    params = np.asarray(params)
    return log_prior(params) + loglikelihood(params, model, data)

#optimizing on lnL:
def ML_optimize(model=flare_func, data=None, p0=[5., 140., 80., 30.], loglikelihood=logL):
    #p0 = nparams * [0]
    neg_logL = lambda params: -loglikelihood(params, model, data)
    return optimize.fmin_bfgs(neg_logL, p0, disp=1)


# perform MCMC:
#     note that data is a 2d array with t, f, and df;
#     e.g., data=np.vstack((min_of_mjd(df_lc.t.values, mjd0=57666.165), df_lc.f.values/plot_f_unit, df_lc.df.values/plot_f_unit))
def compute_mcmc(model=flare_func, data=None, p0=[5., 140., 80., 30.],
                 log_posterior=log_posterior, loglikelihood=logL,
                 nwalkers=50, nburn=1000, nsteps=2000, a=2., verbose=True):
    ndim = len(p0)  # this determines the model
    rng = np.random.RandomState(0)
    # starting_guesses = np.abs(rng.randn(nwalkers, ndim))
    if ndim == 2:
        starting_guesses = np.vstack((np.random.rand(nwalkers) * 2.e-5, np.random.rand(nwalkers) * (-6.0))).T
    elif ndim == 4:
        starting_guesses = np.vstack((np.random.rand(nwalkers) * 2.e-4, np.random.rand(nwalkers) * 10.,
                                      np.random.rand(nwalkers) * 10., np.random.rand(nwalkers) * 20.)).T
    elif ndim == 5:
        starting_guesses = np.vstack((np.random.rand(nwalkers) * 2.e-4, np.random.rand(nwalkers) * 10.,
                                      np.random.rand(nwalkers) * 10., np.random.rand(nwalkers) * 20.,
                                      np.random.rand(nwalkers) * 2.e-4,)).T

        # starting_guesses = emcee.utils.sample_ball(p0, np.sqrt(np.abs(np.asarray(p0))), nwalkers)
        # print(starting_guesses)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[model, data, loglikelihood])
    sampler.a = a
    sampler.run_mcmc(starting_guesses, nsteps)
    trace = sampler.chain[:, nburn:, :].reshape(-1, ndim)
    if verbose:
        print("Mean acceptance fraction: {0:.3g}".format(np.mean(sampler.acceptance_fraction)))
    return trace


def plot_traces_mcmc(trace1, model1, data=None, percentile=95, plot_trace=False,
                     plot_data=False, outfile=None, fill=True,
                     p_best=None, t_plot=np.arange(0, 181, 0.01)):
    # if p_best is provided, plot the best model; otherwise plot the boundary of confidence range
    fig = plt.figure()
    sim_data = np.zeros((trace1.shape[0], data[0].shape[0])).astype('float')
    for i, p_ in enumerate(trace1):
        sim_data[i] = model1(data[0], *p_)
        if i < 100 and plot_trace:
            plt.plot(data[0], sim_data[i])

    if not plot_trace:
        perc1_ = np.percentile(sim_data, (100. - percentile) / 2., axis=0)
        perc2_ = np.percentile(sim_data, percentile + (100. - percentile) / 2., axis=0)
        if p_best is None:
            plt.plot(data[0], perc1_, 'k', label="{0}% interval".format(percentile))
            plt.plot(data[0], perc2_, 'k')
        else:
            # p_best
            plt.plot(t_plot, model1(t_plot, *p_best), color='k', ls='--', lw=2., label="best-fit model")

        if fill:
            plt.fill_between(data[0], perc1_, perc2_, facecolor='k', interpolate=True, alpha=0.2,
                             label="{0}% interval".format(percentile))

    if plot_data:
        plt.errorbar(min_of_mjd(df_lc.t.values, mjd0=57666.165), df_lc.f.values / plot_f_unit, xerr=2,
                     yerr=df_lc.df.values / plot_f_unit,
                     fmt='.', color='k', ecolor='k', capthick=0, alpha=0.8, label="5-min bin")


    plt.xlim(-5, 195)
    plt.xlabel('Minute since MJD %.3f' % 57666.165)
    plt.ylabel(r'Flux > 200 GeV (10$^{-6}$ ph m$^{-2}$ s$^{-1}$)')
    plt.legend(loc='best')
    if outfile is not None:
        if outfile[-3:] == 'pdf':
            plt.savefig(outfile)
        else:
            plt.savefig(outfile, dpi=300)
    return sim_data


# posterior report:
def report_pdf(x, x_grid, bw, plot=False):
    print('*' * 16)
    kde_skl = KernelDensity(kernel='gaussian', bandwidth=bw)
    kde_skl.fit(x[:, np.newaxis])
    # score_samples() returns the log-likelihood of the samples
    log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
    kde_pdf = np.exp(log_pdf)

    # parametric fit: assume normal distribution
    loc_param, scale_param = stats.norm.fit(x)

    param_density = stats.norm.pdf(x_grid, loc=loc_param, scale=scale_param)

    if plot:
        fig, ax = plt.subplots(figsize=(10, 6))
        den_, bin_, _ = plt.hist(x, bins=100, normed=True)

        ax.plot(x_grid, kde_pdf, 'r-', label='KDE (Gaussian kernel bw={:.3f})'.format(bw))
        ax.plot(x_grid, param_density, 'k--', label='Gaussian fit density')

    print("Data 68% quantile {}--{}".format(np.percentile(x, 16), np.percentile(x, 84)))

    print('#' * 8)
    kde_peak = x_grid[np.where(kde_pdf == max(kde_pdf))]
    print("KDE peak={}".format(kde_peak))

    print("Data quantile wrt KDE neg_err={}, KDE pos_err={}".format(np.percentile(x, 16) - kde_peak,
                                                                    np.percentile(x, 84) - kde_peak))

    for i, x_ in enumerate(x_grid):
        if abs(np.trapz(kde_pdf[:i], x=x_grid[:i]) - 0.16) < 4e-4:
            kde16 = x_
            #print("KDE 16% quantile: {}".format(kde16))
        elif abs(np.trapz(kde_pdf[:i], x=x_grid[:i]) - 0.5) < 3e-4:
            kde50 = x_
            #print("KDE 50% quantile: {}".format(kde50))
        elif abs(np.trapz(kde_pdf[:i], x=x_grid[:i]) - 0.84) < 5e-4:
            kde84 = x_
            #print("KDE 84% quantile: {}".format(kde84))
            break

    print("KDE 68% interval {} -- {}".format(kde16, kde84))
    print("KDE 68% interval neg_err={}, KDE pos_err={}".format(kde16 - kde_peak, kde84 - kde_peak))
    print("KDE 50% quantile {}".format(kde50))

    print('#' * 8)

    if plot:
        ax.axvline(kde16, color='g', ls='--', lw=2, alpha=0.3)
        ax.axvline(kde84, color='g', ls='--', lw=2, alpha=0.3)
        ax.axvline(kde50, color='r', ls='--', lw=2, alpha=0.9, label="50% kde")
        ax.axvline(kde_peak, color='b', ls='--', lw=2, alpha=0.9, label="peak kde")
        ax.legend(loc='best')

    print("Gaussian peak={}".format(x_grid[np.where(param_density == max(param_density))]))

    print("Gaussian mu={}, sigma={}".format(loc_param, scale_param))
    print("Gaussian {}--{}".format(loc_param - scale_param, loc_param + scale_param))

    print('*' * 16)



def run_mcmc(data):
    p1 = ML_optimize(flare_func, data, [5., 140., 80., 30.], logL)
    trace1 = compute_mcmc(model=flare_func, data=data, p0=p1,
                          log_posterior=log_posterior, nwalkers=50, nburn=1000, nsteps=2000, a=3.5)
    columns = [r'$I_0$', r'$t_p$', r'$t_r$', r'$t_d$', r'$I_B$']
    df1 = pd.DataFrame(trace1, columns=columns[:4])
    pMCMC = df1.mean().values
    #par_names=['amplitude', 'tpeak', 'trise', 'tdecay']
    #for i in [0,1, 2, 3]:
    #    print("Flare LC exponential fit with yerr L1 loss param %d (%s) = %.4g +/- %.4g" % (i, par_names[i], pMCMC[i], np.sqrt(abs(covLCL1[i,i]))))

    plot_traces_mcmc(trace1, flare_func, data=data, plot_data=True, p_best=pMCMC,
                     outfile=None)

    for col_ in df1.columns:
        x = df1[col_].values
        # bw = (np.percentile(x, 99)-np.percentile(x, 1))/49.
        # x_grid = np.arange(np.percentile(x, 1), np.percentile(x, 99), bw/20.)
        bw = (max(x) - min(x)) / 100.
        x_grid = np.arange(min(x), max(x), bw / 100.)

        print("Param {}".format(col_))
        report_pdf(x, x_grid, bw)

    plt.show()


def simple_scipy_fit(df_lc, plot_f_unit = 1.e-6):
    #####################
    # perform the fit
    #####################
    parLCL1, covLCL1 = curve_fit(flare_func, min_of_mjd(df_lc.t.values, mjd0=57666.165),
                            df_lc.f.values/plot_f_unit, p0=[5., 140., 80., 30.],
                            sigma=df_lc.df.values/plot_f_unit, absolute_sigma=True,
                            method='trf', ftol=1e-10, loss='soft_l1') #approx abs

    par_names=['amplitude', 'tpeak', 'trise', 'tdecay']
    for i in [0,1, 2, 3]:
        print("Flare LC exponential fit with yerr L1 loss param %d (%s) = %.4g +/- %.4g" % (i, par_names[i], parLCL1[i], np.sqrt(abs(covLCL1[i,i]))))

    chi2LC2L1, dof2L1, redchi2LC2L1 = chisq((df_lc.f.values/plot_f_unit),
                                      flare_func(min_of_mjd(df_lc.t.values, mjd0=57666.165), *parLCL1),
                                      (df_lc.df.values/plot_f_unit),
                                      4)
    print("5-min LC fit Chisq is %.3f, dof is %d, reduced Chisq is %.2f" % (chi2LC2L1, dof2L1, redchi2LC2L1))

    # plot LC
    data_ts = min_of_mjd(df_lc.t.values, mjd0=57666.165)
    plt.errorbar(data_ts, df_lc.f.values/plot_f_unit, xerr=2, yerr=df_lc.df.values/plot_f_unit,
                fmt='.', color='k', ecolor='k', capthick=0, alpha=0.8, label="5-min bin")

    # plot best-fit model
    plot_ts = np.arange(np.min(data_ts), np.max(data_ts), (np.max(data_ts) - np.min(data_ts))/100. )
    plt.plot(plot_ts, flare_func(plot_ts, *parLCL1), 'k--', alpha=0.8) #, label="Best fit 5-min")
    plt.xlim(-15, 195)
    plt.ylim(0,5)
    plt.xlabel('Minute since MJD %.5f' % 57666.165)
    plt.ylabel(r'Flux > 200 GeV (10$^{-6}$ ph m$^{-2}$ s$^{-1}$)')
    plt.legend(loc='upper left', ncol=1, fontsize=11)
    plt.tight_layout()
    plt.show()
    #plt.savefig("BLLac_20161005_LC200GeV_expo_fit.png", dpi=300)


def yay_or_nay(question, default_no=True):
    choices = ' [y/N]: ' if default_no else ' [Y/n]: '
    default_answer = 'n' if default_no else 'y'
    reply = str(input(question + choices)).lower().strip() or default_answer
    if reply[0] == 'y':
        return True
    if reply[0] == 'n':
        return False
    else:
        return False if default_no else True


if __name__ == "__main__":

    # read the light curve files
    df_lc = pd.read_csv("data/BLLac/BLLac_20161005_5minLC_200GeV.txt", sep=r"\s+", header=None,
                         names=['t', 'f', 'df', 'lt'])

    plot_f_unit = 1.e-6

    if yay_or_nay("Wanna throw the MCMC hammer?"):
        data = np.vstack(
            (min_of_mjd(df_lc.t.values, mjd0=57666.165), df_lc.f.values / plot_f_unit, df_lc.df.values / plot_f_unit))
        run_mcmc(data)
    else:
        simple_scipy_fit(df_lc)




