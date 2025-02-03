import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import scipy as sp
import glob
import os
from scipy.optimize import minimize
import math


class NanoFTIRDataSet:
    def __init__(self, harmonic=2, balanced_detection=True, sim_depth=1, sim_averages=None, real_only=False):
        self.substrate_paths = None
        self.sample_paths = None
        self.harmonic = None
        self.real_only = real_only

        self.meta_data = None
        self.substrate_ifg = None
        self.sample_ifg = None
        self.substrate_spec = None
        self.sample_spec = None
        self.wavenumbers = None

        self.balanced_detection = balanced_detection
        self.harmonic = harmonic
        self.sim_depth = sim_depth
        self.sim_averages = sim_averages

        self.fontsize = 14
        plt.rcParams.update({
            'font.size': self.fontsize,  # Default font size
            'axes.titlesize': self.fontsize,  # Title font size
            'axes.labelsize': self.fontsize,  # Axis label font size
            'xtick.labelsize': self.fontsize,  # X-axis tick font size
            'ytick.labelsize': self.fontsize,  # Y-axis tick font size
            'legend.fontsize': self.fontsize,  # Legend font size
            'figure.dpi': 100,  # Default figure dpi
        })

    def load_meta_data(self, paths):
        if not isinstance(paths, list):
            paths = [paths]

        p = glob.glob(paths[0]+'/*Interferograms.txt')[0]
        self.meta_data = {}
        with open(p, 'r', encoding='utf8') as f:
            data = f.readlines()

            self.meta_data['date'] = data[4].split()[-2]
            self.meta_data['time'] = data[4].split()[-1]
            self.meta_data['depth'] = int(data[8].split()[-1])
            self.meta_data['interferometerCenter'] = float(data[9].split()[-2])
            self.meta_data['interferometerDistance'] = float(data[9].split()[-1])
            self.meta_data['numAverages'] = int(data[10].split()[-1])
            self.meta_data['integrationTime'] = float(data[11].split()[-1])
            self.meta_data['scalingFactor'] = float(data[12].split()[-1])
            self.meta_data['tappingAmplitude'] = float(data[19].split()[-1])
            self.meta_data['setpoint'] = float(data[23].split()[-1])
            # self.meta_data['qFactor'] = float(data[28].split()[-1])

            # convert interferometer distance to cm
            self.interferometer_distance = self.meta_data['interferometerDistance']*1e-4
            self.frequency_resolution = 1/(2*self.interferometer_distance)
            self.cut_off_frequency = self.meta_data['depth']*self.frequency_resolution/2

            self.averages = self.meta_data['numAverages']
            self.depth = int(self.meta_data['depth']/self.sim_depth)

            self.meta_data['frequencyResolution'] = self.frequency_resolution
            self.meta_data['cutOffFrequency'] = self.cut_off_frequency

    def load_ifg(self, paths, verbose=True):
        self.load_meta_data(paths)

        if not isinstance(paths, list):
            paths = [paths]

        if self.sim_averages is None:
            optical_amplitude = np.zeros((len(paths), self.averages, self.depth))
            optical_phase = np.zeros((len(paths), self.averages, self.depth))
            aux_amplitude = np.zeros((len(paths), self.averages, self.depth))
            aux_phase = np.zeros((len(paths), self.averages, self.depth))
        else:
            optical_amplitude = np.zeros((len(paths), self.sim_averages, self.depth))
            optical_phase = np.zeros((len(paths), self.sim_averages, self.depth))
            aux_amplitude = np.zeros((len(paths), self.sim_averages, self.depth))
            aux_phase = np.zeros((len(paths), self.sim_averages, self.depth))

        for i, path in enumerate(paths):
            p = glob.glob(path+'/*Interferograms.txt')[0]

            # CHECK HEADER
            try:
                df = pd.read_csv(p, sep='\t', header=30)

                optical_amplitude[i, :, :] = df[f'O{self.harmonic}A'].values[::self.sim_depth].reshape(
                    self.averages, self.depth
                    )[:self.sim_averages]
            except KeyError:
                df = pd.read_csv(p, sep='\t', header=29)

                optical_amplitude[i, :, :] = df[f'O{self.harmonic}A'].values[::self.sim_depth].reshape(
                    self.averages, self.depth
                )[:self.sim_averages]

            optical_phase[i, :, :] = df[f'O{self.harmonic}P'].values[::self.sim_depth].reshape(
                self.averages, self.depth
                )[:self.sim_averages]

            if self.balanced_detection:
                try:
                    aux_amplitude[i, :, :] = df[f'A{self.harmonic}A'].values[::self.sim_depth].reshape(
                        self.averages, self.depth
                        )[:self.sim_averages]
                    aux_phase[i, :, :] = df[f'A{self.harmonic}P'].values[::self.sim_depth].reshape(
                        self.averages, self.depth
                        )[:self.sim_averages]
                except KeyError:
                    aux_amplitude[i, :, :] = df[f'B{self.harmonic}A'].values[::self.sim_depth].reshape(
                        self.averages, self.depth
                        )[:self.sim_averages]
                    aux_phase[i, :, :] = df[f'B{self.harmonic}P'].values[::self.sim_depth].reshape(
                        self.averages, self.depth
                        )[:self.sim_averages]

        if self.balanced_detection:
            raw_sigma = optical_amplitude*np.exp(1j*optical_phase)
            raw_ifg = raw_sigma-np.mean(raw_sigma, axis=-1)[..., np.newaxis]

            aux_sigma = aux_amplitude*np.exp(1j*aux_phase)
            aux_ifg = aux_sigma-np.mean(aux_sigma, axis=-1)[..., np.newaxis]

            ifg, scaling_factor, phase = self.balanced_correction(raw_ifg, aux_ifg)

            if self.real_only:
                if verbose:
                    for i in np.mean(ifg, axis=1):
                        ln1 = plt.plot(np.real(i), np.imag(i), c='tab:blue', marker='.', ls='', lw=1, alpha=0.5)

                angle = data_angle(np.mean(ifg, axis=(0, 1)))
                ifg *= np.exp(-1j*angle)
                raw_ifg *= np.exp(-1j*angle)
                aux_ifg *= np.exp(-1j*angle)

                if verbose:
                    for i in np.mean(ifg, axis=1):
                        ln2 = plt.plot(np.real(i), np.imag(i), c='tab:orange', marker='.', ls='', lw=1, alpha=0.5)

                    plt.title(f'Before and after rotation. Angle: {angle*180/np.pi:.2f} deg')
                    lns = ln1+ln2
                    plt.legend(lns, ['Before', 'After'])
                    plt.xlabel('Real')
                    plt.ylabel('Imaginary')
                    plt.show()

            if verbose:
                fig = plt.figure(figsize=(6, 6))
                gs = GridSpec(4, 4, figure=fig)

                ax_main = fig.add_subplot(gs[1:4, 0:3])
                ax_xhist = fig.add_subplot(gs[0, 0:3], sharex=ax_main)
                ax_yhist = fig.add_subplot(gs[1:4, 3], sharey=ax_main)

                for i in raw_ifg.reshape(-1, self.depth):
                    ln1 = ax_main.plot(np.real(i), np.imag(i), c='tab:blue', marker='.', ls='', lw=1, alpha=0.5)

                # for i in aux_ifg.reshape(-1, self.depth):
                #     ln2 = plt.plot(np.real(i), np.imag(i), c='tab:green', marker='.', ls='', lw=1, alpha=0.5)

                for i in aux_ifg.reshape(-1, self.depth)*scaling_factor*np.exp(1j*phase):
                    ln3 = ax_main.plot(np.real(i), np.imag(i), c='tab:green', marker='.', ls='', lw=1, alpha=0.5)

                for i in ifg.reshape(-1, self.depth):
                    ln4 = ax_main.plot(np.real(i), np.imag(i), c='tab:orange', marker='.', ls='', lw=1, alpha=0.5)

                ax_xhist.set_title(f'Interferogram $Ĩ_{self.harmonic}$')
                lns = ln1+ln3+ln4
                ax_main.legend(lns, ['Det. O', 'Scaled Det. A', 'Corrected'])
                max_abs = np.max(np.abs(ifg))
                ax_main.set_xlim(-max_abs, max_abs)
                ax_main.set_ylim(-max_abs, max_abs)
                ax_main.set_xlabel('Real')
                ax_main.set_ylabel('Imaginary')

                ax_xhist.tick_params(axis="both", which="both", labelbottom=False, labelleft=False)
                ax_yhist.tick_params(axis="both", which="both", labelbottom=False, labelleft=False)

                ax_xhist.hist(np.real(aux_ifg.reshape(-1, self.depth)*scaling_factor*np.exp(1j*phase)).flatten(),
                              bins=100, color='tab:green', alpha=0.5)
                ax_yhist.hist(np.imag(aux_ifg.reshape(-1, self.depth)*scaling_factor*np.exp(1j*phase)).flatten(),
                              bins=100, color='tab:green', alpha=0.5, orientation='horizontal')

                ax_xhist.set_xticks([])
                ax_yhist.set_yticks([])
                ax_xhist.set_ylabel("Marginal")
                ax_yhist.set_xlabel("Marginal")

                ax_main.tick_params(axis="both", which="both", labelbottom=True, labelleft=True)

                fig.tight_layout()
                plt.show()

            if self.real_only:
                ifg = np.real(ifg)
                raw_ifg = np.real(raw_ifg)
                aux_ifg = np.real(aux_ifg)

            return ifg, raw_ifg, aux_ifg, scaling_factor, phase
        else:
            sigma = optical_amplitude*np.exp(1j*optical_phase)
            ifg = sigma-np.mean(sigma, axis=-1)[..., np.newaxis]

            if self.real_only:
                angle = data_angle(np.mean(ifg, axis=(0, 1)))
                ifg *= np.exp(-1j*angle)
                ifg = np.real(ifg)

            return ifg

    def _rms(self, params, raw, aux):
        corr = raw-params[0]*aux*np.exp(1j*params[1])
        fft = sp.fft.fft(corr)
        return np.sqrt(np.mean(np.abs(fft[..., self.lolim:self.hilim])**2))

    def balanced_correction(self, raw_ifg, aux_ifg):
        self.lower_bound = 0.1
        self.upper_bound = 30

        # TODO FIND BEST NOISE FLOOR REGION
        self.lolim = int(4000/self.frequency_resolution)
        self.hilim = int(5000/self.frequency_resolution)

        # res = minimize(self._rms, [1, 0], args=(raw_ifg, aux_ifg),
        #                bounds=[(self.lower_bound, self.upper_bound), (-np.pi, np.pi)])
        # scaling_factor = res.x[0]
        # phase = res.x[1]
        # calculate scaling factor and phase for each interferogram and average
        shift_factor = 0+1j*0
        # corr_ifg = np.zeros(raw_ifg.shape, dtype=complex)
        for i in range(raw_ifg.shape[0]):
            for j in range(raw_ifg.shape[1]):
                res = minimize(self._rms, [1, 0], args=(raw_ifg[i, j], aux_ifg[i, j]),
                               bounds=[(self.lower_bound, self.upper_bound), (-np.pi, np.pi)])

                # scaling_factor = res.x[0]
                # phase = res.x[1]

                # corr_ifg[i, j] = raw_ifg[i, j]-scaling_factor*aux_ifg[i, j]*np.exp(1j*phase)
                shift_factor += res.x[0]*np.exp(1j*res.x[1])

        shift_factor /= raw_ifg.shape[0]*raw_ifg.shape[1]
        scaling_factor = np.abs(shift_factor)
        phase = np.angle(shift_factor)

        corr_ifg = raw_ifg-scaling_factor*aux_ifg*np.exp(1j*phase)

        # check if close to bounds:
        if math.isclose(scaling_factor, self.lower_bound, rel_tol=1e-3) or math.isclose(
                        scaling_factor, self.upper_bound, rel_tol=1e-3):
            print(f'Warning: Scaling factor at bound: {scaling_factor}')

        return corr_ifg, scaling_factor, phase

    def load_substrate_ifg(self, paths, verbose=True):
        if self.balanced_detection:
            (self.substrate_ifg, self.substrate_raw_ifg, self.substrate_aux_ifg,
             self.substrate_scaling, self.substrate_phase) = self.load_ifg(paths, verbose=verbose)
        else:
            self.substrate_ifg = self.load_ifg(paths, verbose=verbose)

    def load_sample_ifg(self, paths, verbose=True):
        if self.balanced_detection:
            (self.sample_ifg, self.sample_raw_ifg, self.sample_aux_ifg,
             self.sample_scaling, self.sample_phase) = self.load_ifg(paths, verbose=verbose)
        else:
            self.sample_ifg = self.load_ifg(paths, verbose=verbose)

    def calculate_spectra(self):
        if self.sample_ifg is not None:
            apodized_ifg = self._Window(np.mean(self.sample_ifg, axis=(0, 1)))*self.sample_ifg
            self.sample_spec = sp.fft.fft(apodized_ifg, axis=-1)
            self.sample_spec = self.sample_spec[..., :self.depth//2]

            if self.balanced_detection:
                apodized_raw_ifg = self._Window(np.mean(self.sample_raw_ifg, axis=(0, 1)))*self.sample_raw_ifg
                self.sample_raw_spec = sp.fft.fft(apodized_raw_ifg, axis=-1)
                self.sample_raw_spec = self.sample_raw_spec[..., :self.depth//2]

        if self.substrate_ifg is not None:
            apodized_ifg = self._Window(np.mean(self.substrate_ifg, axis=(0, 1)))*self.substrate_ifg
            self.substrate_spec = sp.fft.fft(apodized_ifg, axis=-1)
            self.substrate_spec = self.substrate_spec[..., :self.depth//2]

            if self.balanced_detection:
                apodized_raw_ifg = self._Window(np.mean(self.substrate_raw_ifg, axis=(0, 1)))*self.substrate_raw_ifg
                self.substrate_raw_spec = sp.fft.fft(apodized_raw_ifg, axis=-1)
                self.substrate_raw_spec = self.substrate_raw_spec[..., :self.depth//2]

        if self.sample_spec is None and self.substrate_spec is None:
            raise ValueError('Interferograms not loaded.')

        dx = 2*self.interferometer_distance/self.depth
        self.wavenumbers = sp.fft.fftfreq(self.depth, d=dx)[:self.depth//2]

    def reference_spectra(self):
        if self.sample_spec is None:
            try:
                self.calculate_spectra()
            except ValueError:
                raise ValueError('Sample spectra not calculated.')
        if self.substrate_spec is None:
            try:
                self.calculate_spectra()
            except ValueError:
                raise ValueError('Substrate spectra not calculated.')

        if self.substrate_spec.shape[0] == 1:
            self.substrate_spec_avg = (
                self.substrate_spec[0, :-1] + self.substrate_spec[0, 1:]
            )/2
            if self.balanced_detection:
                self.substrate_raw_spec_avg = (
                    self.substrate_raw_spec[0, :-1] + self.substrate_raw_spec[0, 1:]
                )

        elif self.sample_spec.shape != self.substrate_spec.shape:
            self.substrate_spec_avg = (
                np.mean(self.substrate_spec, axis=1)[:-1]+np.mean(self.substrate_spec, axis=1)[1:]
                )/2
            if self.balanced_detection:
                self.substrate_raw_spec_avg = (
                    np.mean(self.substrate_raw_spec, axis=1)[:-1]+np.mean(self.substrate_raw_spec, axis=1)[1:]
                    )/2

        else:
            self.substrate_spec_avg = np.mean(self.substrate_spec, axis=1)
            if self.balanced_detection:
                self.substrate_raw_spec_avg = np.mean(self.substrate_raw_spec, axis=1)

        if self.sample_spec.shape[0] == 1:
            self.referenced_spec = self.sample_spec[0]/self.substrate_spec_avg
            if self.balanced_detection:
                self.referenced_raw_spec = self.sample_raw_spec[0]/self.substrate_raw_spec_avg
        else:
            self.referenced_spec = np.mean(self.sample_spec, axis=1)/self.substrate_spec_avg
            if self.balanced_detection:
                self.referenced_raw_spec = np.mean(self.sample_raw_spec, axis=1)/self.substrate_raw_spec_avg

        return self.referenced_spec

    def plot_ifg(self):
        alpha = 1
        if self.substrate_ifg is not None:
            plt.figure(figsize=(8, 6))
            plt.title('Interferogram of substrate')
            if self.balanced_correction:
                plt.plot(np.real(self.substrate_raw_ifg[0, 0]), label='Det. O', c='tab:blue', alpha=alpha)
                plt.plot(np.real(self.substrate_aux_ifg[0, 0]*self.substrate_scaling*np.exp(1j*self.substrate_phase)),
                         label='Det. A, scaled', c='tab:green', alpha=alpha)
            plt.plot(np.real(self.substrate_ifg[0, 0]), label='Corr', c='tab:orange', alpha=1)
            plt.legend()
            plt.xlabel('Depth (a.u.)')
            plt.ylabel(f'Intensity I$_{self.harmonic}$ (a.u.)')
            plt.show()
        if self.sample_ifg is not None:
            plt.figure(figsize=(8, 6))
            plt.title('Interferogram of sample')
            if self.balanced_correction:
                plt.plot(np.real(self.sample_raw_ifg[0, 0]), label='Det. O', c='tab:blue', alpha=alpha)
                plt.plot(np.real(self.sample_aux_ifg[0, 0]*self.sample_scaling*np.exp(1j*self.sample_phase)),
                         label='Det. A, scaled', c='tab:green', alpha=alpha)
            plt.plot(np.real(self.sample_ifg[0, 0]), label='Corr', c='tab:orange', alpha=1)
            plt.legend()
            plt.xlabel('Depth (a.u.)')
            plt.ylabel(f'Intensity I$_{self.harmonic}$ (a.u.)')
            plt.show()
        else:
            raise ValueError('No interferogram loaded.')

    def plot_ref_spec(self, lolim=1100, hilim=2000, fig=None, ax=None, plot_indv=False, comp_plot=False):
        roi = np.where((self.wavenumbers > lolim) & (self.wavenumbers < hilim))

        self.mean_spec = np.mean(self.referenced_spec, axis=0)
        self.psd_spec = np.std(np.angle(self.referenced_spec), axis=0)
        self.asd_spec = np.std(np.abs(self.referenced_spec), axis=0)

        if self.balanced_detection:
            self.mean_raw_spec = np.mean(self.referenced_raw_spec, axis=0)
            self.psd_raw_spec = np.std(np.angle(self.referenced_raw_spec), axis=0)
            self.asd_raw_spec = np.std(np.abs(self.referenced_raw_spec), axis=0)

        mean_psd = np.mean(self.psd_spec[roi])
        mean_asd = np.mean(self.asd_spec[roi])
        print(f'Amplitude noise: {mean_asd:.4f} a.u.')
        print(f'Phase noise: {mean_psd*1e3:.1f} mrad')
        if self.balanced_detection:
            mean_raw_psd = np.mean(self.psd_raw_spec[roi])
            mean_raw_asd = np.mean(self.asd_raw_spec[roi])
            print(f'Raw amplitude noise: {mean_raw_asd:.4f} a.u.')
            print(f'Raw phase noise: {mean_raw_psd*1e3:.1f} mrad')

            self.asd_improv = mean_raw_asd/mean_asd
            self.psd_improv = mean_raw_psd/mean_psd

            print(f'Amplitude noise improvement: {self.asd_improv:.2f}')
            print(f'Phase noise improvement: {self.psd_improv:.2f}')

        if fig is None and ax is None:
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        fig.suptitle(
            f'O{self.harmonic}, depth: {self.depth}, averages: {self.averages},' +
            f'ifg. distance: {1e4*self.interferometer_distance:.0f} microns,\n' +
            f'int. time: {self.meta_data["integrationTime"]:.1f} ms, setpoint: {self.meta_data["setpoint"]:.1f} %' +
            f', tapping ampl.: {self.meta_data["tappingAmplitude"]:.0f} nm'
            )

        cmap_orange = plt.cm.Oranges(np.linspace(0.5, 1, self.referenced_spec.shape[0]))

        if self.balanced_detection and comp_plot:
            cmap_blue = plt.cm.Blues(np.linspace(0.5, 1, self.referenced_raw_spec.shape[0]))
            ax[0].plot(self.wavenumbers[roi], np.angle(self.mean_raw_spec[roi]),
                       label=f'Mean raw PSD: {mean_raw_psd*1e3:.1f} mrad', c='tab:blue')
            if not plot_indv:
                ax[0].fill_between(self.wavenumbers[roi], np.angle(self.mean_raw_spec[roi])-self.psd_raw_spec[roi],
                                   np.angle(self.mean_raw_spec[roi])+self.psd_raw_spec[roi],
                                   alpha=0.5, color='tab:blue')
            else:
                for i in range(self.referenced_raw_spec.shape[0]):
                    ax[0].plot(self.wavenumbers[roi], np.angle(self.referenced_raw_spec[i])[roi],
                               alpha=0.5, lw=1, c=cmap_blue[i])

        ax[0].plot(self.wavenumbers[roi], np.angle(self.mean_spec[roi]),
                   label=f'Mean PSD: {mean_psd*1e3:.1f} mrad', c='tab:orange')
        if not plot_indv:
            ax[0].fill_between(self.wavenumbers[roi], np.angle(self.mean_spec[roi])-self.psd_spec[roi],
                               np.angle(self.mean_spec[roi])+self.psd_spec[roi], alpha=0.5, color='tab:orange')
        else:
            for i in range(self.referenced_spec.shape[0]):
                ax[0].plot(self.wavenumbers[roi], np.angle(self.referenced_spec[i])[roi],
                           alpha=0.5, lw=1, c=cmap_orange[i])

        ax[0].set_title('Phase spectrum')
        ax[0].legend()
        ax[0].set_xlabel(r'Frequency $\omega$ (cm$^{-1}$)')
        ax[0].set_xlim(lolim, hilim)
        ax[0].set_ylabel(rf'Phase $\phi_{self.harmonic}$ (rad)')

        if self.balanced_detection and comp_plot:
            ax[1].plot(self.wavenumbers[roi], np.abs(self.mean_raw_spec[roi]),
                       label=f'Mean raw ASD: {mean_raw_asd:.4f} a.u.', c='tab:blue')
            if not plot_indv:
                ax[1].fill_between(self.wavenumbers[roi], np.abs(self.mean_raw_spec[roi])-self.asd_raw_spec[roi],
                                   np.abs(self.mean_raw_spec[roi])+self.asd_raw_spec[roi], alpha=0.5, color='tab:blue')
            else:
                for i in range(self.referenced_raw_spec.shape[0]):
                    ax[1].plot(self.wavenumbers[roi], np.abs(self.referenced_raw_spec[i])[roi],
                               alpha=0.5, lw=1, c=cmap_blue[i])

        ax[1].plot(self.wavenumbers[roi], np.abs(self.mean_spec[roi]),
                   label=f'Mean ASD: {mean_asd:.4f} a.u.', c='tab:orange')
        if not plot_indv:
            ax[1].fill_between(self.wavenumbers[roi], np.abs(self.mean_spec[roi])-self.asd_spec[roi],
                               np.abs(self.mean_spec[roi])+self.asd_spec[roi], alpha=0.5, color='tab:orange')
        else:
            for i in range(self.referenced_spec.shape[0]):
                ax[1].plot(self.wavenumbers[roi], np.abs(self.referenced_spec[i])[roi],
                           alpha=0.5, lw=1, c=cmap_orange[i])

        ax[1].set_title('Amplitude spectrum')
        ax[1].legend()
        ax[1].set_xlabel(r'Frequency $\omega$ (cm$^{-1}$)')
        ax[1].set_xlim(lolim, hilim)
        ax[1].set_ylabel(rf'Amplitude $s_{self.harmonic}$ (a.u.)')

        fig.tight_layout()
        plt.show()
        # return fig, ax

    def plot_spec(self, lolim=0, hilim=5000, fig=None, ax=None, plot_indv=True, which='substrate'):
        if not self.balanced_detection:
            raise ValueError('Balanced detection not used.')

        roi = np.where((self.wavenumbers > lolim) & (self.wavenumbers < hilim))
        noise_roi = np.arange(self.lolim, self.hilim)

        if fig is None and ax is None:
            fig, ax = plt.subplots(3, 2, figsize=(12, 8))

        fig.suptitle(
            f'O{self.harmonic}, depth: {self.depth}, averages: {self.averages},' +
            f'ifg. distance: {1e4*self.interferometer_distance:.0f} microns,\n' +
            f'int. time: {self.meta_data["integrationTime"]:.1f} ms, setpoint: {self.meta_data["setpoint"]:.1f} %' +
            f', tapping ampl.: {self.meta_data["tappingAmplitude"]:.0f} nm'
            )

        if which == 'substrate':
            spec = np.mean(self.substrate_spec, axis=1)
            ifg = np.mean(self.substrate_ifg, axis=1)
            if self.balanced_detection:
                raw_ifg = np.mean(self.substrate_raw_ifg, axis=1)
                aux_ifg = np.mean(self.substrate_aux_ifg, axis=1)

                apodized_ifg = self._Window(np.mean(raw_ifg, axis=0))*raw_ifg
                raw_spec = sp.fft.fft(apodized_ifg, axis=-1)[..., :self.depth//2]

                scaling = self.substrate_scaling
                phase = self.substrate_phase

        elif which == 'sample':
            spec = np.mean(self.sample_spec, axis=1)
            ifg = np.mean(self.sample_ifg, axis=1)
            if self.balanced_detection:
                raw_ifg = np.mean(self.sample_raw_ifg, axis=1)
                aux_ifg = np.mean(self.sample_aux_ifg, axis=1)

                apodized_ifg = self._Window(np.mean(raw_ifg, axis=0))*raw_ifg
                raw_spec = sp.fft.fft(apodized_ifg, axis=-1)[..., :self.depth//2]

                scaling = self.sample_scaling
                phase = self.sample_phase

        else:
            raise ValueError('Unknown type.')

        # cmap_blue = plt.cm.Blues(np.linspace(0.5, 1, spec.shape[0]))
        # cmap_orange = plt.cm.Oranges(np.linspace(0.5, 1, raw_spec.shape[0]))

        # corrected quantities
        # mean_spec = np.mean(spec, axis=0)
        # psd_spec = np.std(np.angle(spec), axis=0)
        asd_spec = np.std(np.abs(spec), axis=0)
        noise_floor = np.sqrt(np.mean(np.square(np.abs(spec[..., noise_roi]))))

        # self-referencing
        quasi_sample_spec = np.copy(spec[1:-1])
        quasi_substrate_spec = (spec[:-2]+spec[2:])/2
        quasi_ref_sepc = quasi_sample_spec/quasi_substrate_spec
        mean_quasi_ref_spec = np.mean(quasi_ref_sepc, axis=0)
        psd_quasi_ref_spec = np.std(np.angle(quasi_ref_sepc), axis=0)
        # asd_quasi_ref_spec = np.std(np.abs(quasi_ref_sepc), axis=0)

        # raw quantities
        # mean_raw_spec = np.mean(raw_spec, axis=0)
        # psd_raw_spec = np.std(np.angle(raw_spec), axis=0)
        asd_raw_spec = np.std(np.abs(raw_spec), axis=0)
        asd_improv = asd_raw_spec/asd_spec
        raw_noise_floor = np.sqrt(np.mean(np.square(np.abs(raw_spec[..., noise_roi]))))

        # self-referencing
        quasi_sample_raw_spec = np.copy(raw_spec[1:-1])
        quasi_substrate_raw_spec = (raw_spec[:-2]+raw_spec[2:])/2
        quasi_ref_raw_spec = quasi_sample_raw_spec/quasi_substrate_raw_spec
        mean_quasi_ref_raw_spec = np.mean(quasi_ref_raw_spec, axis=0)
        psd_quasi_ref_raw_spec = np.std(np.angle(quasi_ref_raw_spec), axis=0)

        # calculate noise floor 'loss landscape'
        s_num = 50
        p_num = 50
        lb = max(0.1, scaling-scaling/2)
        ub = min(30, scaling+scaling/2)
        scaling_factors = np.linspace(lb, ub, s_num)
        phase_factors = np.linspace(-1, 1, p_num) * np.pi

        sF, pF = np.meshgrid(scaling_factors, phase_factors)
        corr_factors = sF * np.exp(1j * pF)
        idx1 = 0

        all_corrections = raw_ifg[idx1, np.newaxis, :] - (
            corr_factors.flatten()[:, np.newaxis] * aux_ifg[idx1, np.newaxis, :]
            )
        all_ffts = sp.fft.fft(all_corrections, axis=1)

        noise_floor1 = np.sqrt(np.mean(np.abs(all_ffts[:, self.lolim:self.hilim])**2, axis=1))
        min_noisefloorindex_2D = np.unravel_index(np.argmin(noise_floor1), (p_num, s_num))
        min_noise_floor_scaling = scaling_factors[min_noisefloorindex_2D[1]]
        min_noise_floor_phase = phase_factors[min_noisefloorindex_2D[0]]

        # Plot noise floors
        ax[0, 0].set_title('Noise floors')
        im = ax[0, 0].imshow(
            noise_floor1.reshape((p_num, s_num)),
            extent=[scaling_factors[0], scaling_factors[-1], phase_factors[0], phase_factors[-1]],
            aspect='auto', origin='lower'
        )
        ax[0, 0].plot(min_noise_floor_scaling, min_noise_floor_phase, 'ro', label='Old method')
        ax[0, 0].plot(scaling, phase, 'rx', label='New method')

        ax[0, 0].set_xlabel('Scaling factor')
        ax[0, 0].set_ylabel('Phase factor')
        fig.colorbar(im, label='Ampl. noise floor')

        # plot interferograms
        ax[0, 1].plot(self._Window(np.mean(raw_ifg, axis=0))*np.real(raw_ifg[0, :]).max(), c='gray')
        ax[0, 1].plot(np.real(raw_ifg[0, :]), label='Raw', zorder=0, c='tab:blue')
        ax[0, 1].plot(np.real(aux_ifg[0, :]*scaling*np.exp(1j*phase)), label='Scaled Aux', zorder=2, c='tab:green')
        ax[0, 1].plot(np.real(ifg[0, :]), label='Corr', zorder=1, c='tab:orange')
        ax[0, 1].set_xlabel('Depth (a.u.)')
        ax[0, 1].set_ylabel(f'Intensity I$_{self.harmonic}$ (a.u.)')

        # plot amplitude spectrum
        ax[1, 0].plot(self.wavenumbers, np.abs(raw_spec[0]), label='Raw', zorder=0, c='tab:blue')
        ax[1, 0].hlines(raw_noise_floor, self.wavenumbers[noise_roi][0], self.wavenumbers[noise_roi][-1],
                        label='Noise floor', zorder=0, color='tab:green')
        ax[1, 0].plot(self.wavenumbers, np.abs(spec[0, :]), label='Corr', zorder=1, c='tab:orange')
        ax[1, 0].hlines(noise_floor, self.wavenumbers[noise_roi][0], self.wavenumbers[noise_roi][-1],
                        label='Noise floor', zorder=1, color='tab:red')
        ax[1, 0].set_ylabel(rf'Amplitude $s_{self.harmonic}$ (a.u.)')

        # plot amplitude spectrum std
        ax[1, 1].plot(self.wavenumbers, np.abs(asd_raw_spec), label='Raw', zorder=1, c='tab:blue')
        ax[1, 1].plot(self.wavenumbers, np.abs(asd_spec), zorder=1, c='tab:orange',
                      label=f'Mean Improv.: {np.mean(asd_improv[roi]):.2f}')
        ax[1, 1].set_ylabel(r'Ampl. noise (a.u.)')

        # plot self referenced phase spectrum
        ax[2, 0].plot(self.wavenumbers[roi], np.angle(mean_quasi_ref_raw_spec[roi])*1e3,
                      label='Raw', zorder=1, c='tab:blue')
        ax[2, 0].fill_between(self.wavenumbers[roi],
                              (np.angle(mean_quasi_ref_raw_spec[roi])-psd_quasi_ref_raw_spec[roi])*1e3,
                              (np.angle(mean_quasi_ref_raw_spec[roi])+psd_quasi_ref_raw_spec[roi])*1e3,
                              alpha=0.5, color='tab:blue')
        ax[2, 0].plot(self.wavenumbers[roi], np.angle(mean_quasi_ref_spec[roi])*1e3,
                      label='Corr', zorder=1, c='tab:orange')
        ax[2, 0].fill_between(self.wavenumbers[roi], (np.angle(mean_quasi_ref_spec[roi])-psd_quasi_ref_spec[roi])*1e3,
                              (np.angle(mean_quasi_ref_spec[roi])+psd_quasi_ref_spec[roi])*1e3,
                              alpha=0.5, color='tab:orange')
        ax[2, 0].set_ylabel(rf'Phase $\bar{{\phi}}_{self.harmonic}$ (mrad)')

        ax[2, 1].plot(self.wavenumbers[roi], (psd_quasi_ref_raw_spec/psd_quasi_ref_spec)[roi], c='tab:blue',
                      label=f'Mean Improv.: {np.mean((psd_quasi_ref_raw_spec/psd_quasi_ref_spec)[roi]):.2f}')
        ax[2, 1].set_ylabel(r'Phase noise improv.')

        for i in range(3):
            for j in range(2):
                ax[i, j].legend()
                if i > 0:
                    ax[i, j].set_xlabel(r'Frequency $\omega$ (cm$^{-1}$)')

        fig.tight_layout()
        plt.show()
        return

    def _Window(self, interferogram):
        zpd = np.argmax(interferogram)

        rightInterferogram = interferogram[zpd:]
        rightX = np.linspace(-0.5, 0.5, len(rightInterferogram))

        rightBlackman = np.blackman(len(rightX))
        leftSlope = rightBlackman[:np.argmax(rightBlackman)]
        rightSlope = rightBlackman[np.argmax(rightBlackman):]
        window = np.ones(len(interferogram))
        window[:len(leftSlope)] = leftSlope
        window[-len(rightSlope):] = rightSlope
        return window

    # smoother version
    def _Window2(self, interferogram):
        zpd = np.argmax(interferogram)
        leftInterferogram = interferogram[:zpd]
        rightInterferogram = interferogram[zpd:]

        leftSlope = np.blackman(2*len(leftInterferogram))
        rightSlope = np.blackman(2*len(rightInterferogram))

        window = np.ones(len(interferogram))
        window[:zpd] = leftSlope[:len(leftInterferogram)]
        window[zpd:] = rightSlope[-len(rightInterferogram):]
        return window


def data_angle(x):
    """
    From Theo
    Return the angle of the best fit line for the array of points (x,y), or the array x in the complex plane.

    Parameters
    ----------
    x : 1D array of complex
        x coordinates, or complexe coordinate of the points to fit.

    Returns
    -------
    angle : float
        Angle of the best fit line in radians.

    """
    xx = np.real(x)
    yy = np.imag(x)

    if np.min(xx) == np.max(xx):        # Data is perfectly vertical
        return np.pi/2
    if np.min(yy) == np.max(yy):        # Data is perfectly hozizontal
        return 0
    regressHor = sp.stats.linregress(xx, yy)
    regressVer = sp.stats.linregress(yy, xx)
    if regressHor.stderr < regressVer.stderr:
        angle = np.arctan(regressHor.slope)
    else:
        angle = np.arctan(1/regressVer.slope)
    return angle


if __name__ == '__main__':
    # %%
    # parent_folder = '/Volumes/MYUSB/2024-12-06 BD4/2024-12-06 15712 NetsIntTime/'
    #parent_folder = '/Volumes/MYUSB/Felix_Alt_BD/2025-01-11 15982/'
    parent_folder = 'C:/Users/neaspec/Desktop/Felix/SNOM_DATA/250130_Felix_Hyperspec/2025-01-31 16467/'
    parent_folder = 'C:/Users/neaspec/Desktop/Martin/martin202501_PMMA/martin202501/2025-02-01 16473/'
    parent_folder = 'C:/Users/neaspec/Desktop/Martin/martin202501_PMMA/martin202501e/2025-02-01 16476/'
    parent_folder = 'C:/Users/neaspec/Desktop/Martin/martin202501_PMMA/martin202501f/2025-02-01 16477/'
    parent_folder = 'C:/Users/neaspec/Desktop/Felix/SNOM_DATA/250203_Felix_LineScans/2025-02-03 16481/'

    # pattern = 'Rng_C_1_Nts_Interleaved_Testing'
    #pattern = 'Rng_C_1_Nts_Standard2'
    # pattern = 'D_2048_T_1C2_A_2'
    pattern = 'check700_avg10'
    pattern = 'check800_4096_0p8'
    pattern = 'Standard_Interleaved_1_Nts'
    # pattern = 'D_2048_T_2C5_A_1'

    # question: how is the window function defined? for 800um scans, the center burst is relatively very on the right (but still 80um from right end). The window function slope should be done in absolute terms, eg over the first and last 40um of the interferogram)

    substrate_path = []
    sample_path = []

    i = 0
    for folder in sorted(os.listdir(parent_folder)):
        if folder.split('NF S ')[-1] == pattern:
            if i % 2 == 0:
                substrate_path.append(os.path.join(parent_folder, folder+'/'))
            else:
                sample_path.append(os.path.join(parent_folder, folder+'/'))
            i += 1

    #%%
    data = NanoFTIRDataSet(balanced_detection=True, harmonic=2, real_only=False)

    data.load_substrate_ifg(substrate_path, verbose=False)
    data.load_sample_ifg(sample_path, verbose=False)
    # data.plot_ifg()
    data.calculate_spectra()
    ref_spec = data.reference_spectra()

    lolim, hilim = 1100, 1770
    data.plot_ref_spec(lolim=lolim, hilim=hilim, plot_indv=False, comp_plot=True)

    data.plot_spec(1110, 1550, which='substrate')
    # data.plot_spec(1100, 1550, which='sample')
    # plt.show()

# %%
