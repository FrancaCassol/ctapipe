"""
Factory for the estimation of the flat field coefficients
"""

from abc import abstractmethod
import numpy as np
from astropy import units as u
from ctapipe.core import Component, Factory

from ctapipe.image import ChargeExtractorFactory, WaveformCleanerFactory
from ctapipe.core.traits import Int
from ctapipe.io.containers import PedestalCameraContainer

__all__ = [
    'PedestalCalculator',
    'ChargeIntegrator',
    'PedestalFactory'
]


class PedestalCalculator(Component):
    """
    Parent class for the pedestal calculators.
    Fills the MON.pedestal container.
    """

    tel_id = Int(
        0,
        help='id of the telescope to calculate the pedestal values'
    ).tag(config=True)
    sample_duration = Int(
        60,
        help='sample duration in seconds'
    ).tag(config=True)
    sample_size = Int(
        10000,
        help='sample size'
    ).tag(config=True)
    n_channels = Int(
        2,
        help='number of channels to be treated'
    ).tag(config=True)

    def __init__(
        self,
        config=None,
        tool=None,
        extractor_product="FullIntegrator",
        **kwargs
    ):
        """
        Parent class for the flat field calculators.
        Fills the MON.flatfield container.

        Parameters
        ----------
        config : traitlets.loader.Config
            Configuration specified by config file or cmdline arguments.
            Used to set traitlet values.
            Set to None if no configuration to pass.
        tool : ctapipe.core.Tool
            Tool executable that is calling this component.
            Passes the correct logger to the component.
            Set to None if no Tool to pass.
        extractor_product : str
            The ChargeExtractor to use.
        kwargs

        """
        super().__init__(config=config, parent=tool, **kwargs)

        # initialize the output
        self.container = PedestalCameraContainer()

        # load the waveform charge extractor and cleaner
        kwargs_ = dict()
        if extractor_product:
            kwargs_['product'] = extractor_product
        self.extractor = ChargeExtractorFactory.produce(
            config=config,
            tool=tool,
            **kwargs_
        )
        self.log.info(f"extractor {self.extractor}")


    @abstractmethod
    def calculate_pedestals(self, event):
        """calculate relative gain from event
        Parameters
        ----------
        event: DataContainer

        Returns: PedestalCameraContainer or None

            None is returned if no new pedestal were calculated
            e.g. due to insufficient statistics.
        """


class ChargeIntegrator(PedestalCalculator):

    def __init__(self, config=None, tool=None, **kwargs):
        """Calculates pedestal parameters from pedestal events


        Parameters: see base class FlatFieldCalculator
        """
        super().__init__(config=config, tool=tool, **kwargs)

        self.log.info("Used events statistics : %d", self.sample_size)

        # members to keep state in calculate_relative_gain()
        self.num_events_seen = 0
        self.time_start = None  # trigger time of first event in sample
        self.charge_medians = None  # med. charge in camera per event in sample
        self.charges = None  # charge per event in sample
        self.sample_bad_pixels = None  # bad pixels per event in sample

    def _extract_charge(self, event):
        """
        Extract the charge and the time from a pedestal event

        Parameters
        ----------
        event : general event container

        """

        waveform = event.r0.tel[self.tel_id].waveform

        # Extract charge and time
        if self.extractor:
            if self.extractor.requires_neighbours():
                g = event.inst.subarray.tel[self.tel_id].camera
                self.extractor.neighbours = g.neighbor_matrix_where

            charge, time, window = self.extractor.extract_charge(waveform)


        return charge, time, window

    def calculate_pedestals(self, event):
        """
        calculate the pedestal statistical values

        Parameters
        ----------
        event : general event container

        """

        # initialize the np array at each cycle
        waveform = event.r0.tel[self.tel_id].waveform


        # patches for MC data
        if not event.mcheader.simtel_version:
            trigger_time = event.r0.tel[self.tel_id].trigger_time
            pixel_status = event.r0.tel[self.tel_id].pixel_status
        else:
            trigger_time = event.trig.gps_time.unix
            pixel_status = np.ones(waveform.shape[1])


        if self.num_events_seen == 0:
            self.time_start = trigger_time
            self.setup_sample_buffers(waveform, self.sample_size)

        # extract the charge of the event and
        # the peak position (assumed as time for the moment)
        charge, time, window = self._extract_charge(event)

        # divide by the width of the integration window
        event_pedestal = charge/window

        self.collect_sample(event_pedestal, pixel_status, time)

        sample_age = trigger_time - self.time_start

        # check if to create a calibration event
        if (
            sample_age > self.sample_duration
            or self.num_events_seen == self.sample_size
        ):
            pedestal_results = calculate_pedestal_results(
                self.charge_medians,
                self.charges,
                self.sample_bad_pixels,
            )
            time_results = calculate_time_results(
                self.time_start,
                trigger_time,
            )

            result = {
                'n_events': self.num_events_seen,
                **pedestal_results,
                **time_results,
            }
            for key, value in result.items():
                setattr(self.container, key, value)

            self.num_events_seen = 0
            return self.container

        else:

            return None

    def setup_sample_buffers(self, waveform, sample_size):
        n_channels = waveform.shape[0]
        n_pix = waveform.shape[1]
        shape = (sample_size, n_channels, n_pix)

        self.charge_medians = np.zeros((sample_size, n_channels))
        self.charges = np.zeros(shape)
        self.sample_bad_pixels = np.zeros(shape)

    def collect_sample(self, charge, pixel_status):

        # extract the charge of the event and
        # the peak position (assumed as time for the moment)
        bad_pixels = np.zeros(charge.shape, dtype=np.bool)
        bad_pixels[:] = pixel_status == 0

        good_charge = np.ma.array(charge, mask=bad_pixels)
        charge_median = np.ma.median(good_charge, axis=1)

        self.charges[self.num_events_seen] = charge
        self.sample_bad_pixels[self.num_events_seen] = bad_pixels
        self.charge_medians[self.num_events_seen] = charge_median
        self.num_events_seen += 1


def calculate_time_results(
    time_start,
    trigger_time,
):

    return {
        'time_mean': (trigger_time - time_start) / 2 * u.s,
        'time_range': [time_start, trigger_time] * u.s,
    }


def calculate_pedestal_results(
    event_median,
    trace_integral,
    bad_pixels_of_sample,
):
    masked_trace_integral = np.ma.array(
        trace_integral,
        mask=bad_pixels_of_sample
    )
    relative_pedestal_event = np.ma.getdata(
        masked_trace_integral / event_median[:, :, np.newaxis]
    )


    return {
        'pedestal_median': np.median(masked_trace_integral, axis=0),
        'pedestal_mean': np.mean(masked_trace_integral, axis=0),
        'pedestal_rms': np.std(masked_trace_integral, axis=0),
        'relative_pedestal_median': np.median(relative_pedestal_event, axis=0),
        'relative_pedestal_mean': np.mean(relative_pedestal_event, axis=0),
        'relative_pedestal_rms': np.std(relative_pedestal_event, axis=0),

    }


class PedestalFactory(Factory):
    """
    Factory to obtain flat-field coefficients
    """
    base = PedestalCalculator
    default = 'PedestalFieldCalculator'
    custom_product_help = ('Pedestal method to use')
