# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
EventSource for LSTCam protobuf-fits.fz-files.

Needs protozfits v1.4.0 from github.com/cta-sst-1m/protozfitsreader
"""

import numpy as np
from os.path import exists
from ctapipe.core import Provenance
from .eventsource import EventSource
from .lsteventsource import MultiFiles
from .containers import NectarCAMDataContainer

__all__ = ['NectarCAMEventSource']


class NectarCAMEventSource(EventSource):

    def __init__(self, config=None, tool=None, **kwargs):
        super().__init__(config=config, tool=tool, **kwargs)

        self.multi_file = MultiFiles(self.input_url)
        self.camera_config = self.multi_file.camera_config

        self.log.info("Read {} input files".format(self.multi_file.num_inputs()))


    def _generator(self):

        # container for LST data
        data = NectarCAMDataContainer()
        data.meta['input_url'] = self.input_url

        # fill data from the CameraConfig table
        self.fill_lst_service_container_from_zfile(data.nectarcam, self.camera_config)

        # loop on events
        for count, event in enumerate(self.multi_file):

            data.count = count

            # fill specific NectarCAM event data
            self.fill_lst_event_container_from_zfile(data.nectarcam, event)

            # fill general R0 data
            self.fill_r0_container_from_zfile(data.r0, event)
            yield data


    @staticmethod
    def is_compatible(file_path):
        from astropy.io import fits
        try:
            # The file contains two tables:
            #  1: CameraConfig
            #  2: Events
            h = fits.open(file_path)[2].header
            ttypes = [
                h[x] for x in h.keys() if 'TTYPE' in x
            ]
        except OSError:
            # not even a fits file
            return False

        except IndexError:
            # A fits file of a different format
            return False

        is_protobuf_zfits_file = (
            (h['XTENSION'] == 'BINTABLE') and
            (h['EXTNAME'] == 'Events') and
            (h['ZTABLE'] is True) and
            (h['ORIGIN'] == 'CTA') and
            (h['PBFHEAD'] == 'R1.CameraEvent')
        )

        is_nectarcam_file = 'nectarcam_counters' in ttypes
        return is_protobuf_zfits_file & is_nectarcam_file

    def fill_lst_service_container_from_zfile(self, container, camera_config):

        container.tels_with_data = [camera_config.telescope_id, ]
        svc_container = container.tel[camera_config.telescope_id].svc

        svc_container.telescope_id = camera_config.telescope_id
        svc_container.cs_serial = camera_config.cs_serial
        svc_container.configuration_id = camera_config.configuration_id
        #svc_container.acquisition_mode = camera_config.acquisition_mode
        svc_container.date = camera_config.date
        svc_container.num_pixels = camera_config.num_pixels
        svc_container.num_samples = camera_config.num_samples
        svc_container.pixel_ids = camera_config.expected_pixels_id
        svc_container.data_model_version = camera_config.data_model_version

        svc_container.num_modules = camera_config.lstcam.num_modules
        svc_container.module_ids = camera_config.lstcam.expected_modules_id
        svc_container.idaq_version = camera_config.lstcam.idaq_version
        svc_container.cdhs_version = camera_config.lstcam.cdhs_version
        svc_container.algorithms = camera_config.lstcam.algorithms
        svc_container.pre_proc_algorithms = camera_config.lstcam.pre_proc_algorithms




    def fill_lst_event_container_from_zfile(self, container, event):

        event_container = container.tel[self.camera_config.telescope_id].evt

        event_container.configuration_id = event.configuration_id
        event_container.event_id = event.event_id
        event_container.tel_event_id = event.tel_event_id
        event_container.pixel_status = event.pixel_status
        event_container.ped_id = event.ped_id
        event_container.module_status = event.lstcam.module_status
        event_container.extdevices_presence = event.lstcam.extdevices_presence
        event_container.tib_data = event.lstcam.tib_data
        event_container.cdts_data = event.lstcam.cdts_data
        event_container.swat_data = event.lstcam.swat_data
        event_container.counters = event.lstcam.counters


    def fill_r0_camera_container_from_zfile(self, container, event):

        container.num_samples = self.camera_config.num_samples
        container.trigger_time = event.trigger_time_s
        container.trigger_type = event.trigger_type

        # verify the number of gains
        if event.waveform.shape[0] == (self.camera_config.num_pixels *
                                       container.num_samples):
            n_gains = 1
        elif event.waveform.shape[0] == (self.camera_config.num_pixels *
                                         container.num_samples * 2):
            n_gains = 2
        else:
            raise ValueError("Waveform matrix dimension not supported: "
                             "N_chan x N_pix x N_samples= '{}'"
                             .format(event.waveform.shape[0]))


        container.waveform = np.array(
            (
                event.waveform
            ).reshape(n_gains, self.camera_config.num_pixels, container.num_samples))


    def fill_r0_container_from_zfile(self, container, event):
        container.obs_id = -1
        container.event_id = event.event_id

        container.tels_with_data = [self.camera_config.telescope_id, ]
        r0_camera_container = container.tel[self.camera_config.telescope_id]
        self.fill_r0_camera_container_from_zfile(
            r0_camera_container,
            event
        )

