"""Tests port-level amplitude/sampling-rate resolution in ``QmController``."""

import pytest

pytest.importorskip("qm")

from qibolab._core.components import (
    AcquisitionChannel,
    DcChannel,
    IqChannel,
    IqConfig,
    OscillatorConfig,
)
from qibolab._core.instruments.qm import Octave, QmController
from qibolab._core.instruments.qm.components import (
    MwFemOscillatorConfig,
    OpxOutputConfig,
    QmAcquisitionConfig,
)
from qibolab._core.instruments.qm.controller import (
    MAX_AMPLITUDE_MW_FEM,
    MAX_VOLTAGE,
    MAX_VOLTAGE_AMPLIFIED,
    channel_max_voltage,
    channel_sampling_rate,
)
from qibolab._core.pulses import Acquisition, Gaussian, Pulse, Readout, Rectangular


@pytest.fixture
def controller() -> QmController:
    channels = {
        "mw/drive": IqChannel(device="con1/1", path="1", lo="mw/lo", mixer=None),
        "mw/probe": IqChannel(device="con1/1", path="2", lo="mw/probe_lo", mixer=None),
        "mw/acquisition": AcquisitionChannel(
            device="con1/1", path="1", probe="mw/probe"
        ),
        "lf/direct": DcChannel(device="con1/2", path="1"),
        "lf/amplified": DcChannel(device="con1/2", path="2"),
        "oct/drive": IqChannel(device="oct1", path="1", lo="oct/lo", mixer=None),
    }
    return QmController(
        address="0.0.0.0:0",
        channels=channels,
        octaves={"oct1": Octave(name="oct1", port=11000, connectivity="con2")},
        fems={"con1/1": "MW", "con1/2": "LF", "con2": "opx1"},
    )


@pytest.fixture
def configs() -> dict:
    return {
        "mw/lo": MwFemOscillatorConfig(frequency=5e9, power=4, sampling_rate=2e9),
        "mw/drive": IqConfig(frequency=5.1e9),
        "mw/probe_lo": MwFemOscillatorConfig(frequency=7e9, power=-5),
        "mw/probe": IqConfig(frequency=7.05e9),
        "mw/acquisition": QmAcquisitionConfig(delay=200, smearing=0),
        "lf/direct": OpxOutputConfig(output_mode="direct"),
        "lf/amplified": OpxOutputConfig(output_mode="amplified"),
        "oct/lo": OscillatorConfig(frequency=5e9, power=0),
        "oct/drive": IqConfig(frequency=5.1e9),
    }


def test_channel_max_voltage_per_module():
    assert channel_max_voltage(MwFemOscillatorConfig(frequency=5e9, power=4)) == (
        MAX_AMPLITUDE_MW_FEM
    )
    assert channel_max_voltage(OpxOutputConfig(output_mode="direct")) == MAX_VOLTAGE
    assert (
        channel_max_voltage(OpxOutputConfig(output_mode="amplified"))
        == MAX_VOLTAGE_AMPLIFIED
    )
    assert channel_max_voltage(IqConfig(frequency=5e9)) == MAX_VOLTAGE


@pytest.mark.parametrize(
    "channel,expected",
    [
        ("mw/drive", "mw/lo"),
        ("mw/probe", "mw/probe_lo"),
        ("mw/acquisition", "mw/probe_lo"),
        ("lf/direct", "lf/direct"),
        ("oct/drive", "oct/drive"),
    ],
)
def test_port_config(controller, configs, channel, expected):
    assert controller.port_config(channel, configs) is configs[expected]


def _registered_waveforms(controller, op):
    pulse = controller.config.pulses[op]
    names = pulse.waveforms
    if isinstance(names, dict):
        return [controller.config.waveforms[names["single"]]]
    return [controller.config.waveforms[names.I], controller.config.waveforms[names.Q]]


@pytest.mark.parametrize(
    "channel,full_scale",
    [
        ("mw/drive", MAX_AMPLITUDE_MW_FEM),
        ("lf/direct", MAX_VOLTAGE),
        ("lf/amplified", MAX_VOLTAGE_AMPLIFIED),
        ("oct/drive", MAX_VOLTAGE),
    ],
)
def test_constant_pulse_amplitude_maps_to_port_full_scale(
    controller, configs, channel, full_scale
):
    controller.configure_channel(channel, configs)
    pulse = Pulse(duration=40, amplitude=0.3, envelope=Rectangular())
    op = controller.register_pulse(channel, configs, pulse)
    i = _registered_waveforms(controller, op)[0]
    assert i.sample == pytest.approx(0.3 * full_scale)


def test_mw_fem_arbitrary_waveform_uses_port_sampling_rate(controller, configs):
    """A 2 GSps MW-FEM port needs 2 samples per ns - read from the LO config,
    since the channel's own IqConfig has no sampling rate."""
    assert channel_sampling_rate(controller.port_config("mw/drive", configs)) == 2
    controller.configure_channel("mw/drive", configs)
    pulse = Pulse(duration=40, amplitude=1.0, envelope=Gaussian(rel_sigma=0.2))
    op = controller.register_pulse("mw/drive", configs, pulse)
    i, _ = _registered_waveforms(controller, op)
    assert len(i.samples) == 80
    assert max(i.samples) == pytest.approx(MAX_AMPLITUDE_MW_FEM, rel=1e-3)


def test_mw_fem_readout_uses_probe_port(controller, configs):
    controller.configure_channel("mw/acquisition", configs)
    probe = Pulse(duration=100, amplitude=0.4, envelope=Rectangular())
    readout = Readout(acquisition=Acquisition(duration=100), probe=probe)
    op = controller.config.register_acquisition_pulse(
        "mw/acquisition",
        readout,
        channel_sampling_rate(controller.port_config("mw/acquisition", configs)),
        channel_max_voltage(controller.port_config("mw/acquisition", configs)),
    )
    acquisition = controller.config.elements["mw/acquisition"].operations[op]
    names = controller.config.pulses[acquisition].waveforms
    samples = controller.config.waveforms[names.I].samples
    assert max(samples) == pytest.approx(0.4 * MAX_AMPLITUDE_MW_FEM)
