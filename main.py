import logging
import warnings

warnings.filterwarnings(
    action='ignore'
)

import click
import librosa
from df import enhance, init_df
import soundfile
from torch import tensor, Tensor
import numpy as np
import numpy.typing as npt
from joblib import Parallel, delayed, cpu_count
from pathlib import Path
from tqdm import tqdm



logger = logging.getLogger("enhance")
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.handlers.clear()
logger.addHandler(handler)
logger.setLevel(logging.INFO)

logger.info("Initial setup")
MODEL, DF_STATE, _ = init_df(log_level="DEBUG")
SR = DF_STATE.sr()
FRAME = 38400
HOP = 19200
NCPU = int(cpu_count()/2)

def load_and_process(path:Path|str, perc_damp:float, df_state = DF_STATE) -> Tensor:
    logger.info("Loading and framing audio")
    audio, sr = librosa.load(str(path), sr = df_state.sr())
    audio_framed = librosa.util.frame(audio, frame_length=FRAME, hop_length=HOP)
    logger.info(f"Audio shape: {audio_framed.shape}")

    def get_perc(y:npt.NDArray)->npt.NDArray:
        return librosa.effects.percussive(y)

    logger.info("Dampening percussives")    
    results = Parallel(n_jobs=NCPU)(
        delayed(get_perc)(audio_framed[:,i]) 
        for i in tqdm(range(audio_framed.shape[1]), bar_format='{desc}: {percentage:3.0f}%')
    )

    percussive = np.array(results).T
    audio_dampened = audio_framed - (percussive * perc_damp)

    logger.info("Rescaling audio intensity")
    rms = np.median(librosa.feature.rms(y = audio_dampened))
    to_raise = np.power(10, 3.5 - (np.log10(rms/0.00002)))
    audio_amp = audio_dampened * to_raise
    audio_t = tensor(audio_amp)
    audio_t = audio_t[None, :, :]

    return audio_t


def enhance_audio(audio_t:Tensor, atten_db: float, model = MODEL, df_state = DF_STATE)->npt.NDArray:
    logger.info("Starting Enhancement")
    results = [
        enhance(model = MODEL, df_state=DF_STATE, audio = audio_t[:,:,idx], atten_lim_db=atten_db).numpy().squeeze()
        for idx in tqdm(range(audio_t.shape[-1]), bar_format='{desc}: {percentage:3.0f}%')
    ]
    enhanced = np.array(results).T
    logger.info(f"Enhanced shape: {enhanced.shape}")
    return enhanced

def write_audio(enhanced:npt.NDArray, out_path:Path|str)->None:
    logger.info("Writing Audio")
    NFRAMES = enhanced.shape[1]
    hann = librosa.filters.get_window("hann", Nx = FRAME).reshape((-1, 1))
    enhanced = enhanced * hann
    out_mat = np.empty((FRAME, FRAME + (HOP * (NFRAMES-1))))
    out_mat[:] = np.nan
    logger.info("Filling matrix")
    for i in range(enhanced.shape[1]):
        idx = int(i * HOP)
        end_idx = idx + FRAME
        out_mat[
            i,
            idx:end_idx
        ] = enhanced[:,i]
    logger.info("Normalizing")
    out_arr = np.nansum(out_mat, axis = 0)
    soundfile.write(file = str(out_path), data = out_arr, samplerate=SR)
    logger.info("Sound written")

@click.command()
@click.argument(
    "path",
    type = click.Path(path_type=Path)
)
@click.option(
    "--perc_damp",
    type = click.FloatRange(
        min = 0,
        max = 1,
        clamp= True
    ),
    default = 0.45
)
@click.option(
    "--atten_db",
    type = click.FloatRange(
        min = 0,
        max = 100,
        clamp = True
    ),
    default = 30
)
def main(path:Path, perc_damp, atten_db):
    loc = path.parent
    out = loc.joinpath("enhanced")
    if not out.exists():
        logger.info(f"Creating ouput directory at {str(out)}")
        out.mkdir(parents=True)
    out_file = out.joinpath(path.name)
    audio_t = load_and_process(str(path), perc_damp = perc_damp)
    audio_e = enhance_audio(audio_t, atten_db = atten_db)
    write_audio(audio_e, str(out_file))

if __name__ == "__main__":

    main()
