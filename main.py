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
import logging
import warnings

warnings.filterwarnings(
    action='ignore',
    category=UserWarning
)

logger = logging.getLogger("enhance")
logging.basicConfig(level = logging.INFO)

logger.info("Initial setup")
MODEL, DF_STATE, _ = init_df(log_level="DEBUG")
NCPU = int(cpu_count()/2)

def load_and_process(path:Path|str, perc_damp:float, df_state = DF_STATE) -> Tensor:
    logger.info("Loading and framing audio")
    audio, sr = librosa.load(str(path), sr = df_state.sr())
    audio_framed = librosa.util.frame(audio, frame_length=int(sr), hop_length=int(sr))

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
        enhance(model = MODEL, df_state=DF_STATE, audio = audio_t[:,:,idx], atten_lim_db=atten_db).numpy()
        for idx in tqdm(range(audio_t.shape[-1]), bar_format='{desc}: {percentage:3.0f}%')
    ]
    enhanced = np.array(results).T.reshape(-1, order = "F")
    return enhanced

def write_audio(enhanced:npt.NDArray, out_path:Path|str)->None:
    logger.info("Writing Audio")
    soundfile.write(file = str(out_path), data = enhanced, samplerate=DF_STATE.sr())

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
