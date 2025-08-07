import click
import librosa
from df import enhance, init_df
from df.enhance import load_audio, save_audio
from torch import tensor, Tensor
import numpy as np
import numpy.typing as npt
from joblib import Parallel, delayed, cpu_count
from pathlib import Path
from tqdm import tqdm
import logging

logger = logging.getLogger("enhance")
logger.setLevel(logging.DEBUG)

logger.info("Initial setup")
MODEL, DF_STATE, _ = init_df(log_level="DEBUG")
NCPU = int(cpu_count()/2)

def load_and_process(path:Path|str, df_state = DF_STATE) -> Tensor:
    logger.info("Loading and framing audio")
    audio, sr = librosa.load(str(path), sr = df_state.sr())
    audio_framed = librosa.util.frame(audio, frame_length=2048, hop_length=2048)

    def get_perc(y:npt.NDArray)->npt.NDArray:
        return librosa.effects.percussive(y)

    logger.info("Dampening percussives")    
    results = Parallel(n_jobs=NCPU)(
        delayed(get_perc)(audio_framed[:,i]) 
        for i in tqdm(range(audio_framed.shape[1]))
    )

    percussive = np.concatenate(results)
    percussive_pad = np.pad(percussive, (0, audio.size - percussive.size))
    audio_dampened = audio - (percussive_pad * 0.6)

    logger.info("Rescaling audio intensity")
    rms = librosa.feature.rms(y = audio_dampened).mean()
    to_raise = np.power(10, 3.5 - (np.log10(rms/0.00002)))
    audio_amp = audio_dampened * to_raise
    audio_t = tensor(audio_amp.reshape((1, audio_amp.size)))

    return audio_t


def enhance_audio(audio_t:Tensor, model = MODEL, df_state = DF_STATE)->Tensor:
    logger.info("Starting Enhancement")
    enhanced = enhance(model = MODEL,  df_state=DF_STATE, audio = audio_t, atten_lim_db = 25)
    return enhanced

def write_audio(enhanced:Tensor, out_path:Path|str)->None:
    logger.info("Writing Audio")
    save_audio(file = str(out_path), audio = enhanced, sr = DF_STATE.sr())


@click.command()
@click.argument(
    "path",
    type = click.Path(path_type=Path)
)
def main(path:Path):
    loc = path.parent
    out = loc.joinpath("enhanced")
    if not out.exists():
        out.mkdir()
 
    out_file = out.joinpath(path.name)
    print(NCPU)
    audio_t = load_and_process(str(path))
    audio_e = enhance_audio(audio_t)
    write_audio(audio_e, str(out_file))

if __name__ == "__main__":
    main()
