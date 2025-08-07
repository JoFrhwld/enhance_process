import click
import librosa
from df import enhance, init_df
from df.enhance import load_audio, save_audio
from torch import tensor
import numpy as np
from joblib import Parallel, delayed, cpu_count
from pathlib import Path
from tqdm import tqdm

MODEL, DF_STATE, _ = init_df()
NCPU = int(cpu_count()/2)

def load_and_process(path, df_state = DF_STATE):
    audio, sr = librosa.load(str(path), sr = df_state.sr())
    audio_framed = librosa.util.frame(audio, frame_length=2048, hop_length=2048)

    def get_perc(y):
        return librosa.effects.percussive(y)

    results = Parallel(n_jobs=NCPU)(
        delayed(get_perc)(audio_framed[:,i]) 
        for i in tqdm(range(audio_framed.shape[1]))
    )

    percussive = np.concatenate(results)
    percussive_pad = np.pad(percussive, (0, audio.size - percussive.size))

    audio_dampened = audio - (percussive_pad * 0.6)
    rms = librosa.feature.rms(y = audio_dampened).mean()

    to_raise = np.power(10, 3.5 - (np.log10(rms/0.00002)))
    audio_amp = audio_dampened * to_raise

    audio_t = tensor(audio_amp.reshape((1, audio_amp.size)))

    return audio_t


def enhance_audio(audio_t, model = MODEL, df_state = DF_STATE):
    enhanced = enhance(model, df_state, audio_t, atten_lim_db = 25)
    return enhanced

def write_audio(enhanced, out_path):
    print(out_path)
    save_audio(file = out_path, audio = enhanced, sr = DF_STATE.sr())


@click.command()
@click.argument(
    "path",
    type = click.Path(path_type=Path)
)
def main(path:Path):
    #print(list(Path("corpus2").glob("*")))
    loc = path.parent
    out = loc.joinpath("enhanced")
    if not out.exists():
        out.mkdir()
    
    out_file = out.joinpath(path.name)

    audio_t = load_and_process(str(path))
    audio_e = enhance_audio(audio_t)
    write_audio(audio_e, str(out_file))

if __name__ == "__main__":
    main()
