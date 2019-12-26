import io
import librosa
import numpy as np
from pydub import AudioSegment

import sox
import tempfile

def resample(data,file_format,fs, mono=True):

    ext = file_format

    with tempfile.NamedTemporaryFile(suffix='.'+ext) as temp:
        temp.write(data)
        temp.flush()

        with tempfile.NamedTemporaryFile(suffix='.'+ext) as tmp:
            tfm = sox.Transformer()
            tfm.rate(fs,quality='v')
            tfm.remix(remix_dictionary=None,num_output_channels=1)
            tfm.build(temp.name,tmp.name)
            X = AudioSegment.from_file(tmp.name,ext)

    x = np.asarray(X.get_array_of_samples()).astype('float'+str(8*X.sample_width))
    x = x/float(1 << ((8*X.sample_width)-1))

    return x
