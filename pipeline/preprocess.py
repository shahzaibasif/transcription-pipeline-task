from pydub import AudioSegment
def preprocess_audio(input_file,output_file="temp.wav"):
    audio=AudioSegment.from_file(input_file).set_channels(1).set_frame_rate(16000)
    audio.export(output_file,format="wav")
    return output_file
