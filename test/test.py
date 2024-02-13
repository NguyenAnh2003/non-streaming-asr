# import torchaudio
#
# audio, sample = torchaudio.load("hehe.wav")
#
# print(f"Audio array: {audio} Sample: {sample}")
from logger.my_logger import setup_logger

logger = setup_logger("test")
logger.getLogger(__name__)
logger.info("HELLO")