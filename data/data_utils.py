from torchaudio._internal import download_url_to_file
from torchaudio.datasets.utils import _extract_tar
import os

URL = "dev-clean"

FOLDER_IN_ARCHIVE = "LibriSpeech"

SAMPLE_RATE = 16000

_DATA_SUBSETS = [
    "dev-clean",
    "dev-other",
    "test-clean",
    "test-other",
    "train-clean-100",
    "train-clean-360",
    "train-other-500",
]

_CHECKSUMS = {
    "http://www.openslr.org/resources/12/dev-clean.tar.gz": "76f87d090650617fca0cac8f88b9416e0ebf80350acb97b343a85fa903728ab3",  # noqa: E501
    "http://www.openslr.org/resources/12/dev-other.tar.gz": "12661c48e8c3fe1de2c1caa4c3e135193bfb1811584f11f569dd12645aa84365",  # noqa: E501
    "http://www.openslr.org/resources/12/test-clean.tar.gz": "39fde525e59672dc6d1551919b1478f724438a95aa55f874b576be21967e6c23",  # noqa: E501
    "http://www.openslr.org/resources/12/test-other.tar.gz": "d09c181bba5cf717b3dee7d4d592af11a3ee3a09e08ae025c5506f6ebe961c29",  # noqa: E501
    "http://www.openslr.org/resources/12/train-clean-100.tar.gz": "d4ddd1d5a6ab303066f14971d768ee43278a5f2a0aa43dc716b0e64ecbbbf6e2",  # noqa: E501
    "http://www.openslr.org/resources/12/train-clean-360.tar.gz": "146a56496217e96c14334a160df97fffedd6e0a04e66b9c5af0d40be3c792ecf",  # noqa: E501
    "http://www.openslr.org/resources/12/train-other-500.tar.gz": "ddb22f27f96ec163645d53215559df6aa36515f26e01dd70798188350adcb6d2",  # noqa: E501
}


def download_libirspeech_dataset(root: str = f"D:\\", url: str = URL):
    """ auto download librispeech dataset """
    # base url
    base_url = "http://www.openslr.org/resources/12/"
    # extension file
    extension = ".tar.gz"

    # filename with url(openslr url) + extension file
    filename = url + extension
    archive = os.path.join(root, filename) # store tar file in archive folder
    download_url = os.path.join(base_url, filename)
    if not os.path.isfile(archive):
        checksum = _CHECKSUMS.get(download_url, None)
        download_url_to_file(download_url, archive) # download tar file
    _extract_tar(archive) # extract archive Libirspeech folder?

def get_librispeech_metadata(fileid: str, root: str, folder: str,
                             ext_audio: str, ext_text: str):

    pass

if __name__ == "__main__":

    print("DONE")