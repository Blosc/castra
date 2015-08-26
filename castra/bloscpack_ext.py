from io import BytesIO

import numpy as np
import blosc
from pandas.msgpack import Packer, unpackb, packb
from bloscpack.abstract_io import PlainSink, pack, unpack
from bloscpack.args import calculate_nchunks, BloscArgs
from bloscpack.numpy_io import PlainNumpySource
from bloscpack.file_io import (CompressedFPSink, CompressedFPSource,
                               _read_metadata)
from bloscpack.append import append_fp, _rewrite_metadata_fp


pack_array_header = Packer().pack_array_header


class ObjectSeriesSink(PlainSink):
    def __init__(self, metadata, encoding='utf8'):
        self.metadata = metadata
        self.encoding = encoding
        length = metadata['length']
        nbytes = metadata['nbytes']
        header = pack_array_header(length)
        head_size = len(header)
        self.buffer = np.empty(nbytes + head_size, 'c')
        self.buffer[:head_size] = header
        self.ptr = self.buffer[head_size:].__array_interface__['data'][0]

    def put(self, compressed):
        bwritten = blosc.decompress_ptr(compressed, self.ptr)
        self.ptr += bwritten

    @property
    def ndarray(self):
        data = unpackb(self.buffer.tobytes(), encoding=self.encoding)
        return np.array(data, object, copy=False)


class ObjectSeriesSource(PlainNumpySource):
    def __init__(self, series, encoding='utf8'):
        length = len(series)
        head_size = len(pack_array_header(length))
        self.ndarray = np.fromstring(packb(series.tolist(),
                                     encoding=encoding), 'c')[head_size:]
        self.size = len(self.ndarray)
        self.metadata = {u'length': length, u'nbytes': self.size}
        self.ptr = self.ndarray.__array_interface__['data'][0]


def pack_object_series(series, sink, encoding='utf8', chunk_size='1M',
                       blosc_args=None, bloscpack_args=None,
                       metadata_args=None):
    if blosc_args is None:
        blosc_args = BloscArgs(typesize=1)
    else:
        blosc_args.typesize = 1
    source = ObjectSeriesSource(series, encoding=encoding)
    nchunks, chunk_size, last_chunk_size = \
        calculate_nchunks(source.size, chunk_size)
    pack(source, sink, nchunks, chunk_size, last_chunk_size,
         metadata=source.metadata, blosc_args=blosc_args,
         bloscpack_args=bloscpack_args, metadata_args=metadata_args)


def pack_object_series_file(series, fn, encoding='utf8', chunk_size='1M',
                            blosc_args=None, bloscpack_args=None,
                            metadata_args=None):
    with open(fn, 'wb') as fp:
        sink = CompressedFPSink(fp)
        pack_object_series(series, sink, chunk_size=chunk_size,
                           encoding=encoding, blosc_args=blosc_args,
                           bloscpack_args=bloscpack_args,
                           metadata_args=metadata_args)


def append_object_series_file(series, fn, encoding='utf8'):
    length = len(series)
    head_size = len(pack_array_header(length))
    bytes = packb(series.tolist(), encoding=encoding)[head_size:]
    nbytes = len(bytes)
    with open(fn, 'rb+') as fil:
        append_fp(fil, BytesIO(bytes), nbytes)
        fil.seek(32)
        meta = _read_metadata(fil)[0]
        meta[u'length'] += length
        meta[u'nbytes'] += nbytes
        fil.seek(32)
        _rewrite_metadata_fp(fil, meta)


def unpack_object_series(source, encoding='utf8'):
    sink = ObjectSeriesSink(source.metadata, encoding)
    unpack(source, sink)
    return sink.ndarray


def unpack_object_series_file(fn, encoding='utf8'):
    with open(fn, 'rb') as fil:
        source = CompressedFPSource(fil)
        return unpack_object_series(source, encoding)
