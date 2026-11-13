import unittest
from unittest.mock import patch,Mock
from okno.http_range import RangeReader


class RangeReaderTests(unittest.TestCase):
    def fixture(self,url,headers,timeout):
        a,b=map(int,headers['Range'].split('=')[1].split('-'))
        b=min(b,len(self.data)-1)
        r=Mock(status_code=206,headers={'Content-Range':f'bytes {a}-{b}/{len(self.data)}'},content=self.data[a:b+1])
        r.__enter__=Mock(return_value=r);r.__exit__=Mock(return_value=False)
        return r
    def test_seek_read_cache_and_eof(self):
        self.data=b'abcdefghij'
        with patch('requests.Session.get',side_effect=self.fixture) as call:
            with RangeReader('https://example.org/test.tif') as f:
                self.assertEqual(f.read(3),b'abc');f.seek(-2,1);self.assertEqual(f.read(4),b'bcde')
                f.seek(-2,2);self.assertEqual(f.read(9),b'ij');self.assertEqual(f.read(),b'')
                self.assertEqual(call.call_count,2)
    def test_partial_response_required(self):
        r=Mock(status_code=200);r.__enter__=Mock(return_value=r);r.__exit__=Mock(return_value=False)
        with patch('requests.Session.get',return_value=r):
            with self.assertRaises(ValueError):RangeReader('https://example.org/test.tif')
        with self.assertRaises(FileNotFoundError):RangeReader('test/test')


if __name__=='__main__':unittest.main()
