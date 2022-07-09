# -*- coding: utf-8 -*-
"""
.. codeauthor:: Alexander Vorndran <alexander.vorndran@tu-ilmenau.de>
.. codeauthor:: Söhnke Fischedick <soehnke.fischedick@tu-ilmenau.de>
"""
# pylint: disable=no-member
import pytest
import numpy as np

from ...hash_function import numpy_hash_normalized, js_hash


@pytest.mark.parametrize("shape, expected_digest", [
    ( 1, "34eb8956007a364690fedc311d9b60e59141ea5d56e26f6cff67fd5334cb84ae"),  # noqa: E201
    ( 2, "84b42942d54de3ad1576a22d917e9c0d1ab04a1a0c61bcfaeaeab98e327d3fa9"),  # noqa: E201
    ( 3, "acea31062db1c5cc518f7af3e2a2b6b3c774ae9c49d2604fbbe208b938a45476"),  # noqa: E201
    ( 4, "af4807d9bbfcc35ff194fccdf56ea498924254c65135cd962d3e4750dc09372c"),  # noqa: E201
    ( 5, "cae2b6be7938de06ec72dd715f35400712570f1d6f934c96f136d6d8914a96bc"),  # noqa: E201
    ( 6, "dbee6111efa99464a03cff0f05ed6e5ae3301b158bd4f38bb6f1b77e6bec0d87"),  # noqa: E201
    ( 7, "1c2504ff6f609569959c33e155613e644fbc7b36fcc8385b818090716892a6c9"),  # noqa: E201
    ( 8, "262e39dba0b920a67d44e65debf14d0c47fdcefceb81290d935457aca92d2379"),  # noqa: E201
    ( 9, "edbc1f05066a804787bb1513985ca6678738dcd4096ea217c660c5805a7877a2"),  # noqa: E201
    (10, "39f70cb55e370c61934e41185abc3f64a9497dbcf42730c1151ff49a3d9837a7"),
    (11, "3d12a3fd4746307d2ae1dd989ed435c8346ac2764428927ea3d40bdc3b21cf36"),
    (12, "c0812bddbb39cb1389fcedc3baf6aa34435781f792f5817358ad8db05f2b602d"),
    (13, "a087cb1b464203144ee2c44ce3c49ea5c3df31b2ebd3b89bbc8f507351db59be"),
    (14, "f04dee9f7640401036b9ea90dd4910c9a157c89de568a11da23d725b406d1685"),
    (15, "c49d4d2e6d088cb3072147abded90293a80f8bd6f0a9e084af6092f0054c1c1d"),
    (16, "17d5824b151f197885398d7139de771ebf54113bd6587b5d2f944f44308feebf"),
    (17, "f14af57b116670512bbded869b019b5c1415f34fec57b1ad6752013c3d77085f"),
    (18, "e2862f5690416266aa49c1327b430a389061b911af6f327a229568a6868c7d57"),
    (19, "5266d69c4127883b27164b5321dd6ebf09ffb3e1fa553c9ef8e64147f978b0e8"),
    (20, "a83f85231656c2dbd8d0c6f5343a8d3f6a1e2476f1e539acf78f88aab055f7f2"),
])
def test_squared_numpy_hash_normalized(shape, expected_digest):
    """Test numpy_hash_normalized with a fixed number of square inputs"""
    arr = np.arange(shape*shape).reshape((shape, shape))
    np_digest = numpy_hash_normalized(arr, as_jshash=False)
    assert np_digest == expected_digest


@pytest.mark.parametrize("shape, expected_digest", [
    ( 1, 4097823119),   # noqa: E201
    ( 2, 55494066),     # noqa: E201
    ( 3, 614328915),    # noqa: E201
    ( 4, 70592091),     # noqa: E201
    ( 5, 3555986139),   # noqa: E201
    ( 6, 3674687909),   # noqa: E201
    ( 7, 2964968623),   # noqa: E201
    ( 8, 2002134017),   # noqa: E201
    ( 9, 928509417),    # noqa: E201
    (10, 2179325499),
    (11, 568119425),
    (12, 3084267280),
    (13, 1080921143),
    (14, 883147978),
    (15, 534979880),
    (16, 1208374966),
    (17, 3867780471),
    (18, 3877732852),
    (19, 1652670006),
    (20, 2441977209),
])
def test_squared_js_hash(shape, expected_digest):
    """Test numpy_hash_normalized with a fixed number of square inputs"""
    arr = np.arange(shape*shape).reshape((shape, shape))
    np_digest = js_hash(arr)
    assert np_digest == expected_digest
