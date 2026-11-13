"""Каталог покрытия: прогнозные точки отделены от спутникового расширения."""
from .flood import POINTS

EXTRA_SITES={
    'irtysh_omsk':dict(region='Омская область',river='Иртыш',name='Омск',lat=55.00222,lon=73.32482,gauge_code='11048'),
    'tom_tomsk':dict(region='Томская область',river='Томь',name='Томск',lat=56.447785,lon=84.960621,gauge_code='10251'),
}
SITES={**POINTS,**EXTRA_SITES}
