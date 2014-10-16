import numpy as np

{% for ctype, dtype in DEVICE_VECTOR_TYPES -%}
from .{{ dtype[3:] }}.device_vector import (DeviceVector as DeviceVector{{ dtype[3:].title() }},
                              DeviceVectorView as DeviceVectorView{{ dtype[3:].title() }})
{% endfor %}


def from_array(data):
{%- for ctype, dtype in DEVICE_VECTOR_TYPES %}
    if data.dtype == np.bool:
        return DeviceVectorUint8.from_array(data)
    {% if loop.first %}if{% else %}elif{% endif %} data.dtype == {{ dtype }}:
        return DeviceVector{{ dtype[3:].title() }}.from_array(data)
{% endfor %}
    raise ValueError('Unsupported type: %r' % data)


def view_from_vector(vector, first_i=0, last_i=-1):
{%- for ctype, dtype in DEVICE_VECTOR_TYPES %}
    {% if loop.first %}if{% else %}elif{% endif %} vector.dtype == {{ dtype }}:
        return DeviceVectorView{{ dtype[3:].title() }}(vector, first_i, last_i)
{% endfor %}
    raise ValueError('Unsupported type: %r' % vector)
