{% for ctype, dtype in DEVICE_VECTOR_TYPES -%}
from .{{ dtype[3:] }}.device_vector import (DeviceVector as DeviceVector{{ dtype[3:].title() }},
                              DeviceVectorView as DeviceVectorView{{ dtype[3:].title() }})
{% endfor %}
