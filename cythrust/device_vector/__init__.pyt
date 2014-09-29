{% for ctype, dtype in DEVICE_VECTOR_TYPES -%}
from .{{ dtype[3:] }} import DeviceVector as DeviceVector{{ dtype[3:].title() }}
{% endfor %}
