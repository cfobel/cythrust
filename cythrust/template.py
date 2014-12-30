SORT_TEMPLATE = '''
from cythrust.thrust.sort cimport {% if stable %}stable_{% endif %}sort_by_key as c_sort_by_key, {% if stable %}stable_{% endif %}sort as c_sort
from cythrust.thrust.iterator.zip_iterator cimport make_zip_iterator
{% for m in key_modules -%}
from {{ m }}.device_vector cimport DeviceVectorView as {{ key_dtypes[loop.index0].__name__.title() }}View
{% endfor %}
{% for m in value_modules -%}
from {{ m }}.device_vector cimport DeviceVectorView as {{ value_dtypes[loop.index0].__name__.title() }}View
{% endfor %}
{%- if key_modules|length > 1 %}
from cythrust.thrust.tuple cimport make_tuple{{ key_modules|length }}  # keys
{%- endif %}
{%- if value_modules|length > 1 %}
from cythrust.thrust.tuple cimport make_tuple{{ value_modules|length }}  # values
{%- endif %}

{% if value_modules|length > 0 %}
def sort_func(
        {% for d in key_dtypes -%}
        {{ d.__name__.title() }}View keys{{ loop.index }},
        {% endfor -%}
        {% for d in value_dtypes -%}
        {{ d.__name__.title() }}View values{{ loop.index }}
        {%- if not loop.last %},{% endif %}
        {% endfor -%}):

    c_sort_by_key(
        {% if key_modules|length > 1 -%}
        make_zip_iterator(make_tuple{{ key_modules|length }}(
        {%- endif %}
        {% for c in key_modules -%}
        keys{{ loop.index }}._vector.begin()
        {%- if not loop.last %}, {% endif -%}
        {% endfor -%}
        {% if key_modules|length > 1 -%}
        ))
        {%- endif %},
        {% if key_modules|length > 1 -%}
        make_zip_iterator(make_tuple{{ key_modules|length }}(
        {%- endif %}
        {% for k in key_modules -%}
        keys{{ loop.index }}._vector.end()
        {%- if not loop.last %}, {% endif -%}
        {% endfor -%}
        {% if key_modules|length > 1 -%}
        ))
        {%- endif %},

        {% if value_modules|length > 1 -%}
        make_zip_iterator(make_tuple{{ value_modules|length }}(
        {%- endif -%}
        {%- for v in value_modules %}
        values{{ loop.index }}._vector.begin()
        {%- if not loop.last %}, {% endif -%}
        {% endfor -%}
        {% if value_modules|length > 1 -%}
        ))
        {%- endif %})
{% else %}
def sort_func(
        {% for d in key_dtypes -%}
        {{ d.__name__.title() }}View keys{{ loop.index }}
        {%- if not loop.last %},{% endif %}  # {{ c }}
        {% endfor -%}):

    c_sort(
        {% if key_modules|length > 1 -%}
        make_zip_iterator(make_tuple{{ key_modules|length }}(
        {%- endif %}
        {% for k in key_dtypes -%}
        keys{{ loop.index }}._vector.begin()
        {%- if not loop.last %}, {% endif -%}
        {% endfor -%}
        {% if key_modules|length > 1 -%}
        ))
        {%- endif %},
        {% if key_modules|length > 1 -%}
        make_zip_iterator(make_tuple{{ key_modules|length }}(
        {%- endif %}
        {% for k in key_dtypes -%}
        keys{{ loop.index }}._vector.end()
        {%- if not loop.last %}, {% endif -%}
        {% endfor -%}
        {% if key_modules|length > 1 -%}
        ))
        {%- endif %})
{%- endif %}
'''