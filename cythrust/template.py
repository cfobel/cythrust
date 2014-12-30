SORT_TEMPLATE = '''
from cythrust.thrust.sort cimport sort_by_key as c_sort_by_key, sort as c_sort
from cythrust.thrust.iterator.zip_iterator cimport make_zip_iterator
{% for c in key_columns + value_columns -%}
from {{ df.get_vector_module(c) }}.device_vector cimport DeviceVectorView as {{ df.get_dtype(c).__name__.title() }}View
{% endfor %}
{%- if key_columns|length > 1 %}
from cythrust.thrust.tuple cimport make_tuple{{ key_columns|length }}  # keys
{%- endif %}
{%- if value_columns|length > 1 %}
from cythrust.thrust.tuple cimport make_tuple{{ value_columns|length }}  # values
{%- endif %}

{% if value_columns|length > 0 %}
def sort_func(
        {% for c in key_columns -%}
        {{ df.get_dtype(c).__name__.title() }}View keys{{ loop.index }},  # {{ c }}
        {% endfor -%}
        {% for c in value_columns -%}
        {{ df.get_dtype(c).__name__.title() }}View values{{ loop.index }}
        {%- if not loop.last %},{% endif %}  # {{ c }}
        {% endfor -%}):

    c_sort_by_key(
        {% if key_columns|length > 1 -%}
        make_zip_iterator(make_tuple{{ key_columns|length }}(
        {%- endif %}
        {% for c in key_columns -%}
        keys{{ loop.index }}._vector.begin()
        {%- if not loop.last %}, {% endif -%}
        {% endfor -%}
        {% if key_columns|length > 1 -%}
        ))
        {%- endif %},
        {% if key_columns|length > 1 -%}
        make_zip_iterator(make_tuple{{ key_columns|length }}(
        {%- endif %}
        {% for k in key_columns -%}
        keys{{ loop.index }}._vector.end()
        {%- if not loop.last %}, {% endif -%}
        {% endfor -%}
        {% if key_columns|length > 1 -%}
        ))
        {%- endif %},

        {% if value_columns|length > 1 -%}
        make_zip_iterator(make_tuple{{ value_columns|length }}(
        {%- endif -%}
        {%- for v in value_columns %}
        values{{ loop.index }}._vector.begin()
        {%- if not loop.last %}, {% endif -%}
        {% endfor -%}
        {% if value_columns|length > 1 -%}
        ))
        {%- endif %})
{% else %}
def sort_func(
        {% for c in key_columns -%}
        {{ df.get_dtype(c).__name__.title() }}View keys{{ loop.index }}
        {%- if not loop.last %},{% endif %}  # {{ c }}
        {% endfor -%}):

    c_sort(
        {% if key_columns|length > 1 -%}
        make_zip_iterator(make_tuple{{ key_columns|length }}(
        {%- endif %}
        {% for k in key_columns -%}
        keys{{ loop.index }}._vector.begin()
        {%- if not loop.last %}, {% endif -%}
        {% endfor -%}
        {% if key_columns|length > 1 -%}
        ))
        {%- endif %},
        {% if key_columns|length > 1 -%}
        make_zip_iterator(make_tuple{{ key_columns|length }}(
        {%- endif %}
        {% for k in key_columns -%}
        keys{{ loop.index }}._vector.end()
        {%- if not loop.last %}, {% endif -%}
        {% endfor -%}
        {% if key_columns|length > 1 -%}
        ))
        {%- endif %})
{%- endif %}
'''
