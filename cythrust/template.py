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


# The following template serves as a base for implementing a custom function
# accepting one or more `DeviceVectorView` arguments.
BASE_TEMPLATE = '''
from libc.stdint cimport (int8_t, int16_t, int32_t, int64_t,
                          uint8_t, uint16_t, uint32_t, uint64_t)
from cythrust.thrust.iterator.zip_iterator cimport make_zip_iterator
from cythrust.thrust.tuple cimport
{%- for i in range(2, 8) %} make_tuple{{ i }}
{%- if not loop.last %},{% endif -%}
{% endfor %}
{% for m in module_names -%}
from {{ m }}.device_vector cimport DeviceVectorView as {{ dtypes[loop.index0].__name__.title() }}View
{% endfor %}

def {% if func_name %}{{ func_name }}{% else %}__foo__{% endif %} (
        {{ preargs }}
        {%- for d in dtypes -%}
        {{ d.__name__.title() }}View {% if view_names %}{{ view_names[loop.index0] }}{% else %}view{{ loop.index }}{% endif %}
        {%- if not loop.last %},{% endif %}
        {% endfor -%}
        {{ postargs }}):
    pass'''


REDUCE_BY_KEY_TEMPLATE = '''
from cython.operator cimport dereference as deref, preincrement as inc
from libc.stdint cimport (int8_t, int16_t, int32_t, int64_t,
                          uint8_t, uint16_t, uint32_t, uint64_t)
from cythrust.thrust.reduce cimport (reduce as c_reduce,
                                     accumulate_by_key as c_accumulate_by_key,
                                     reduce_by_key as c_reduce_by_key)
from cythrust.thrust.iterator.zip_iterator cimport make_zip_iterator, zip_iterator

# Input type imports
{% for m in key_modules -%}
from {{ m }}.device_vector cimport DeviceVectorView as {{ key_dtypes[loop.index0].__name__.title() }}View
from {{ m }}.device_vector cimport Value as {{ key_dtypes[loop.index0].__name__.title() }}Value
from {{ m }}.device_vector cimport Iterator as {{ key_dtypes[loop.index0].__name__.title() }}Iterator
{% endfor %}
{% for m in value_modules -%}
from {{ m }}.device_vector cimport DeviceVectorView as {{ value_dtypes[loop.index0].__name__.title() }}View
from {{ m }}.device_vector cimport Value as {{ value_dtypes[loop.index0].__name__.title() }}Value
from {{ m }}.device_vector cimport Iterator as {{ value_dtypes[loop.index0].__name__.title() }}Iterator
{% endfor %}
{%- if key_modules|length > 1 %}
from cythrust.thrust.tuple cimport make_tuple{{ key_modules|length }}, tuple{{ key_modules|length }}  # keys
{%- endif %}
{%- if value_modules|length > 1 %}
from cythrust.thrust.tuple cimport make_tuple{{ value_modules|length }}, tuple{{ value_modules|length }}  # values
{%- endif %}

# Output type imports
{% for m in key_out_modules -%}
from {{ m }}.device_vector cimport DeviceVectorView as {{ key_out_dtypes[loop.index0].__name__.title() }}View
from {{ m }}.device_vector cimport Value as {{ key_out_dtypes[loop.index0].__name__.title() }}Value
from {{ m }}.device_vector cimport Iterator as {{ key_out_dtypes[loop.index0].__name__.title() }}Iterator
{% endfor %}
{% for m in value_out_modules -%}
from {{ m }}.device_vector cimport DeviceVectorView as {{ value_out_dtypes[loop.index0].__name__.title() }}View
from {{ m }}.device_vector cimport Value as {{ value_out_dtypes[loop.index0].__name__.title() }}Value
from {{ m }}.device_vector cimport Iterator as {{ value_out_dtypes[loop.index0].__name__.title() }}Iterator
{% endfor %}
{%- if key_out_modules|length > 1 %}
from cythrust.thrust.tuple cimport make_tuple{{ key_out_modules|length }}, tuple{{ key_out_modules|length }}  # keys
{%- endif %}
{%- if value_out_modules|length > 1 %}
from cythrust.thrust.tuple cimport make_tuple{{ value_out_modules|length }}, tuple{{ value_out_modules|length }}  # values
{%- endif %}


# Input iterator type defs
{% if key_modules|length > 1 %}
ctypedef tuple{{ key_modules|length }}[
{%- for d in key_dtypes -%}
{{ key_dtypes[loop.index0].__name__.title() }}Iterator
{%- if not loop.last %}, {% endif -%}
{% endfor -%}
] keys_tuple
ctypedef zip_iterator[keys_tuple] keys_iterator
{%- else -%}
ctypedef {{ key_dtypes[0].__name__.title() }}Iterator keys_iterator
{% endif %}

{%- if value_modules|length > 1 %}
ctypedef tuple{{ value_modules|length }}[
{%- for d in value_dtypes -%}
{{ value_dtypes[loop.index0].__name__.title() }}Iterator
{%- if not loop.last %}, {% endif -%}
{% endfor -%}
] values_tuple
ctypedef zip_iterator[values_tuple] values_iterator
{%- else -%}
ctypedef {{ value_dtypes[0].__name__.title() }}Iterator values_iterator
{%- endif %}

# Output iterator type defs
{% if key_out_modules|length > 1 %}
ctypedef tuple{{ key_out_modules|length }}[
{%- for d in key_out_dtypes -%}
{{ key_out_dtypes[loop.index0].__name__.title() }}Iterator
{%- if not loop.last %}, {% endif -%}
{% endfor -%}
] keys_out_tuple
ctypedef zip_iterator[keys_out_tuple] keys_out_iterator
{%- else -%}
ctypedef {{ key_out_dtypes[0].__name__.title() }}Iterator keys_out_iterator
{% endif %}

{%- if value_out_modules|length > 1 %}
ctypedef tuple{{ value_out_modules|length }}[
{%- for d in value_out_dtypes -%}
{{ value_out_dtypes[loop.index0].__name__.title() }}Iterator
{%- if not loop.last %}, {% endif -%}
{% endfor -%}
] values_out_tuple
ctypedef zip_iterator[values_out_tuple] values_out_iterator
{%- else -%}
ctypedef {{ value_out_dtypes[0].__name__.title() }}Iterator values_out_iterator
{%- endif %}


# Functors
from cythrust.thrust.functional cimport equal_to,
{%- for op in reduce_ops -%}
{{ reduce_ops[loop.index0] }}
{%- if not loop.last %}, {% endif -%}
{% endfor -%}
{%- if value_modules|length > 1 %}, reduce{{ value_modules|length }}{% endif %}


def reduce_by_key_func(
        {% for d in key_dtypes -%}
        {{ d.__name__.title() }}View keys{{ loop.index }},
        {% endfor -%}
        {% for d in value_dtypes -%}
        {{ d.__name__.title() }}View values{{ loop.index }},
        {% endfor -%}
        {% for d in key_out_dtypes -%}
        {{ d.__name__.title() }}View keys_out{{ loop.index }},
        {% endfor -%}
        {% for d in value_out_dtypes -%}
        {{ d.__name__.title() }}View values_out{{ loop.index }}
        {%- if not loop.last %},{% endif %}
        {% endfor -%}):
    {% if value_out_modules|length > 1 %}
    cdef reduce{{ value_out_modules|length }}[
{%- for c_type in value_ctypes -%}
{{ reduce_ops[loop.index0] }}[{{ c_type }}]
{%- if not loop.last %}, {% endif -%}
{% endfor -%}] *op = new reduce{{ value_out_modules|length }}[
{%- for c_type in value_ctypes -%}
{{ reduce_ops[loop.index0] }}[{{ c_type }}]
{%- if not loop.last %}, {% endif -%}
{% endfor -%}]()
    {% else %}
    cdef {{ reduce_ops[0] }}[{{ value_ctypes[0] }}] *op = new {{ reduce_ops[0] }}[{{ value_ctypes[0] }}]()
    {% endif %}

    {% if key_out_modules|length > 1 %}
    cdef equal_to[tuple{{ key_modules|length }}[
{%- for c_type in key_ctypes -%}
{{ c_type }}
{%- if not loop.last %}, {% endif -%}
{% endfor -%}]] *compare_op = new equal_to[tuple{{ key_modules|length }}[
{%- for c_type in key_ctypes -%}
{{ c_type }}
{%- if not loop.last %}, {% endif -%}
{% endfor -%}]]()
    {% else %}
    cdef equal_to[{{ key_ctypes[0] }}] *compare_op = new equal_to[{{ key_ctypes[0] }}]()
    {% endif %}

    {% if key_out_modules|length > 1 %}
    cdef keys_iterator keys_i_begin = make_zip_iterator(make_tuple{{ key_modules|length }}(
        {% for d in key_dtypes -%}
        <{{ key_dtypes[loop.index0].__name__.title() }}Iterator>keys{{ loop.index }}._begin
        {%- if not loop.last %},{% endif %}
        {% endfor -%}))
    cdef keys_iterator keys_i_end = make_zip_iterator(make_tuple{{ key_modules|length }}(
        {% for d in key_dtypes -%}
        <{{ key_dtypes[loop.index0].__name__.title() }}Iterator>keys{{ loop.index }}._end
        {%- if not loop.last %},{% endif %}
        {% endfor -%}))
    cdef keys_out_iterator keys_o_begin = make_zip_iterator(make_tuple{{ key_out_modules|length }}(
        {% for d in key_out_dtypes -%}
        <{{ key_out_dtypes[loop.index0].__name__.title() }}Iterator>keys_out{{ loop.index }}._begin
        {%- if not loop.last %},{% endif %}
        {% endfor -%}))
    cdef keys_out_iterator keys_o_end = make_zip_iterator(make_tuple{{ key_out_modules|length }}(
        {% for d in key_out_dtypes -%}
        <{{ key_out_dtypes[loop.index0].__name__.title() }}Iterator>keys_out{{ loop.index }}._end
        {%- if not loop.last %},{% endif %}
        {% endfor -%}))
    {% else %}
    cdef keys_iterator keys_i_begin = keys1._begin;
    cdef keys_iterator keys_i_end = keys1._end;
    cdef keys_out_iterator keys_o_begin = keys_out1._begin;
    cdef keys_out_iterator keys_o_end = keys_out1._end;
    {% endif %}

    {% if value_out_modules|length > 1 %}
    cdef values_iterator values_i_begin = make_zip_iterator(make_tuple{{ value_modules|length }}(
        {% for d in value_dtypes -%}
        <{{ value_dtypes[loop.index0].__name__.title() }}Iterator>values{{ loop.index }}._begin
        {%- if not loop.last %},{% endif %}
        {% endfor -%}))
    cdef values_iterator values_i_end = make_zip_iterator(make_tuple{{ value_modules|length }}(
        {% for d in value_dtypes -%}
        <{{ value_dtypes[loop.index0].__name__.title() }}Iterator>values{{ loop.index }}._end
        {%- if not loop.last %},{% endif %}
        {% endfor -%}))
    cdef values_out_iterator values_o_begin = make_zip_iterator(make_tuple{{ value_out_modules|length }}(
        {% for d in value_out_dtypes -%}
        <{{ value_out_dtypes[loop.index0].__name__.title() }}Iterator>values_out{{ loop.index }}._begin
        {%- if not loop.last %},{% endif %}
        {% endfor -%}))
    cdef values_out_iterator values_o_end = make_zip_iterator(make_tuple{{ value_out_modules|length }}(
        {% for d in value_out_dtypes -%}
        <{{ value_out_dtypes[loop.index0].__name__.title() }}Iterator>values_out{{ loop.index }}._end
        {%- if not loop.last %},{% endif %}
        {% endfor -%}))
    {% else %}
    cdef values_iterator values_i_begin = values1._begin;
    cdef values_iterator values_i_end = values1._end;
    cdef values_out_iterator values_o_begin = values_out1._begin;
    cdef values_out_iterator values_o_end = values_out1._end;
    {% endif %}

    cdef size_t N = <size_t>(
        <keys_iterator>(c_reduce_by_key(keys_i_begin, keys_i_end, values_i_begin,
                        keys_o_begin, values_o_begin,
                        deref(compare_op), deref(op)).first) - keys_o_begin)
    return N
'''


COUNT_TEMPLATE = '''
from cython.operator cimport dereference as deref, preincrement as inc
from libc.stdint cimport (int8_t, int16_t, int32_t, int64_t,
                          uint8_t, uint16_t, uint32_t, uint64_t)
from cythrust.thrust.reduce cimport (reduce as c_reduce,
                                     accumulate_by_key as c_accumulate_by_key,
                                     reduce_by_key as c_reduce_by_key)
from cythrust.thrust.iterator.zip_iterator cimport make_zip_iterator, zip_iterator
from cythrust.thrust.iterator.constant_iterator cimport make_constant_iterator, constant_iterator

# Input type imports
{% for m in key_modules -%}
from {{ m }}.device_vector cimport DeviceVectorView as {{ key_dtypes[loop.index0].__name__.title() }}View
from {{ m }}.device_vector cimport Value as {{ key_dtypes[loop.index0].__name__.title() }}Value
from {{ m }}.device_vector cimport Iterator as {{ key_dtypes[loop.index0].__name__.title() }}Iterator
{% endfor %}
{%- if key_modules|length > 1 %}
from cythrust.thrust.tuple cimport make_tuple{{ key_modules|length }}, tuple{{ key_modules|length }}  # keys
{%- endif %}
{%- if value_modules|length > 1 %}
from cythrust.thrust.tuple cimport make_tuple{{ value_modules|length }}, tuple{{ value_modules|length }}  # values
{%- endif %}

# Output type imports
{% for m in key_out_modules -%}
from {{ m }}.device_vector cimport DeviceVectorView as {{ key_out_dtypes[loop.index0].__name__.title() }}View
from {{ m }}.device_vector cimport Value as {{ key_out_dtypes[loop.index0].__name__.title() }}Value
from {{ m }}.device_vector cimport Iterator as {{ key_out_dtypes[loop.index0].__name__.title() }}Iterator
{% endfor %}
{% for m in value_out_modules -%}
from {{ m }}.device_vector cimport DeviceVectorView as {{ value_out_dtypes[loop.index0].__name__.title() }}View
from {{ m }}.device_vector cimport Value as {{ value_out_dtypes[loop.index0].__name__.title() }}Value
from {{ m }}.device_vector cimport Iterator as {{ value_out_dtypes[loop.index0].__name__.title() }}Iterator
{% endfor %}
{%- if key_out_modules|length > 1 %}
from cythrust.thrust.tuple cimport make_tuple{{ key_out_modules|length }}, tuple{{ key_out_modules|length }}  # keys
{%- endif %}
{%- if value_out_modules|length > 1 %}
from cythrust.thrust.tuple cimport make_tuple{{ value_out_modules|length }}, tuple{{ value_out_modules|length }}  # values
{%- endif %}


# Input iterator type defs
{% if key_modules|length > 1 %}
ctypedef tuple{{ key_modules|length }}[
{%- for d in key_dtypes -%}
{{ key_dtypes[loop.index0].__name__.title() }}Iterator
{%- if not loop.last %}, {% endif -%}
{% endfor -%}
] keys_tuple
ctypedef zip_iterator[keys_tuple] keys_iterator
{% else -%}
ctypedef {{ key_dtypes[0].__name__.title() }}Iterator keys_iterator
{% endif %}

# Output iterator type defs
{% if key_out_modules|length > 1 %}
ctypedef tuple{{ key_out_modules|length }}[
{%- for d in key_out_dtypes -%}
{{ key_out_dtypes[loop.index0].__name__.title() }}Iterator
{%- if not loop.last %}, {% endif -%}
{% endfor -%}
] keys_out_tuple
ctypedef zip_iterator[keys_out_tuple] keys_out_iterator
{% else -%}
ctypedef {{ key_out_dtypes[0].__name__.title() }}Iterator keys_out_iterator
{% endif %}

{%- if value_out_modules|length > 1 %}
ctypedef tuple{{ value_out_modules|length }}[
{%- for d in value_out_dtypes -%}
{{ value_out_dtypes[loop.index0].__name__.title() }}Iterator
{%- if not loop.last %}, {% endif -%}
{% endfor -%}
] values_out_tuple
ctypedef zip_iterator[values_out_tuple] values_out_iterator
{%- else -%}
ctypedef {{ value_out_dtypes[0].__name__.title() }}Iterator values_out_iterator
{%- endif %}


# Functors
from cythrust.thrust.functional cimport equal_to, plus


def count_by_key_func(
        {% for d in key_dtypes -%}
        {{ d.__name__.title() }}View keys{{ loop.index }},
        {% endfor -%}
        {% for d in key_out_dtypes -%}
        {{ d.__name__.title() }}View keys_out{{ loop.index }},
        {% endfor -%}
        {% for d in value_out_dtypes -%}
        {{ d.__name__.title() }}View values_out{{ loop.index }}
        {%- if not loop.last %},{% endif %}
        {% endfor -%}):
    cdef plus[{{ value_out_ctypes[0] }}] *op = new plus[{{ value_out_ctypes[0] }}]()

    {% if key_out_modules|length > 1 %}
    cdef equal_to[tuple{{ key_modules|length }}[
{%- for c_type in key_ctypes -%}
{{ c_type }}
{%- if not loop.last %}, {% endif -%}
{% endfor -%}]] *compare_op = new equal_to[tuple{{ key_modules|length }}[
{%- for c_type in key_ctypes -%}
{{ c_type }}
{%- if not loop.last %}, {% endif -%}
{% endfor -%}]]()
    {% else %}
    cdef equal_to[{{ key_ctypes[0] }}] *compare_op = new equal_to[{{ key_ctypes[0] }}]()
    {% endif %}

    {% if key_out_modules|length > 1 %}
    cdef keys_iterator keys_i_begin = make_zip_iterator(make_tuple{{ key_modules|length }}(
        {% for d in key_dtypes -%}
        <{{ key_dtypes[loop.index0].__name__.title() }}Iterator>keys{{ loop.index }}._begin
        {%- if not loop.last %},{% endif %}
        {% endfor -%}))
    cdef keys_iterator keys_i_end = make_zip_iterator(make_tuple{{ key_modules|length }}(
        {% for d in key_dtypes -%}
        <{{ key_dtypes[loop.index0].__name__.title() }}Iterator>keys{{ loop.index }}._end
        {%- if not loop.last %},{% endif %}
        {% endfor -%}))
    cdef keys_out_iterator keys_o_begin = make_zip_iterator(make_tuple{{ key_out_modules|length }}(
        {% for d in key_out_dtypes -%}
        <{{ key_out_dtypes[loop.index0].__name__.title() }}Iterator>keys_out{{ loop.index }}._begin
        {%- if not loop.last %},{% endif %}
        {% endfor -%}))
    cdef keys_out_iterator keys_o_end = make_zip_iterator(make_tuple{{ key_out_modules|length }}(
        {% for d in key_out_dtypes -%}
        <{{ key_out_dtypes[loop.index0].__name__.title() }}Iterator>keys_out{{ loop.index }}._end
        {%- if not loop.last %},{% endif %}
        {% endfor -%}))
    {% else %}
    cdef keys_iterator keys_i_begin = keys1._begin;
    cdef keys_iterator keys_i_end = keys1._end;
    cdef keys_out_iterator keys_o_begin = keys_out1._begin;
    cdef keys_out_iterator keys_o_end = keys_out1._end;
    {% endif %}

    cdef constant_iterator[{{ value_out_ctypes[0] }}] values_i_begin = constant_iterator[{{ value_out_ctypes[0] }}](1)

    {% if value_out_modules|length > 1 %}
    cdef values_out_iterator values_o_begin = make_zip_iterator(make_tuple{{ value_out_modules|length }}(
        {% for d in value_out_dtypes -%}
        <{{ value_out_dtypes[loop.index0].__name__.title() }}Iterator>values_out{{ loop.index }}._begin
        {%- if not loop.last %},{% endif %}
        {% endfor -%}))
    cdef values_out_iterator values_o_end = make_zip_iterator(make_tuple{{ value_out_modules|length }}(
        {% for d in value_out_dtypes -%}
        <{{ value_out_dtypes[loop.index0].__name__.title() }}Iterator>values_out{{ loop.index }}._end
        {%- if not loop.last %},{% endif %}
        {% endfor -%}))
    {% else %}
    cdef values_out_iterator values_o_begin = values_out1._begin;
    cdef values_out_iterator values_o_end = values_out1._end;
    {% endif %}

    cdef size_t N = <size_t>(
        <keys_iterator>(c_reduce_by_key(keys_i_begin, keys_i_end,
                                        values_i_begin,
                                        keys_o_begin, values_o_begin,
                                        deref(compare_op), deref(op)).first) -
                                        keys_o_begin)
    return N
'''

TRANSFORM_SETUP_TEMPLATE = '''
{% for t in transforms %}
from {{ t.functor_name }}.{{ t.functor_name }} cimport {{ t.functor_name }}
{% endfor %}
from cythrust.thrust.iterator.zip_iterator cimport make_zip_iterator, zip_iterator
from cythrust.thrust.tuple cimport (make_tuple2, make_tuple3, make_tuple4,
                                    make_tuple5, make_tuple6, make_tuple7,
                                    make_tuple8, make_tuple9, tuple2, tuple3,
                                    tuple4, tuple5, tuple6, tuple7, tuple8,
                                    tuple9)
from cythrust.thrust.copy cimport copy_n

ctypedef {% if transforms|length > 1 %}tuple{{ transforms|length }}[{% endif %}
{%- for t in transforms -%}
{{ t.functor_name }}.iterator
{%- if not loop.last %}, {% endif -%}
{% endfor -%}
{% if transforms|length > 1 %}]{% endif %} input_tuple
ctypedef zip_iterator[input_tuple] input_iterator
'''

TRANSFORM_TEMPLATE = '''
{% for t in transforms %}
    cdef {{ t.functor_name }} *op{{ loop.index }} = new {{ t.functor_name }}(
    {%- for column in t.thrust_code.graph_inputs -%}
    <{{ t.functor_name }}.{{ column }}_t>{{ column }}._begin
    {%- if not loop.last %}, {% endif -%}
    {% endfor %})
{% endfor %}

    cdef size_t N = {{ out_views[0] }}._end - {{ out_views[0] }}._begin

    copy_n(
    {% if transforms|length > 1 %}make_zip_iterator(make_tuple{{ transforms|length }}({% endif %}
{%- for t in transforms -%}
    op{{ loop.index }}.begin()
{%- if not loop.last %}, {% endif -%}
{% endfor -%}
{% if transforms|length > 1 %})){% endif %},
    N,
    {% if out_views|length > 1 %}make_zip_iterator(make_tuple{{ out_views|length }}({% endif %}
{%- for v in out_views -%}
    {{ v }}._begin
{%- if not loop.last %}, {% endif -%}
{% endfor -%}
{% if out_views|length > 1 %})){% endif %})
    return N
'''

SCATTER_SETUP_TEMPLATE = '''
from cythrust.thrust.iterator.zip_iterator cimport make_zip_iterator, zip_iterator
from cythrust.thrust.tuple cimport (make_tuple2, make_tuple3, make_tuple4,
                                    make_tuple5, make_tuple6, make_tuple7,
                                    make_tuple8, make_tuple9, tuple2, tuple3,
                                    tuple4, tuple5, tuple6, tuple7, tuple8,
                                    tuple9)
from cythrust.thrust.copy cimport copy_n

{% for k, transforms in (('in', transforms_in), ('out', transforms_out)) %}
{% for t in transforms %}
from {{ t.functor_name }}.{{ t.functor_name }} cimport {{ t.functor_name }}
{% endfor %}
ctypedef {% if transforms|length > 1 %}tuple{{ transforms|length }}[{% endif %}
{%- for t in transforms -%}
{{ t.functor_name }}.iterator
{%- if not loop.last %}, {% endif -%}
{% endfor -%}
{% if transforms|length > 1 %}]{% endif %} {{ k }}put_tuple
ctypedef zip_iterator[{{ k }}put_tuple] {{ k }}put_iterator
{% endfor %}
'''

SCATTER_TEMPLATE = '''
{%- for k, transforms in (('in', transforms_in), ('out', transforms_out)) -%}
{% for t in transforms %}
    cdef {{ t.functor_name }} *op_{{ k }}{{ loop.index }} = new {{ t.functor_name }}(
    {%- for column in t.thrust_code.graph_inputs -%}
    <{{ t.functor_name }}.{{ column }}_t>{{ column }}._begin
    {%- if not loop.last %}, {% endif -%}
    {% endfor %})
{% endfor %}
{% endfor %}

    copy_n(
{% if transforms_in|length > 1 %}make_zip_iterator(make_tuple{{ transforms_in|length }}({% endif %}
{%- for t in transforms_in -%}
    op_in{{ loop.index }}.begin()
{%- if not loop.last %}, {% endif -%}
{% endfor -%}
{% if transforms_in|length > 1 %})){% endif %},
    N,
{% if transforms_out|length > 1 %}make_zip_iterator(make_tuple{{ transforms_out|length }}({% endif %}
{%- for t in transforms_out -%}
    op_out{{ loop.index }}.begin()
{%- if not loop.last %}, {% endif -%}
{% endfor -%}
{% if transforms_out|length > 1 %})){% endif %})
    return N
'''

REDUCE_SETUP_TEMPLATE = '''
import pandas as pd
from cython.operator cimport dereference as deref, preincrement as inc
from libc.stdint cimport (int8_t, int16_t, int32_t, int64_t,
                          uint8_t, uint16_t, uint32_t, uint64_t)
from cythrust.thrust.iterator.zip_iterator cimport make_zip_iterator, zip_iterator
from cythrust.thrust.tuple cimport (make_tuple2, make_tuple3, make_tuple4,
                                    make_tuple5, make_tuple6, make_tuple7,
                                    make_tuple8, make_tuple9, tuple2, tuple3,
                                    tuple4, tuple5, tuple6, tuple7, tuple8,
                                    tuple9)
from cythrust.thrust.reduce cimport reduce_n
from cythrust.thrust.copy cimport copy_n

{% for t in transforms %}
from {{ t.functor_name }}.{{ t.functor_name }} cimport {{ t.functor_name }}
{% endfor %}
ctypedef {% if transforms|length > 1 %}tuple{{ transforms|length }}[{% endif %}
{%- for t in transforms -%}
{{ t.functor_name }}.iterator
{%- if not loop.last %}, {% endif -%}
{% endfor -%}
{% if transforms|length > 1 %}]{% endif %} input_tuple
ctypedef zip_iterator[input_tuple] input_iterator

# Functors
from cythrust.thrust.functional cimport {% for name in named_positions -%}
{{ name }},
{%- endfor %} {% for op in reduce_ops -%}
{{ op }}
{%- if not loop.last %}, {% endif -%}
{% endfor -%}
{%- if transforms|length > 1 %}, reduce{{ transforms|length }}{% endif %}


ctypedef {% if transforms|length > 1 %}tuple{{ transforms|length }}[{% endif %}
{%- for t in transforms %}{{ t.dfg.operation_graph.owner.out.dtype }}_t
    {%- if not loop.last %}, {% endif -%}
{% endfor %}
{%- if transforms|length > 1 %}]{% endif %} result_type
'''

REDUCE_TEMPLATE = '''
{% for t in transforms %}
    cdef {{ t.functor_name }} *op_{{ loop.index }} = new {{ t.functor_name }}(
    {%- for column in t.thrust_code.graph_inputs -%}
    <{{ t.functor_name }}.{{ column }}_t>{{ column }}._begin
    {%- if not loop.last %}, {% endif -%}
    {% endfor %})
{% endfor %}

    cdef
{%- if transforms|length > 1 %} reduce{{ transforms|length }}[{% endif %}
{%- for t in transforms %} {{ reduce_ops[loop.index0] }}[{{ t.dfg.operation_graph.owner.out.dtype }}_t]
    {%- if not loop.last %}, {% endif -%}
{% endfor %}
{%- if transforms|length > 1 %}]{% endif %} *reduce_op = new
{%- if transforms|length > 1 %} reduce{{ transforms|length }}[{% endif %}
{%- for t in transforms %} {{ reduce_ops[loop.index0] }}[{{ t.dfg.operation_graph.owner.out.dtype }}_t]
    {%- if not loop.last %}, {% endif -%}
{% endfor %}
{%- if transforms|length > 1 %}]{% endif %}()

    cdef result_type init_values =
    {%- if init_values|length > 1 %} result_type({% endif %}
    {%- for v in init_values %}{{ v }}
        {%- if not loop.last %}, {% endif -%}
    {% endfor %}
    {%- if transforms|length > 1 %}){% endif %}

    cdef result_type result = reduce_n(
{%- if init_values|length > 1 %} make_zip_iterator(make_tuple{{ transforms|length }}({% endif %}
{%- for v in init_values %} op_{{ loop.index }}.begin()
    {%- if not loop.last %}, {% endif -%}
{% endfor %}
{%- if init_values|length > 1 %})){% endif %}, N,
{%- if init_values|length > 1 %} result_type({% endif %}
{%- for v in init_values %}{{ v }}
    {%- if not loop.last %}, {% endif -%}
{% endfor %}
{%- if transforms|length > 1 %}){% endif %}, deref(reduce_op))

    del op_1
    del reduce_op

    {% if transforms|length > 1 %}
    return_vals = []
    {% for t in transforms %}
    cdef {{ named_positions[loop.index0] }}[{{ t.dfg.operation_graph.owner.out.dtype }}_t] *{{ named_positions[loop.index0] }}_ = new {{ named_positions[loop.index0] }}[{{ t.dfg.operation_graph.owner.out.dtype }}_t]()
    return_vals.append(deref({{ named_positions[loop.index0] }}_)(result))
    del {{ named_positions[loop.index0] }}_
    {% endfor %}
    return return_vals
    {% else %}
    return result
    {% endif %}
'''
