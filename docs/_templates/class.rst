{{ name }}
{{ underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ fullname }}

   {% block methods %}

   {% if methods %}
   .. rubric:: Methods

   .. autosummary::
      :toctree:
      :recursive:
   {% for item in methods %}{% if item != '__init__' %}
      ~{{ name }}.{{ item }}
   {%- endif %}{% endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: Attributes

   .. autosummary::
      :toctree:
      :recursive:
   {% for item in attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
