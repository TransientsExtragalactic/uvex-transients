{{ name }}
{{ underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ fullname }}
   :members:
   :show-inheritance:

   {% block methods %}

   {% if methods %}
   .. rubric:: Methods

   .. autosummary::
      :nosignatures:
   {% for item in methods %}{% if item != '__init__' %}
      ~{{ name }}.{{ item }}
   {%- endif %}{% endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: Attributes

   .. autosummary::
      :nosignatures:
   {% for item in attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
