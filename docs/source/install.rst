Installation
============

.. contents:: On This Page
   :local:
   :depth: 2

pip
--------------

.. code-block:: bash

   pip install aub-htp

uv
--

`Install uv <https://docs.astral.sh/uv/>`_, then:

.. code-block:: bash

   uv init myproject && cd myproject
   uv add aub-htp

conda
-----

`Install Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_, then:

.. code-block:: bash

   conda install -c conda-forge aub-htp

poetry
------

`Install Poetry <https://python-poetry.org/docs/>`_, then:

.. code-block:: bash

   poetry new myproject && cd myproject
   poetry add aub-htp

Development install
--------------------

You can also install the development version and build ``aub-htp`` from the source code. This approach is
favored among those who wish to modify the package's code. 

.. code-block:: bash

   git clone https://github.com/AUB-HTP/AUB-HTP
   cd AUB-HTP
   pip install .
