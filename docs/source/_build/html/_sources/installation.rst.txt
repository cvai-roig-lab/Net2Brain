==============
Installation
==============

Compatibility and System Requirements
=====================================

Net2Brain is thoroughly tested on systems running Python versions up to 3.10 and is compatible with Windows, macOS, and Linux operating systems.

Please note that netsets, such as Detectron2 and VISSL, are designed specifically for Linux-based systems. Detailed installation instructions for these netsets are available within the provided notebooks.

Installation with Conda
=======================

To create a Conda environment and install Net2Brain along with its dependencies, follow the steps below:

1. Open your terminal or command prompt.

2. If desired, create a new conda environment:

   .. code-block:: bash

      conda create --name net2brain_env # python=3.10 (optional)
      conda activate net2brain_env

3. With the Conda environment activated, install the Net2Brain repository:

   .. code-block:: bash

      pip install -U git+https://github.com/cvai-roig-lab/Net2Brain

4. After the installation is complete, you can import Net2Brain into your Python environment:

   .. code-block:: python

      import net2brain

Now you are ready to explore the functionalities of Net2Brain for your neural research projects within a dedicated Conda environment.

Troubleshooting
---------------

If you encounter any installation issues:

- Verify that you have Python 3.10 or lower installed on your system.
- Ensure that pip is updated to the latest version within the Conda environment.
- Check if your system meets the requirements for specific netsets like Detectron2 and VISSL if you intend to use them.
- For further assistance, refer to the [Net2Brain documentation](https://net2brain.readthedocs.io/) or raise an issue on the [Net2Brain GitHub repository](https://github.com/cvai-roig-lab/Net2Brain/issues).
