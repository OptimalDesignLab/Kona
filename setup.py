from distutils.core import setup

setup(name = 'Kona',
      version = '1.0',
      author = 'Jason E. Hicken',
      author_email = 'hickej2@rpi.edu',
      url = 'https://github.com/OptimalDesignLab/Kona',
      package_dir = {'kona':''},
      packages = ['kona', 'kona.examples']
      )