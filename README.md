<p>
    <a href='https://travis-ci.org/OptimalDesignLab/Kona?branch=master'>
        <img id=travis alt="Build Status" 
          src="https://travis-ci.org/OptimalDesignLab/Kona.svg?branch=master" 
          />
        <script language="javascript" type="text/javascript">
            var d = new Date(); 
            document.getElementById("travis").src = "https://travis-ci.org/OptimalDesignLab/Kona.svg?branch=master&ver=" + d.getTime();
        </script>
    </a>
    <a href='https://coveralls.io/r/OptimalDesignLab/Kona?branch=master'>
        <img alt='Coverage Status' id=coveralls 
        src='https://coveralls.io/repos/OptimalDesignLab/Kona/badge.svg?branch=master' 
        />
        <script language="javascript" type="text/javascript">
            var d = new Date(); 
            document.getElementById("coveralls").src = "https://coveralls.io/repos/OptimalDesignLab/Kona/badge.svg?branch=master&ver=" + d.getTime();
        </script>
    </a>
</p>

# Kona

Kona is an optimization library that implements a set of algorithms for 
PDE-constrained optimization.

An important aspect of its implementation is that it makes no assumptions 
regarding the dimension, type or parallelization of the control, state or 
constraint vectors. The user must implement these vectors and, through reverse 
communication, Kona asks the user to perform various operations on these 
vectors. This model allows Kona to be used in a variety of parallel 
environments, because it remains agnostic to how the user defines and stores 
these vectors.

Additionally, Kona separates optimization algorithms from the underlying PDE 
solver such that any new optimization algorithm can be implemented in Kona 
using Kona's own abstracted vector classes. This allows for rapid development 
and testing of new algorithms independently from the PDE solver, and 
guarantees that any solver that has already been integrated with Kona will 
work correctly underneath any new algorithm that may be added in the future.

An older version of Kona written in C++ can be found 
[here](https://bitbucket.org/odl/kona).
