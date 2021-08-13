with import <nixpkgs> {};

let
  cmdstanpy = python38.pkgs.buildPythonPackage rec {
    pname = "cmdstanpy";
    version = "0.9.76";

    src = python38.pkgs.fetchPypi {
      inherit pname version;
      sha256 = "17y8mkqgxfcbc7l2fcyx4nqimrx2jzymyp6p74rfywfq9d22zk2n";
    };

    doCheck = false;

    propagatedBuildInputs = with python38.pkgs; [
      numpy
      cython  
      ujson
      pandas
    ];
  };

  pystan = python38.pkgs.buildPythonPackage rec {
    pname = "pystan";
    version = "2.19.1.1";

    src = python38.pkgs.fetchPypi {
      inherit pname version;
      sha256 = "0f5hbv9dhsx3b5yn5kpq5pwi1kxzmg4mdbrndyz2p8hdpj6sv2zs";
    };

    doCheck = false;

    propagatedBuildInputs = with python38.pkgs; [
      numpy
      cython  
      pandas
    ];
  };


  pythonEnv = python38.withPackages (ps: [
    ps.cython
    ps.numpy
    ps.pandas
    ps.jupyterlab
    ps.altair
    ps.matplotlib
    ps.tqdm
    ps.ipywidgets
  ]);

in mkShell {
  buildInputs = [
    pythonEnv
    pystan
    pandoc
    texlive.combined.scheme-full
  ];
}


