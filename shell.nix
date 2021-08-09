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
    cmdstanpy
  ];
}


