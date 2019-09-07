# Xarray File Utilities

from pathlib import Path
import numpy as np
import xarray as xr

INTERMEDIATE_DATA_PATH = Path('/efs/shared/dataDrivenRamen')


def _handle_data_path(data_name: str, session: str, dir_path=None, make_parents=False):

    """ Modular Function to Resolve the path to the file to either be opened or saved

    Parameters
    ----------
    data_name : str
    session : str
    session : str
    dir_path :  pathlib.Path, optional
        Destination of data to be save other than the default intermediate location
    make_parents : bool, optional
        if True, it will create all of the parent folders for the Data File
    Returns
    -------
    data_file_path : Path()
        The Resolved Path to the file designated by User given parameters

    """

    # Handle Use Cases of the dir_path variable
    if dir_path is not None:
        if isinstance(dir_path, str):
            data_path = Path(dir_path)  # Make a Path instance
        elif isinstance(dir_path, Path):
            data_path = dir_path  # already a Path instance
        else:
            raise TypeError  # Currently only support str and Path()
    else:
        if not INTERMEDIATE_DATA_PATH.exists():
            INTERMEDIATE_DATA_PATH.mkdir(parents=False, exist_ok=True)  # Makes the default intermediate data folder
        data_path = INTERMEDIATE_DATA_PATH

    # Make the Path to the Data File
    data_file_path = data_path / 'ephys_data' / session / data_name  # Note: data_name is the File Header

    if make_parents:
        # Check if the usual data hierarchical structure is there (creates it if it isn't)
        if not data_file_path.parents[0].exists():
            data_file_path.parents[0].mkdir(parents=True, exist_ok=True)

    # data_file_path.parents[0].resolve()
    assert data_file_path.parents[0].exists(), "{data_file_path} doesn't exist".format(data_file_path=data_file_path)
    # f"{data_file_path} doesn't exist"  # Replace above once py3.5 support dropped

    return data_file_path



def _save_xarray_data(data: xr.core.dataarray.DataArray, data_name: str, session: str, destination=None,
                      make_parents=False, verbose = False):
    # TODO: Add *Args to allow for identifier information to be appended to the name of the file

    """

    Parameters
    ----------
    data : ndarray
    data_name : str
    session : str
    destination: str, pathlib.Path, optional
        Destination of data to be saved
    make_parents: bool, optional
        if True, it will create all of the parent folders for the Data File

    """

    file_name = data_name + '.nc'  # Add the .nc stem to the data_name

    data_file_path = _handle_data_path(data_name=file_name, session=session, dir_path=destination,
                                       make_parents=make_parents)
    if verbose:
        # print(f"Saving {data_name} Data to", data_file_path.name)  # Uncomment once py3.5 support Dropped
        print("Saving {data_name} Data to {file_path}".format(data_name=data_name, file_path=data_file_path.name))

    data.to_netcdf(str(data_file_path))  # Save Data


def _load_xarray_data(data_name: str, session: str, source=None, verbose = False):
    # TODO: Add *Args to allow for identifier information to be appended to the name of the file

    """

    Parameters
    ----------
    data_name : str
    session : str
    source : str, pathlib.Path, optional
        Location of data to be loaded

    Returns
    -------
    np.load(data_file_path) : array, tuple, dict, etc.
        Data stored in the file

    """

    file_name = data_name + '.nc'

    data_file_path = _handle_data_path(data_name=file_name, session=session, dir_path=source,
                                       make_parents=False)

    if verbose:
        # print(f"Saving {data_name} Data to", data_file_path.name)  # Uncomment once py3.5 support Dropped
        print("Loading {data_name} Data to {file_path}".format(data_name=data_name, file_path=data_file_path.name))

    data_file_path = str(data_file_path)  # Convert Path to String for backwards compatibility

    return xr.load_dataarray(data_file_path)


def _save_numpy_data(data: np.ndarray, data_name: str, bird_id: str, session: str, destination=None, make_parents=False,
                     verbose = False):
    # TODO: Add *Args to allow for identifier information to be appended to the name of the file

    """

    Parameters
    ----------
    data : ndarray
    data_name : str
    bird_id : str
    session : str
    destination: str, pathlib.Path, optional
        Destination of data to be saved
    make_parents: bool, optional
        if True, it will create all of the parent folders for the Data File

    """

    data_file_path = _handle_data_path(data_name=data_name, region=bird_id, session=session, dir_path=destination,
                                       make_parents=make_parents)
    if verbose:
        # print(f"Saving {data_name} Data to", data_file_path.name)  # Uncomment once py3.5 support Dropped
        print("Saving {data_name} Data to {file_path}".format(data_name=data_name, file_path=data_file_path.name))

    np.save(str(data_file_path), data)  # Save Data


def _load_numpy_data(data_name: str, bird_id: str, session: str, source=None, verbose = False):
    # TODO: Add *Args to allow for identifier information to be appended to the name of the file

    """

    Parameters
    ----------
    data_name : str
    bird_id : str
    session : str
    source : str, pathlib.Path, optional
        Location of data to be loaded

    Returns
    -------
    np.load(data_file_path) : array, tuple, dict, etc.
        Data stored in the file

    """


    file_name = data_name + '.npy'

    data_file_path = _handle_data_path(data_name=file_name, region=bird_id, session=session, dir_path=source,
                                       make_parents=False)

    if verbose:
        # print(f"Saving {data_name} Data to", data_file_path.name)  # Uncomment once py3.5 support Dropped
        print("Saving {data_name} Data to {file_path}".format(data_name=data_name, file_path=data_file_path.name))

    data_file_path = str(data_file_path)  # Convert Path to String for backwards compatibility

    return np.load(data_file_path)
