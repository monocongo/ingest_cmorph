import argparse
from datetime import datetime
import logging
import netCDF4
import numpy as np
import warnings

#-----------------------------------------------------------------------------------------------------------------------
# set up a basic, global _logger which will write to the console as standard error
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d  %H:%M:%S')
_logger = logging.getLogger(__name__)

#-----------------------------------------------------------------------------------------------------------------------
# ignore warnings
warnings.simplefilter('ignore', Warning)

#-----------------------------------------------------------------------------------------------------------------------
def ingest_cmorph_to_netcdf(cmorph_file, 
                            descriptor_file,
                            netcdf_file):
    
    # read data description info
    ctl = _read_description(descriptor_file)
    
    # create a corresponding NetCDF
    with netCDF4.Dataset(netcdf_file, 'w') as output_dataset:
        
        # create the time, x, and y dimensions
        output_dataset.createDimension('time', None)
        output_dataset.createDimension('lon', ctl['xdef_count'])
        output_dataset.createDimension('lat', ctl['ydef_count'])
    
        output_dataset.title = ctl['title']
        
        # create the coordinate variables
        time_variable = output_dataset.createVariable('time', 'i4', ('time',))
        x_variable = output_dataset.createVariable('lon', 'f4', ('lon',))
        y_variable = output_dataset.createVariable('lat', 'f4', ('lat',))
        
        # set the coordinate variables' attributes
        units_start_year = 1800
        time_variable.units = 'days since %s-01-01 00:00:00' % units_start_year
        x_variable.units = 'degrees east'
        y_variable.units = 'degrees north'
        
        # compute the time value 
        start_date = datetime(units_start_year, 1, 1)
        time_variable[:] = np.array([(ctl['date'] - start_date).days])
        
        # generate longitude and latitude values, assign these to the NetCDF coordinate variables
        lon_values = list(_frange(ctl['xdef_start'], ctl['xdef_start'] + (ctl['xdef_count'] * ctl['xdef_increment']), ctl['xdef_increment']))
        lat_values = list(_frange(ctl['ydef_start'], ctl['ydef_start'] + (ctl['ydef_count'] * ctl['ydef_increment']), ctl['ydef_increment']))
        x_variable[:] = np.array(lon_values, 'f4')
        y_variable[:] = np.array(lat_values, 'f4')
    
        # read the variable data from the CMORPH file, mask and reshape accordingly, and then assign into the variable
        data = np.fromfile(cmorph_file, 'f')
        if not ctl['little_endian']:
            data = data.byteswap()
        data = np.ma.masked_values(data, ctl['undef'])
        data_variable = output_dataset.createVariable('prcp', 
                                                      data.dtype, 
                                                      ('time', 'lon', 'lat',), 
                                                      fill_value=ctl['undef'])
        data_variable[:] = np.reshape(data, (1, ctl['xdef_count'], ctl['ydef_count']))
        
#-----------------------------------------------------------------------------------------------------------------------
def _frange(start, stop, step):
    i = start
    while i < stop:
        yield i
        i += step

#-----------------------------------------------------------------------------------------------------------------------
def _read_description(descriptor_file):
    """
    Reads a data descriptor file, example below:
    
        DSET ../0.25deg-DLY_00Z/%y4/%y4%m2/CMORPH_V1.0_RAW_0.25deg-DLY_00Z_%y4%m2%d2  
        TITLE  CMORPH Version 1.0BETA Version, daily precip from 00Z-24Z 
        OPTIONS template little_endian
        UNDEF  -999.0
        XDEF 1440 LINEAR    0.125  0.25
        YDEF  480 LINEAR  -59.875  0.25
        ZDEF   01 LEVELS 1
        TDEF 99999 LINEAR  01jan1998 1dy 
        VARS 1
        cmorph   1   99 yyyyy CMORPH Version 1.o daily precipitation (mm)  
        ENDVARS
        
    :param descriptor_file: ASCII file with data description information
    :return: dictionary of data description keys/values
    """
    
    data_dict = {}    
    with open(descriptor_file, 'r') as fp:
        for line in fp:
            words = line.split()
            if words[0] == 'UNDEF':
                data_dict['undef'] = float(words[1])
            elif words[0] == 'XDEF':
                data_dict['xdef_count'] = int(words[1])
                data_dict['xdef_start'] = float(words[3])
                data_dict['xdef_increment'] = float(words[4])
            elif words[0] == 'YDEF':
                data_dict['ydef_count'] = int(words[1])
                data_dict['ydef_start'] = float(words[3])
                data_dict['ydef_increment'] = float(words[4])
            elif words[0] == 'TDEF':
                data_dict['date'] = datetime.strptime(words[3], '%d%b%Y')  # example: "01jan1998"
            elif words[0] == 'OPTIONS':
                if words[2] == 'big_endian':
                    data_dict['little_endian'] = False
                else:   # assume words[2] == 'little_endian'
                    data_dict['little_endian'] = True
            elif words[0] == 'cmorph':  # looking for a line like this: "cmorph   1   99 yyyyy CMORPH Version 1.o daily precipitation (mm)"
                data_dict['variable_description'] = ' '.join(words[4:])
            elif words[0] == 'TITLE':
                data_dict['title'] = ' '.join(words[1:])

    return data_dict

#-----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    """
    example command line invocation:
    
    $ python -u ingest_cmorph.py --in_file C:/home/data/cmorph_file.cm \
                                 --descriptor_file C:/home/data/cmorph_file.ctl \
                                 --out_file C:/home/data/cmorph_file.nc

    This module is used to perform ingest of binary CMORPH datasets to NetCDF.
    """

    try:

        # log some timing info, used later for elapsed time
        start_datetime = datetime.now()
        _logger.info("Start time:    %s", start_datetime)

        # parse the command line arguments
        parser = argparse.ArgumentParser()
        parser.add_argument("--in_file", 
                            help="Binary CMORPH data file", 
                            required=True)
        parser.add_argument("--descriptor_file", 
                            help="Data descriptor file corresponding to the input binary CMORPH data file", 
                            required=True)
        parser.add_argument("--out_file", 
                            help="NetCDF output file containing variables read from the input data", 
                            required=True)
        args = parser.parse_args()

        # perform the ingest to NetCDF
        ingest_cmorph_to_netcdf(args.in_file, args.descriptor_file, args.out_file)

        # report on the elapsed time
        end_datetime = datetime.now()
        _logger.info("End time:      %s", end_datetime)
        elapsed = end_datetime - start_datetime
        _logger.info("Elapsed time:  %s", elapsed)

    except Exception as ex:
        _logger.exception('Failed to complete', exc_info=True)
        raise
    